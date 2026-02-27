import copy
import torch
import torch.distributed as dist
from einops import rearrange

from .base import BaseTrainer

class CQLTrainer(BaseTrainer):
    name = 'cql'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 train_sampler,
                 eval_loader,
                 eval_sampler,
                 logger, 
                 aug_func,
                 model):
        
        self.target_model = copy.deepcopy(model).to(device)   
        
        super().__init__(cfg, 
                         device, 
                         train_loader, 
                         train_sampler,
                         eval_loader, 
                         eval_sampler, 
                         logger, 
                         aug_func, 
                         model)          
        
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.gamma = cfg.gamma
        self.n_step = cfg.n_step
        self.feature_normalization = cfg.pretrain.feature_normalization
        
        # distributional
        self.num_atoms = self.cfg.pretrain.num_atoms
        self.v_min = self.cfg.pretrain.v_min
        self.v_max = self.cfg.pretrain.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # cql
        self.cql_coefficient = self.cfg.pretrain.cql_coefficient
        
        if self.cfg.pretrain.compile:
            self.target_model = torch.compile(self.target_model)

    def compute_loss(self, batch):
        ##############
        # forward
        obs = batch['obs']    
        next_obs = batch['next_obs']    
        act = batch['act']
        rew = batch['rew']
        done = batch['done'].type(torch.int64)
        rtg = batch['rtg']
        game_id = batch['game_id']
        
        # augmentation
        n, t, f, c, h, w = obs.shape
        x, nx = obs, next_obs
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        nx = rearrange(nx, 'n t f c h w -> n (t f c) h w')
        x, nx = self.aug_func(x), self.aug_func(nx)
        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        nx = rearrange(nx, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online encoder
        x, _ = self.model.backbone(x)
        x, _ = self.model.neck(x, game_id) # game-wise spatial embedding
        if self.feature_normalization:
            x = x/torch.linalg.norm(x,dim=-1).unsqueeze(2)
        q_dist, _ = self.model.head(x, game_id) # game-wise prediction head
        q_dist = rearrange(q_dist, 'n t d n_a -> (n t) d n_a')
        
        # KL-divergence
        act_idx = act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
        log_pred_q_dist = (q_dist.log()).gather(1, act_idx).squeeze(1)
        
        with torch.no_grad():
            # Calculate n-th next state's q-value distribution
            # next_target_q_dist: (n, 1, a, num_atoms)
            # target_q_dist: (n, num_atoms)
            nx, _ = self.target_model.backbone(nx)
            nx, _ = self.target_model.neck(nx, game_id)
            if self.feature_normalization:
                nx = nx/torch.linalg.norm(nx,dim=-1).unsqueeze(2)
            next_target_q_dist, _ = self.target_model.head(nx, game_id)
            
            next_target_q_dist = rearrange(next_target_q_dist, 'n t d n_a -> (n t) d n_a')
            next_target_q =  (next_target_q_dist * self.support.reshape(1,1,-1)).sum(-1)     
            next_act = torch.argmax(next_target_q, 1)  
            next_act_idx = next_act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
            target_q_dist = (next_target_q_dist).gather(1, next_act_idx).squeeze(1)
        
            # C51 (https://arxiv.org/abs/1707.06887, Algorithm 1)
            # Compute the projection 
            # Tz = R_n + (γ^n)Z (w/ n-step return) (N, N_A)
            gamma_n = (self.gamma ** self.n_step)
            Tz = rtg + gamma_n * self.support.unsqueeze(0) * (1-done)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            # L2-projection
            b = (Tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros((n, self.num_atoms), device=self.device)
            for idx in range(n):
                # += operation do not allow to add value to same index multiple times
                m[idx].index_add_(0, l[idx], target_q_dist[idx] * (u[idx] - b[idx]))
                m[idx].index_add_(0, u[idx], target_q_dist[idx] * (b[idx] - l[idx]))
        
        # kl-divergence KL(p||q)=plogp-plogq
        # Here, plogp is just a constant
        EPS = 1e-5
        kl_div = -torch.sum(m * log_pred_q_dist, -1)
        kl_div = (kl_div.clamp(min=EPS, max=1 / EPS)).mean()
        
        # cql
        q_values = (q_dist * self.support.reshape(1,1,-1)).sum(-1)
        chosen_action_q = q_values.gather(1, act.reshape(-1,1)).squeeze(1)
        
        cql_loss = (torch.logsumexp(q_values, 1) - chosen_action_q).mean()
        loss = kl_div + self.cql_coefficient * cql_loss
        
        ###############
        # logs        
        log_data = {'loss': loss.item(),
                    'kl_loss': kl_div.item(),
                    'cql_loss': cql_loss.item()}
        
        return loss, log_data
    
    def update(self, batch, step):
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = self.cfg.pretrain.target_tau * target.data + (1 - self.cfg.pretrain.target_tau) * online.data

    def save_checkpoint(self, epoch, step, name=None):
        # Save only once
        if self.logger.rank == 0:
            if name is None:
                name = 'epoch'+str(epoch)
            save_dict = {'backbone': self.model.backbone.state_dict(),
                          'neck': self.model.neck.state_dict(),
                          'head': self.model.head.state_dict(),
                          'target_backbone': self.target_model.backbone.state_dict(),
                          'target_neck': self.target_model.neck.state_dict(),
                          'target_head': self.target_model.head.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'lr_scheduler': self.lr_scheduler.state_dict(),
                          'scaler': self.scaler.state_dict(),
                          'epoch': epoch,
                          'step': step,}
            self.logger.save_dict(save_dict=save_dict,
                                  name=name)
