import copy
import torch
import torch.distributed as dist
from einops import rearrange
from dataclasses import asdict

from .base import BaseTrainer
from src.utils.schedulers import LinearScheduler


class SPRTrainer(BaseTrainer):
    name = 'spr'
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
    
        # Remember to deepcopy PRIOR to compile/DDP, DDP seems to hate copying from already compiled models
        self.target_backbone = copy.deepcopy(model.backbone).to(device)
        self.target_neck = copy.deepcopy(model.neck).to(device)

        super().__init__(cfg, 
                         device, 
                         train_loader, 
                         train_sampler,
                         eval_loader, 
                         eval_sampler, 
                         logger, 
                         aug_func, 
                         model)

        total_steps = len(self.train_loader) * self.cfg.pretrain.num_epochs
        cfg.tau_scheduler.max_step = total_steps
        self.tau_scheduler = LinearScheduler(**asdict(cfg.tau_scheduler))

        if self.cfg.pretrain.compile:
            self.target_backbone = torch.compile(self.target_backbone)
            self.target_neck = torch.compile(self.target_neck)

    def compute_loss(self, batch):
        x = batch['obs']
        act = batch['act']
        game_id = batch['game_id']
        done = batch['done'].type(torch.int64)

        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        target_x = x[:,1:]
        pred_x = x[:,:1]

        # online encoder
        pred_x, _ = self.model.backbone(pred_x)
        pred_x, _ = self.model.neck(pred_x, game_id[:,:1])

        norm_x = torch.linalg.vector_norm(pred_x.detach().squeeze(), ord=2, dim=1)
        var_norm_x, mean_norm_x = torch.var_mean(norm_x)

        # prediction
        pred_x, _ = self.model.head(pred_x, act[:,:-1], game_id[:,:-1])
        pred_x = rearrange(pred_x, 'n t d -> (n t) d')
      
       # momentum encoder
        with torch.no_grad():
            target_x, _ = self.target_backbone(target_x)
            target_x, _ = self.target_neck(target_x, game_id[:,1:])
            
            norm_target_x = torch.linalg.vector_norm(target_x.detach().flatten(start_dim=0, end_dim=1), ord=2, dim=1)
            var_norm_target_x, mean_norm_target_x = torch.var_mean(norm_target_x)
            target_x = rearrange(target_x, 'n t d -> (n t) d')

        norm_log_data = {
            'mean_norm_x': mean_norm_x.item(),
            'var_norm_x': var_norm_x.item(),
            'mean_norm_target_x': mean_norm_target_x.item(),
            'var_norm_target_x': var_norm_target_x.item(),
            'max_norm_x': torch.max(norm_x).item(),
            'min_norm_x': torch.min(norm_x).item()
        }

        # DDP gather
        if self.cfg.pretrain.distributed:
            pred_x_list = [torch.empty_like(pred_x) for _ in range(self.cfg.num_gpus_per_node)]
            target_x_list = [torch.empty_like(target_x) for _ in range(self.cfg.num_gpus_per_node)]
            # Share variables for loss computation across all GPUs
            dist.all_gather(tensor=pred_x, tensor_list=pred_x_list)
            dist.all_gather(tensor=target_x, tensor_list=target_x_list)
            # all_gather() purely copies the value, and the resulting tensor will have no computational graph.
            # To maintain the gradient graph, each GPU must plaster their own tensor (with computational graph) in the corresponding place.
            pred_x_list[self.logger.rank] = pred_x
            pred_x = torch.cat(pred_x_list, dim=0).requires_grad_()
            target_x = torch.cat(target_x_list, dim=0)
        
        similarity = pred_x @ target_x.t()
        gt_label = torch.arange(similarity.shape[0]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(similarity, gt_label)
        self_pred_cnt = (torch.argmax(similarity, dim=1) == gt_label).sum().item()
        self_pred_acc = self_pred_cnt / similarity.shape[0]

        log_data = {
            'loss': loss.item(),
            'self_pred_acc': self_pred_acc,
        }        
        log_data.update(norm_log_data)

        return loss, log_data

    def update(self, batch, step):
        # EMA
        tau = self.tau_scheduler.get_value(step)
        for online, target in zip(self.model.backbone.parameters(), self.target_backbone.parameters()):
            target.data = tau*target.data + (1-tau)*online.data
        for online, target in zip(self.model.neck.parameters(), self.target_neck.parameters()):
            target.data = tau*target.data + (1-tau)*online.data     

    def save_checkpoint(self, epoch, step, name=None):
        if self.logger.rank == 0:
            if name is None:
                name = 'epoch'+str(epoch)
            save_dict = {'backbone': self.model.backbone.state_dict(),
                         'neck': self.model.neck.state_dict(),
                         'head': self.model.head.state_dict(),
                         'target_backbone': self.target_backbone.state_dict(),
                         'target_neck': self.target_neck.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict(),
                         'scaler': self.scaler.state_dict(),
                         'epoch': epoch,
                         'step': step,}
            self.logger.save_dict(save_dict=save_dict,
                                  name=name)