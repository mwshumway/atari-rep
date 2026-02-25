from .base import BaseAgent

import copy
import torch
from einops import rearrange
import numpy as np
import tqdm

class RainbowAgent(BaseAgent):
    name = "rainbow"

    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 model,
                 buffer,
                 logger,
                 aug_func):
        super().__init__(cfg, device, train_env, eval_env, model, buffer, logger, aug_func)
        
        # Load and freeze the target model
        self.target_model = copy.deepcopy(self.model).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # Distributional RL parameters
        self.num_atoms = self.model.head.num_atoms
        self.v_min, self.v_max = self.cfg.agent.v_min, self.cfg.agent.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        if self.cfg.agent.compile:
            self._compile()
    
    def _compile(self):
        self.model.backbone = torch.compile(self.model.backbone)
        self.model.neck = torch.compile(self.model.neck)
        self.model.head = torch.compile(self.model.head)

        self.target_model.backbone = torch.compile(self.target_model.backbone)
        self.target_model.neck = torch.compile(self.target_model.neck)
        self.target_model.head = torch.compile(self.target_model.head)
    
    def predict(self, model, backbone_feat, eps, n, t) -> torch.Tensor:
        """
        :param model: the online model used for action selection
        :param backbone_feat: pre-extracted backbone features (n, t, f, c, h, w)
        :param eps: epsilon for epsilon-greedy action selection
        :param n: batch size
        :param t: number of time steps
        :return: action (n, t)
        """
        game_id = torch.LongTensor([[self.game_id]]).to(self.device)
        game_id = game_id.repeat(n, 1)
        if self.cfg.agent.rep:
            _, neck_info = model.neck(backbone_feat, game_id) # game-wise spatial embedding
            x = neck_info[self.cfg.agent.rep_candidate]
        else:
            x, _ = model.neck(backbone_feat, game_id) # game-wise spatial embedding
        q_dist, _ = model.head(x, game_id) # game-wise prediction head
        
        # distribution -> value
        support = rearrange(self.support, 'n_a -> 1 1 1 n_a')
        q_value = (q_dist * support).sum(-1)
        argmax_action = torch.argmax(q_value, -1).cpu().numpy()
        
        # eps-greedy
        prob = np.random.rand(n, t)
        is_rand = (prob <= eps)
        rand_action = np.random.randint(0, self.cfg.action_size, (n, t))
        action = is_rand * rand_action + (1-is_rand) * argmax_action
        
        return action
    
    def forward(self, online_model, target_model, batch, mode, reduction='mean'):
        # get samples from buffer
        obs = batch['obs']
        act = batch['act']
        done = batch['done']
        next_obs = batch['next_obs']
        G = batch['G']
        
        n_step = batch['n_step']
        gamma = batch['gamma']
        
        tree_idxs = batch['tree_idxs']
        weights = batch['weights']
        
        n, t, f, c, _, _ = obs.shape
        obs = rearrange(obs, 'n t f c h w -> n (t f c) h w')
        next_obs = rearrange(next_obs, 'n t f c h w -> n (t f c) h w')
        obs = self.aug_func(obs)
        if self.cfg.agent.aug_target:
            next_obs = self.aug_func(next_obs)
        obs = rearrange(obs, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        next_obs = rearrange(next_obs, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # Calculate current state's q-value distribution
        game_id = torch.LongTensor([[self.game_id]]).to(self.device) 
        game_id = game_id.repeat(n, 1)
        x, _ = online_model.backbone(obs)
        if self.cfg.agent.rep:
            _, neck_info = online_model.neck(x, game_id) # game-wise spatial embedding
            x = neck_info[self.cfg.agent.rep_candidate]
        else:
            x, _ = online_model.neck(x, game_id) # game-wise spatial embedding
        _, head_info = online_model.head(x, game_id) # game-wise prediction head
        log_pred_q_dist = rearrange(head_info['log'], 'n t d n_a -> (n t) d n_a')
        
        # gather action
        act_idx = act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
        log_pred_q_dist = (log_pred_q_dist).gather(1, act_idx).squeeze(1) # (n, n_a)
        
        with torch.no_grad():
            # Calculate n-th next state's q-value distribution
            # next_target_q_dist: (n, a, num_atoms)
            # target_q_dist: (n, num_atoms)
            ntx, _ = target_model.backbone(next_obs)
            if self.cfg.agent.rep:
                _, neck_info = target_model.neck(ntx, game_id) # game-wise spatial embedding
                ntx = neck_info[self.cfg.agent.rep_candidate]
            else:
                ntx, _ = target_model.neck(ntx, game_id)
            next_target_q_dist, _ = target_model.head(ntx, game_id)
            next_target_q_dist = rearrange(next_target_q_dist, 'n t d n_a -> (n t) d n_a')
            
            if self.cfg.agent.double:
                nox, _ = online_model.backbone(next_obs)
                if self.cfg.agent.rep:
                    _, neck_info = online_model.neck(nox, game_id) # game-wise spatial embedding
                    nox = neck_info[self.cfg.agent.rep_candidate]
                else:
                    nox, _ = online_model.neck(nox, game_id)
                next_online_q_dist, _ = online_model.head(nox, game_id)
                next_online_q_dist = rearrange(next_online_q_dist, 'n t d n_a -> (n t) d n_a')                
                next_online_q =  (next_online_q_dist * self.support.reshape(1,1,-1)).sum(-1)
                next_act = torch.argmax(next_online_q, 1)
            else:       
                next_target_q =  (next_target_q_dist * self.support.reshape(1,1,-1)).sum(-1)     
                next_act = torch.argmax(next_target_q, 1)  
                
            next_act_idx = next_act.reshape(-1, 1, 1).expand(-1, 1, self.num_atoms)
            target_q_dist = next_target_q_dist.gather(1, next_act_idx).squeeze(1)
        
            # C51 (https://arxiv.org/abs/1707.06887, Algorithm 1)
            # Compute the projection 
            # Tz = R_n + (γ^n)Z (w/ n-step return) (N, N_A)
            gamma_n = (gamma ** n_step)
            Tz = G.unsqueeze(-1) + gamma_n * self.support.unsqueeze(0) * (1-done).unsqueeze(-1)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            # L2-projection
            b = (Tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            # m = torch.zeros((n, self.num_atoms), device=self.device)
            # for idx in range(n):
            #     # += operation do not allow to add value to same index multiple times
            #     m[idx].index_add_(0, l[idx], target_q_dist[idx] * (u[idx] - b[idx]))
            #     m[idx].index_add_(0, u[idx], target_q_dist[idx] * (b[idx] - l[idx]))
            offset = torch.linspace(0, (n - 1) * self.num_atoms, n, device=self.device).long().unsqueeze(1)
            l_idx = (l + offset).view(-1)
            u_idx = (u + offset).view(-1)
            lower_vals = (target_q_dist * (u.float() - b)).view(-1)
            upper_vals = (target_q_dist * (b - l.float())).view(-1)
            m_flat = torch.zeros(n * self.num_atoms, device=self.device)

            m_flat.index_add_(0, l_idx, lower_vals)
            m_flat.index_add_(0, u_idx, upper_vals)
            m = m_flat.view(n, self.num_atoms)
                            
        # kl-divergence KL(p||q)=plogp-plogq
        # Here, plogp is just a constant
        EPS = 1e-5
        kl_div = -torch.sum(m * log_pred_q_dist, -1)
        kl_div = torch.clamp(kl_div, EPS, 1 / EPS)
        
        if reduction == 'mean':
            loss = (kl_div * weights).mean()
        else:
            loss = (kl_div * weights)

        # update priority
        if (self.buffer.name == 'per_buffer') and (mode == 'train'):
            self.buffer.update_priorities(idxs=tree_idxs, priorities=kl_div.detach().cpu().numpy())

        # prediction and target
        pred_q_dist = torch.exp(log_pred_q_dist)
        preds = (pred_q_dist * self.support.reshape(1,-1)).sum(-1)
        target_q = (target_q_dist * self.support.reshape(1,-1)).sum(-1)
        targets = G + gamma_n * target_q * (1-done)

        return loss, preds, targets

    def train(self):
        obs = self.train_env.reset()
        self.initial_model = copy.deepcopy(self.model).to('cpu')
        online_model = self.model
        target_model = self.target_model

        if self.cfg.agent.exploration_model == "online":
            exploration_model = online_model
        elif self.cfg.agent.exploration_model == "target":
            exploration_model = target_model

        optimize_step = 1
        self.eps = self.eps_scheduler.get_value(0)

        # Initial rollout before training
        rollout_logs = self.rollout(online_model)
        self.logger.update_log(mode="eval", **rollout_logs)
        self.logger.write_log(mode="eval")
        self.probe_on_policy(target_model, outer_step=0)
        self.logger.probe_logger.reset()
        
        for step in tqdm.tqdm(range(1, self.cfg.agent.num_timesteps+1), desc="Training"):
            online_model.train()
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)
            n, t, _, _, _, _ = obs_tensor.shape
            with torch.no_grad():
                backbone_feat, _ = exploration_model.backbone(obs_tensor.to(self.device))

            action = self.predict(exploration_model, backbone_feat, self.eps_scheduler.get_value(step), n, t)
            next_obs, reward, done, info = self.train_env.step(action.item())  # TODO: support > 1 training envs
            self.buffer.store(obs, action, reward, done)
            self.logger.step(obs, reward, done, info, mode="train")

            obs = self.train_env.reset() if info.traj_done else next_obs

            if step >= self.cfg.agent.min_buffer_size:
                self.eps = self.eps_scheduler.get_value(step - self.cfg.agent.min_buffer_size)

                for _ in range(self.cfg.agent.optimize_per_env_step):
                    self.prior_weight = self.prior_weight_scheduler.get_value(optimize_step)
                    optimize_step_after_reset = optimize_step % self.cfg.agent.reset_per_optimize_step
                    self.gamma = self.gamma_scheduler.get_value(optimize_step_after_reset)
                    self.n_step = int(np.round(self.n_step_scheduler.get_value(optimize_step_after_reset)))

                    batch = self.buffer.sample(self.cfg.agent.batch_size, self.n_step, self.gamma, self.prior_weight)
                    loss, _, _ = self.forward(online_model, target_model, batch, mode='train')
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.agent.clip_grad_norm)
                    self.optimizer.step()

                    update_layers = list(set(['backbone', 'neck', 'head']) - set(self.cfg.load_model.freeze_layers))
                    tau= self.cfg.agent.target_tau

                    with torch.no_grad():
                        for layer in update_layers:
                            online_layer = getattr(online_model, layer)
                            target_layer = getattr(target_model, layer)
                            for online_param, target_param in zip(online_layer.parameters(), target_layer.parameters()):
                                target_param.data.mul_(tau).add_(online_param.data, alpha=1-tau)
                            
                            if self.cfg.agent.update_buffer:
                                for online_param, target_param in zip(online_layer.parameters(), target_layer.parameters()):
                                    target_param.data.copy_(online_param.data)
                    
                    self.logger.update_log(mode="train", loss=loss.item(), eps=self.eps, gamma=self.gamma, n_step=self.n_step, prior_weight=self.prior_weight)
                    optimize_step += 1
                
                if (step % self.cfg.agent.eval_freq == 0) and (self.cfg.agent.eval_freq > 0):
                    eval_logs = self.evaluate(online_model)
                    self.logger.update_log(mode="eval", **eval_logs)
                
                if (step % self.cfg.agent.rollout_freq == 0) and (self.cfg.agent.rollout_freq > 0):
                    rollout_logs = self.rollout(target_model)
                    self.logger.update_log(mode="eval", **rollout_logs)
                    self.logger.write_log(mode="eval")
                
                if (step % self.cfg.agent.probe_on_policy_freq == 0) and (self.cfg.agent.probe_on_policy_freq > 0):
                    self.probe_on_policy(target_model, step)
                    self.logger.probe_logger.reset()
                
                if (step % self.cfg.agent.save_freq == 0) and (self.cfg.agent.save_freq > 0):
                    self.save_progress()
                
                if step % self.cfg.agent.log_freq == 0:
                    self.logger.write_log(mode="train")

        # Final evaluation after training
        rollout_logs = self.rollout(online_model)
        self.logger.update_log(mode="eval", **rollout_logs)
        self.logger.write_log(mode="eval")
    
    def evaluate(self, model):
        """Evaluate the model on all of the evaluation environments."""
        return {}
    
    def rollout(self, model):
        """Rollout the model on all the evaluation environments."""
        obs = self.eval_env.reset() # (n, t, num_envs, f, c, h, w)
        all_envs_done = False
        
        for step in tqdm.tqdm(range(self.cfg.agent.max_rollout_steps), desc="Rollout"):
            obs_tensor = self.buffer.encode_obs(obs, prediction=True).to(self.device)
            n, t, num_envs, f, c, h, w = obs_tensor.shape
            obs_tensor = rearrange(obs_tensor, 'n t num_envs f c h w -> (n num_envs) t f c h w')
            with torch.no_grad():
                backbone_feat, _ = model.backbone(obs_tensor)
                action = self.predict(model, backbone_feat, eps=self.cfg.agent.eval_eps, n=n * num_envs, t=t)
            
            next_obs, reward, done, info = self.eval_env.step(action.reshape(-1))

            self.logger.step(obs, reward, done, info, mode="eval")

            if self.logger.is_traj_done(mode="eval"):
                all_envs_done = True
                break

            obs = next_obs

        return {
            "rollout_steps": step + 1,
            "max_steps_reached": all_envs_done
        }
    
    def save_progress(self):
        raise NotImplementedError("Progress saving not implemented yet.")
    
    def probe_on_policy(self, model, outer_step):
        """
        Create an on-policy dataset of the current model and probe it with a linear layer.
        """
        assert self.cfg.wandb.enabled, "On-policy probing requires logging with wandb to track probe learning curve."

        # Keep track of trajectories per environment to compute returns properly
        env_trajectories = [[] for _ in range(self.cfg.num_eval_envs)]
        
        obs = self.eval_env.reset() # (n, t, num_envs, f, c, h, w)
        game_id = torch.full((self.cfg.num_eval_envs, 1), self.game_id, dtype=torch.long, device=self.device)

        for step in tqdm.tqdm(range(self.cfg.agent.max_rollout_steps), desc="On-policy probing rollout"):
            obs_tensor = self.buffer.encode_obs(obs, prediction=True).to(self.device)
            n, t, num_envs, f, c, h, w = obs_tensor.shape
            obs_tensor = rearrange(obs_tensor, 'n t num_envs f c h w -> (n num_envs) t f c h w')

            with torch.no_grad():
                backbone_feat, _ = model.backbone(obs_tensor)
                if self.cfg.agent.rep:
                    _, neck_info = model.neck(backbone_feat, game_id=game_id)
                    neck_feat = neck_info[self.cfg.agent.rep_candidate]
                else:
                    neck_feat, _ = model.neck(backbone_feat, game_id=game_id)
                
                action = self.predict(model, backbone_feat, eps=self.cfg.agent.eval_eps, n=n * num_envs, t=t)

                next_obs, reward, done, info = self.eval_env.step(action.reshape(-1))
                self.logger.step(obs, reward, done, info, mode="probe")
                
                neck_cpu = neck_feat.cpu().view(num_envs, -1)
                actions_list = action.reshape(-1).tolist()
                
                for i in range(num_envs):
                    # Store trajectories including the `done` flag to stop accumulation at episode ends
                    env_trajectories[i].append((neck_cpu[i], actions_list[i], float(reward[i]), bool(done[i])))

                if self.logger.is_traj_done(mode="probe"):
                    break 
                
                obs = next_obs
        
        from src.probe.probe_utils import create_probe_dataset
        dataset = create_probe_dataset(env_trajectories, self.cfg)

        # Train Action Probe
        from src.probe.action import train_action_probe
        train_action_probe(cfg=self.cfg, dataset_list=dataset, outer_step=outer_step, device=self.device, action_meanings=self.train_env.get_action_meanings()) 

        # Train Value Probe
        from src.probe.value import train_value_probe
        train_value_probe(cfg=self.cfg, dataset_list=dataset, outer_step=outer_step, device=self.device)
            
