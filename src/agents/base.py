import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import copy
import numpy as np
import wandb
import matplotlib.pyplot as plt

from collections import deque
from einops import rearrange
from abc import *
from typing import Tuple

from src.common.schedulers import LinearScheduler, ExponentialScheduler, CosineScheduler
from src.common.metrics import explained_variance
from src.common.vis_utils import visualize_histogram
from src.envs.atari import *

import time
from collections import defaultdict

class Profiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.current_start = 0.0
        self.current_name = None

    def tick(self, name):
        # Stop previous timer
        if self.current_name:
            elapsed = time.time() - self.current_start
            self.times[self.current_name] += elapsed
            self.counts[self.current_name] += 1
        
        # Start new timer
        self.current_name = name
        self.current_start = time.time()

    def print_stats(self):
        print("\n--- PERFORMANCE PROFILE ---")
        total_time = sum(self.times.values())
        if total_time == 0: return
        
        # Sort by total time spent
        sorted_times = sorted(self.times.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Section':<20} | {'Total (s)':<10} | {'Per Call (ms)':<15} | {'% of Loop':<10}")
        print("-" * 65)
        for name, duration in sorted_times:
            count = self.counts[name]
            per_call = (duration / count) * 1000 if count > 0 else 0
            percent = (duration / total_time) * 100
            print(f"{name:<20} | {duration:<10.2f} | {per_call:<15.2f} | {percent:<10.1f}%")
        print("---------------------------\n")

# Instantiate globally
profiler = Profiler()


class BaseAgent(metaclass=ABCMeta):
    name = 'BaseAgent'
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.game_id = self.train_env.game_id
        
        self.logger = logger
        self.buffer = buffer
        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.optimizer_type = cfg.optimizer['type']
        self.optimizer = self._build_optimizer(self.model.parameters(), cfg.optimizer)
        
        self.prior_weight_scheduler = self._build_scheduler(cfg.prior_weight_scheduler)
        self.eps_scheduler = self._build_scheduler(cfg.eps_scheduler)
        self.gamma_scheduler = self._build_scheduler(cfg.gamma_scheduler)
        self.n_step_scheduler = self._build_scheduler(cfg.n_step_scheduler)

        self.target_model = None # to be defined in  child class

        self.start_step = 1
        self.global_step = 1
        

    @classmethod
    def get_name(cls):
        return cls.name

    def _build_optimizer(self, param_group, optimizer_cfg):
        if 'type' in optimizer_cfg:
            self.optimizer_type = optimizer_cfg.pop('type')
        if self.optimizer_type == 'adam':
            return optim.Adam(param_group, 
                              **optimizer_cfg)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(param_group, 
                              **optimizer_cfg)
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(param_group, 
                                 **optimizer_cfg)
        else:
            raise ValueError
        
    def _build_scheduler(self, scheduler_cfg):
        scheduler_type = scheduler_cfg.pop('type')
        if scheduler_type == 'linear':
            return LinearScheduler(**scheduler_cfg)
        if scheduler_type == 'exponential':
            return ExponentialScheduler(**scheduler_cfg)
        if scheduler_type == 'cosine':
            return CosineScheduler(**scheduler_cfg)
        else:
            raise ValueError
    
    # @abstractmethod
    def predict(self, model, obs, eps, n, t) -> torch.Tensor:
        pass

    # @abstractmethod
    def forward(self, online_model, target_model, batch, mode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [output] loss
        [output] pred: prediction Q-value
        [output] target: target Q-value
        """
        pass
        # self.prior_weight_scheduler = self._build_scheduler(cfg.prior_weight_scheduler)
        # self.eps_scheduler = self._build_scheduler(cfg.eps_scheduler)
        # self.gamma_scheduler = self._build_scheduler(cfg.gamma_scheduler)
        # self.n_step_scheduler = self._build_scheduler(cfg.n_step_scheduler)

    def save_checkpoint(self):
        assert self.target_model is not None
        save_dict = {
            'model.backbone': self.model.backbone.state_dict(),
            'model.neck': self.model.neck.state_dict(),
            'model.head': self.model.head.state_dict(),
            'target.backbone': self.target_model.backbone.state_dict(),
            'target.neck': self.target_model.neck.state_dict(),
            'target.head': self.target_model.head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'buffer_ckpt': 'latest', 
        }
        save_name = f'ckpt_step_{self.global_step}_game_{self.train_env._game}_seed_{self.cfg.seed}.pth'
        self.logger.save_dict(save_dict, save_name)
    
    def load_checkpoint(self):
        assert self.target_model is not None
        load_dict = {
            "model.backbone": self.model.backbone,
            "model.neck": self.model.neck,
            "model.head": self.model.head,
            "target.backbone": self.target_model.backbone,
            "target.neck": self.target_model.neck,
            "target.head": self.target_model.head,
            "optimizer": self.optimizer,
            'global_step': -1,
            'buffer_ckpt': -1  # Add this
        }
        ret = self.logger.load_dict(load_dict=load_dict,
                                    path=self.cfg.ckpt_path,
                                    device=self.device)
        self.global_step = ret['global_step']
        self.start_step = self.global_step + 1
        self.logger.timestep = self.global_step
        
        # Load buffer with correct ckpt number
        game_name = ''.join(word.capitalize() for word in self.train_env._game.split('_'))
        buffer_ckpt = ret.get('buffer_ckpt', self.buffer.ckpt)  # Use saved ckpt
        
        try:
            self.buffer.load_buffer(
                buffer_dir=self.cfg.buffer_dir,
                game=game_name,
                run_id=self.logger.run_id,
                ckpt=buffer_ckpt
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Buffer checkpoint not found, starting with empty buffer")
        
        print(f'Checkpoint loaded: start step {self.start_step}')

    def train(self):
        obs = self.train_env.reset()
        self.initial_model = copy.deepcopy(self.model)
        online_model = self.model
        target_model = self.target_model
        
        if self.cfg.exploration_model == 'online':
            exploration_model = self.model
        else:
            exploration_model = self.target_model

        assert exploration_model is not None, "Exploration model is not defined."
        
        optimize_step = 1
        self.eps = 1.0

        if self.cfg.resume:
            self.load_checkpoint()

        for env_step in tqdm.tqdm(range(self.start_step, self.cfg.num_timesteps+1), 
                                desc=f"Device: {self.device}, Game_ID: {self.game_id}",
                                initial=self.start_step,
                                total=self.cfg.num_timesteps):
            profiler.tick('Setup/Mode Switch')
            
            self.global_step = env_step
            
            # Collect trajectory
            for module, mode in self.cfg.exploration_mode.items():
                getattr(exploration_model, module).train() if mode == 'train' else \
                    getattr(exploration_model, module).eval()
            
            profiler.tick('Encode/Act')

            obs_tensor = self.buffer.encode_obs(obs, prediction=True) # (n, t, f, c, h, w)
            n, t, _, _, _, _ = obs_tensor.shape
            with torch.no_grad():
                backbone_feat, _ = exploration_model.backbone(obs_tensor)
            
            action = self.predict(exploration_model, backbone_feat, self.eps, n, t)

            profiler.tick('Env Step')

            next_obs, reward, done, info = self.train_env.step(action.item())

            profiler.tick('Buffer Store')

            if self.cfg.buffer.save_backbone_feat:
                self.buffer.store(backbone_feat.cpu().numpy(), action, reward, done)
            else:
                self.buffer.store(obs, action, reward, done)
            self.logger.step(obs, reward, done, info, mode='train')
            
            obs = self.train_env.reset() if info.traj_done else next_obs
            
            if env_step >= self.cfg.min_buffer_size:
                self.eps = self.eps_scheduler.get_value(env_step - self.cfg.min_buffer_size)
                
                # Optimize
                for _ in range(self.cfg.optimize_per_env_step):
                    profiler.tick('Opt: ModeSwitch')
                    # Set modes
                    for module, mode in self.cfg.train_online_mode.items():
                        getattr(online_model, module).train() if mode == 'train' else \
                            getattr(online_model, module).eval()
                    
                    for module, mode in self.cfg.train_target_mode.items():
                        getattr(target_model, module).train() if mode == 'train' else \
                            getattr(target_model, module).eval()
                    
                    profiler.tick('Opt: Schedulers')
                    # Schedulers
                    self.prior_weight = self.prior_weight_scheduler.get_value(optimize_step)
                    optimize_step_after_reset = optimize_step % self.cfg.reset_per_optimize_step
                    self.gamma = self.gamma_scheduler.get_value(optimize_step_after_reset)
                    self.n_step = int(np.round(
                        self.n_step_scheduler.get_value(optimize_step_after_reset)
                    ))
                    
                    profiler.tick('Opt: Sample')
                    # Sample and update
                    batch = self.buffer.sample(
                        self.cfg.batch_size, self.n_step, self.gamma, self.prior_weight
                    )

                    profiler.tick('Opt: Forward/Loss')
                    loss, preds, targets = self.forward(
                        online_model, target_model, batch, mode='train'
                    )
                    
                    profiler.tick('Opt: Backprop')
                    self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad_norm
                    )
                    self.optimizer.step()

                    profiler.tick('Opt: Target Update')
                    # Target update (vectorized)
                    update_layers = list(
                        set(['backbone', 'neck', 'head']) - set(self.cfg.freeze_layers)
                    )
                    tau = self.cfg.target_tau
                    
                    with torch.no_grad():  # Explicit no_grad for target updates
                        for layer in update_layers:
                            online_layer = getattr(online_model, layer)
                            target_layer = getattr(target_model, layer)
                            
                            for online, target in zip(
                                online_layer.parameters(), target_layer.parameters()
                            ):
                                target.data.mul_(tau).add_(online.data, alpha=1-tau)
                            
                            if self.cfg.update_buffer:
                                for online, target in zip(
                                    online_layer.buffers(), target_layer.buffers()
                                ):
                                    target.data.copy_(online.data)
                    
                    train_logs = {
                        'eps': self.eps,
                        'prior_weight': self.prior_weight,
                        'gamma': self.gamma,
                        'n_step': self.n_step,
                        'loss': loss.item(),
                    }
                    self.logger.update_log(mode='train', **train_logs)
                    optimize_step += 1
                
                online_model.eval()
                
                # Evaluate less frequently
                if (env_step % self.cfg.evaluate_freq == 0) and (self.cfg.evaluate_freq != -1):
                    profiler.tick('Eval')
                    eval_logs = self.evaluate()
                    self.logger.update_log(mode='eval', **eval_logs)
                
                if (env_step % self.cfg.rollout_freq == 0) and (self.cfg.rollout_freq != -1):
                    profiler.tick('Rollout')
                    rollout_logs = self.rollout()
                    self.logger.update_log(mode='eval', **rollout_logs)
                
                # Logging
                if env_step % self.cfg.log_freq == 0:
                    profiler.tick('Writing Logs')
                    self.logger.write_log(mode='train')
                    self.logger.write_log(mode='eval')
                
                if env_step % self.cfg.save_buffer_every == 0:
                    print("[Buffer Save] Saving replay buffer...")
                    profiler.tick('Buffer Save')
                    game_name = ''.join(
                        word.capitalize() for word in self.train_env._game.split('_')
                    )
                    self.buffer.save_buffer(
                        buffer_dir=self.cfg.buffer_dir,
                        game=game_name,
                        run_id=self.logger.run_id,
                        ckpt='latest'
                    )

                if (env_step % self.cfg.ckpt_freq == 0) and (self.cfg.ckpt_freq != -1):
                    self.save_checkpoint()
                
                if env_step % 1000 == 0:
                    profiler.print_stats()
        
        # Final evaluation and rollout
        rollout_logs = self.rollout()
        self.logger.update_log(mode='eval', **rollout_logs)
        self.logger.write_log(mode='eval')


    def evaluate(self):
        online_model = self.model
        target_model = self.target_model

        # Sample once
        batch = self.buffer.sample(self.cfg.batch_size, self.n_step, self.gamma, self.prior_weight)

        with torch.no_grad():
            rl_loss, preds, targets = self.forward(online_model, target_model, batch, mode='eval')
        
        pred_var = torch.var(preds)
        target_var = torch.var(targets)
        exp_var = explained_variance(preds, targets)

        # parameter distance
        param_dist = sum(
            ((initial - online)**2).sum()
            for (_, initial), (_, online) in zip(
                self.initial_model.named_parameters(), 
                online_model.named_parameters()
            )
        )
        param_dist = torch.sqrt(param_dist) # type: ignore

        weight_norm = torch.sqrt(
            sum((param**2).sum() for param in online_model.parameters()) # type: ignore
        )

        eval_logs = {
            'pred_var': pred_var.item(),
            'target_var': target_var.item(),
            'exp_var': exp_var.item(),
            'rl_loss': rl_loss.item(),
            'param_dist': param_dist.item(),
            'weight_norm': weight_norm.item(),
        }

        return eval_logs
    

    # def evaluate_old(self):
    #     ######################
    #     # Forward
    #     online_model = self.model
    #     target_model = self.target_model
        
    #     # get output of each sublayer with hook
    #     layer_wise_outputs = {}
    #     def save_outputs_hook(layer_id):
    #         def hook(_, __, output):
    #             layer_wise_outputs[layer_id] = output
    #         return hook
        
    #     def register_hooks(net, prefix=''):
    #         for name, layer in net._modules.items():
    #             if len(list(layer.children())) > 0:  # Check if it has children
    #                 register_hooks(layer, prefix=f"{prefix}.{name}")
    #             else:
    #                 layer_id = f"{prefix}.{name}.{layer.__class__.__name__}"
    #                 layer.register_forward_hook(save_outputs_hook(layer_id))
        
    #     register_hooks(online_model.backbone, 'backbone')
    #     register_hooks(online_model.neck, 'neck')
    #     register_hooks(online_model.head, 'head')
        
    #     # forward
    #     batch = self.buffer.sample(self.cfg.batch_size, self.n_step, self.gamma, self.prior_weight)
    #     batch['obs'].requires_grad=True
    #     rl_loss, preds, targets = self.forward(online_model, target_model, batch, mode='eval')
        
    #     # explained variance    
    #     pred_var = torch.var(preds)
    #     target_var = torch.var(targets)
    #     exp_var = explained_variance(preds, targets)

    #     ##########################
    #     # Smoothness
    #     # smoothness of prediction
    #     # grad_norm: w.r.t the input
    #     grads = torch.autograd.grad(
    #         outputs=preds, inputs=batch['obs'], grad_outputs=torch.ones_like(preds), 
    #         create_graph=False, retain_graph=True, allow_unused=True
    #     )[0]
    #     pred_grad_norm = torch.mean(grads.flatten(1).norm(2, -1))

    #     # fisher trace: w.r.t the parameter
    #     pred_fisher_trace = 0
    #     params = [param for param in self.model.parameters() if param.requires_grad]
    #     for param in params:
    #         grads = torch.autograd.grad(
    #             outputs=preds, inputs=param, grad_outputs=torch.ones_like(preds),  
    #             create_graph=False, retain_graph=True, allow_unused=True
    #         )[0]            
    #         if grads is not None:
    #             pred_fisher_trace += (grads ** 2).mean()

    #     # smoothness of loss
    #     # grad_norm: w.r.t the input
    #     grads = torch.autograd.grad(
    #         outputs=rl_loss, inputs=batch['obs'], 
    #         create_graph=False, retain_graph=True, allow_unused=True
    #     )[0]
    #     loss_grad_norm = torch.mean(grads.flatten(1).norm(2, -1))

    #     # fisher trace: w.r.t the parameter
    #     loss_fisher_trace = 0
    #     params = [param for param in self.model.parameters() if param.requires_grad]
    #     for param in params:
    #         grads = torch.autograd.grad(
    #             outputs=rl_loss, inputs=param,
    #             create_graph=False, retain_graph=True, allow_unused=True
    #         )[0]            
    #         if grads is not None:
    #             loss_fisher_trace += (grads ** 2).mean()

    #     # parameter distance
    #     param_dist, backbone_param_dist, neck_param_dist, head_param_dist = 0.0, 0.0, 0.0, 0.0
    #     for (key, initial), (_, online) in zip(
    #         self.initial_model.named_parameters(), 
    #         online_model.named_parameters()
    #         ):
    #         dist = ((initial - online)**2).sum()
    #         if 'backbone' in key:
    #             backbone_param_dist += dist
    #         elif 'neck' in key:
    #             neck_param_dist += dist
    #         elif 'head' in key:
    #             head_param_dist += dist
                
    #         param_dist += dist

    #     param_dist = torch.sqrt(param_dist)
    #     backbone_param_dist = torch.sqrt(backbone_param_dist)
    #     neck_param_dist = torch.sqrt(neck_param_dist)
    #     head_param_dist = torch.sqrt(head_param_dist)
            
    #     # weight_norm
    #     weight_norm, backbone_weight_norm, neck_weight_norm, head_weight_norm= 0.0, 0.0, 0.0, 0.0
    #     for key, param in online_model.named_parameters():
    #         norm = ((param)**2).sum()
    #         if 'backbone' in key:
    #             backbone_weight_norm += norm
    #         elif 'neck' in key:
    #             neck_weight_norm += norm
    #         elif 'head' in key:
    #             head_weight_norm += norm
    #         weight_norm += norm
            
    #     weight_norm = torch.sqrt(weight_norm)
    #     backbone_weight_norm = torch.sqrt(backbone_weight_norm)
    #     neck_weight_norm = torch.sqrt(neck_weight_norm)
    #     head_weight_norm = torch.sqrt(head_weight_norm)
        
    #     #############################
    #     # log evaluated metrics
    #     eval_logs = {
    #         'pred_var': pred_var.item(),
    #         'target_var': target_var.item(),
    #         'exp_var': exp_var.item(),
    #         'rl_loss': rl_loss.item(),
    #         'pred_grad_norm': pred_grad_norm.item(),
    #         'pred_fisher_trace': pred_fisher_trace.item(),      
    #         'loss_grad_norm': loss_grad_norm.item(),
    #         'loss_fisher_trace': loss_fisher_trace.item(),
    #         'param_dist': param_dist.item(),
    #         'backbone_param_dist': backbone_param_dist.item(),
    #         'neck_param_dist': neck_param_dist.item(),
    #         'head_param_dist': head_param_dist.item(),
    #         'weight_norm': weight_norm.item(),
    #         'backbone_weight_norm': backbone_weight_norm.item(),
    #         'neck_weight_norm': neck_weight_norm.item(),
    #         'head_weight_norm': head_weight_norm.item(),
    #     }
        
    #     if self.cfg.plot_weight_histogram:
    #         weight_histogram= {}
    #         for layer_name, param in online_model.named_parameters():
    #             hist, edges = np.histogram(param.flatten().abs().cpu().detach().numpy(), bins = 50)
    #             histogram = visualize_histogram(hist, edges)
    #             weight_histogram['weight_' + layer_name] = wandb.Image(histogram)
    #         for layer_name, activation in layer_wise_outputs.items(): 
    #             hist, edges = np.histogram(activation[0].flatten().cpu().detach().numpy(), bins = 50)
    #             histogram = visualize_histogram(hist, edges)
    #             weight_histogram['activation_' + layer_name] = wandb.Image(histogram)
            
    #         eval_logs.update(weight_histogram)

    #     return eval_logs 
        
    def rollout(self, max_steps: int = 10000):
        """Run evaluation rollouts with optional max step limit and progress tracking."""
        if self.cfg.rollout_model == 'online':
            rollout_model = self.model
        else:
            rollout_model = self.target_model
        
        assert rollout_model is not None, "Rollout model is not defined."

        game_id = torch.LongTensor(self.eval_env.game_id).to(self.device)
        game_id = rearrange(game_id, 'n -> n 1')

        obs = self.eval_env.reset()  # (n, f, c, h, w)
        frames = deque([], maxlen=1000)

        # print(f"[Rollout] Starting evaluation on {n_envs} environments, "
        #     f"max_steps={max_steps}")

        for step in tqdm.tqdm(range(max_steps), desc="Rollout progress", ncols=90):
            # Save one frame from env 0
            frames.append(obs[0][-1])

            obs_tensor = self.buffer.encode_obs(obs)
            n, t = obs_tensor.shape[0], obs_tensor.shape[1]

            with torch.no_grad():
                backbone_feat, _ = rollout_model.backbone(obs_tensor)
                action = self.predict(rollout_model, backbone_feat, self.cfg.eval_eps, n, t)

            next_obs, reward, done, info = self.eval_env.step(action.reshape(-1))

            self.logger.step(obs, reward, done, info, mode='eval')

            # End condition: all eval envs finished
            if self.logger.is_traj_done(mode='eval'):
                print(f"[Rollout] All trajectories finished at step {step + 1}")
                break

            obs = next_obs

        else:
            print(f"[Rollout] WARNING: Reached max_steps={max_steps} "
                f"without completing all episodes.")

        # video = np.array(frames)
        # if video.shape[1] == 1:
        #     video = video.repeat(3, 1)

        rollout_logs = self.logger.fetch_log(mode='eval')
        rollout_logs["rollout_steps"] = step + 1
        rollout_logs["max_steps_reached"] = (step + 1 == max_steps)

        return rollout_logs

        
        