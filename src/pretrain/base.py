import torch
import torch.optim as optim
from torch import distributed as dist
from dataclasses import asdict
from abc import abstractmethod
from typing import Tuple
from einops import rearrange
import tqdm

from src.utils.schedulers import CosineAnnealingWarmupRestarts
from .pretrain_utils import get_grad_norm_stats, is_main_process

def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

class BaseTrainer():
    name = "base_trainer"

    def __init__(
            self,
            cfg, 
            device,
            train_loader,
            train_sampler,
            eval_loader,
            eval_sampler,
            logger,
            aug_func,
            model
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.eval_loader = eval_loader
        self.eval_sampler = eval_sampler
        self.logger = logger
        self.aug_func = aug_func
        self.model = model

        if self.cfg.pretrain.num_epochs > 0 and self.train_loader is not None:
            self.aug_func = aug_func.to(self.device)
            self.optimizer = self._build_optimizer(asdict(self.cfg.optimizer))
            self.lr_scheduler = self._build_scheduler(self.optimizer, asdict(self.cfg.lr_scheduler))
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.pretrain.use_amp)
        
        self.distributed = self.cfg.pretrain.distributed
        if self.distributed:
            self.model.backbone = torch.nn.parallel.DistributedDataParallel(self.model.backbone)
            self.model.neck = torch.nn.parallel.DistributedDataParallel(self.model.neck)
            self.model.head = torch.nn.parallel.DistributedDataParallel(self.model.head)
        
        if self.cfg.pretrain.compile:
            self.model.backbone = torch.compile(self.model.backbone)
            self.model.neck = torch.compile(self.model.neck)
            self.model.head = torch.compile(self.model.head)
        
        self.start_step = 0
        self.start_epoch = 1
    
    @classmethod
    def get_name(cls):
        return cls.name

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError
    
    def _build_scheduler(self, optimizer, scheduler_cfg):
        first_cycle_steps = len(self.train_loader) * self.cfg.pretrain.num_epochs
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=first_cycle_steps,
                                             **scheduler_cfg)


    @abstractmethod
    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    # custom model update other than backpropagation (e.g., ema)
    def update(self, batch, step):
        pass

    def _collate(self, batch, f=4):
        """
        [params] 
            observation: (n, t+f-1, c, h, w) 
            next_observation: (n, t+f-1, c, h, w)
            action:   (n, t+f-1)
            reward:   (n, t+f-1)
            terminal: (n, n+t+f-1) * different n's (batch vs n_step)
            rtg:      (n, t+f-1)
            game_id:  (n, t+f-1)            
        [returns] 
            (c = 1 in ATARI)
            obs:      (n, t, f, c, h, w) 
            next_obs: (n, t, f, c, h, w)
            action:   (n, t)
            reward:   (n, t)
            done:     (n, n+t)
            rtg:      (n, t)
            game_id:  (n, t)    
        """
        obs = batch['observation']
        action = batch['action']
        reward = batch['reward']
        done = batch['terminal']
        rtg = batch['rtg']
        game_id = batch['game_id']
        next_obs = batch['next_observation']

        # process data-format
        obs = rearrange(obs, 'n tf c h w -> n tf 1 c h w')
        obs = obs.repeat(1, 1, f, 1, 1, 1)
        next_obs = rearrange(next_obs, 'n tf c h w -> n tf 1 c h w')
        next_obs = next_obs.repeat(1, 1, f, 1, 1, 1)
        action = action.long()
        reward = torch.nan_to_num(reward).sign()
        done = done.bool()
        rtg = rtg.float()
        game_id = game_id.long()

        # frame-stack
        if f != 1:
            for i in range(1, f):
                obs[:, :, i] = obs[:, :, i].roll(-i, 1)
                next_obs[:, :, i] = next_obs[:, :, i].roll(-i, 1)
            obs = obs[:, :-(f-1)]
            next_obs = next_obs[:, :-(f-1)]
            action = action[:, f-1:]
            reward = reward[:, f-1:]
            done = done[:, f-1:]
            rtg = rtg[:, f-1:]
            game_id = game_id[:, f-1:]
            
        # lazy frame to float
        obs = obs / 255.0
        next_obs = next_obs / 255.0
            
        batch = {
            'obs': obs,
            'next_obs': next_obs,
            'act': action,
            'rew': reward,
            'done': done,
            'rtg': rtg,
            'game_id': game_id,                            
        }            
            
        return batch


    def train(self):
        step = self.start_step
                
        # train
        use_amp = self.cfg.pretrain.use_amp
        for epoch in range(self.start_epoch, self.cfg.pretrain.num_epochs+1):
            print(f"epoch: {epoch}")
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
                
            for batch in tqdm.tqdm(self.train_loader, disable=not is_main_process()):   
                # forward
                self.model.train()
            
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                batch = self._collate(batch, f=self.cfg.frame)
            
                self.optimizer.zero_grad()    
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, train_logs = self.compute_loss(batch)
        
                # backward
                self.scaler.scale(loss).backward()

                # gradient clipping
                #  the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                if self.cfg.pretrain.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.pretrain.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
                # perform custom update function
                if (step % self.cfg.pretrain.target_update_every == 0) and (self.cfg.pretrain.target_update_every != -1):
                    self.update(batch, step)

                # log        
                grad_stats = get_grad_norm_stats(self.model)
                scheduler_logs = {}
                scheduler_logs['learning_rate'] = self.lr_scheduler.get_lr()[0]
                train_logs.update(grad_stats)
                train_logs.update(scheduler_logs)

                self.logger.update_log(**train_logs)
                if step % self.cfg.pretrain.log_every == 0:
                    self.logger.write_log(step)
                    
                # proceed
                self.lr_scheduler.step()
                step += 1

            torch.cuda.empty_cache()

            # Checkpoint save
            if (epoch % self.cfg.pretrain.save_every == 0) and (self.cfg.pretrain.save_every != -1):
                self.save_checkpoint(epoch, step)

            # Eval log
            epoch_log = {'epoch': epoch}
            self.logger.update_log(**epoch_log)
            
            
            if (epoch % self.cfg.pretrain.eval_every == 0) and (self.cfg.pretrain.eval_every != -1):
                if is_main_process():
                    self.model.eval()
                    eval_logs = self.evaluate(epoch)
                    self.logger.update_log(**eval_logs)
                if self.cfg.pretrain.distributed:
                    dist.barrier() # ensure all ranks wait until eval is done before continuing
                        
            self.logger.write_log(step)
    

    def save_checkpoint(self, epoch, step):
        if self.logger.rank == 0:
            name = 'epoch'+str(epoch)
            save_dict = {'backbone': self.model.backbone.state_dict(),
                          'neck': self.model.neck.state_dict(),
                          'head': self.model.head.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'lr_scheduler': self.lr_scheduler.state_dict(),
                          'scaler': self.scaler.state_dict(),
                          'epoch': epoch,
                          'step': step,}
            self.logger.save_dict(save_dict=save_dict,
                                  name=name)
    
    def evaluate(self, epoch):
        raise NotImplementedError
