import copy
import torch
import torch.distributed as dist
from einops import rearrange
from dataclasses import asdict  

from .base import BaseTrainer
from src.utils.schedulers import LinearScheduler


class ATCTrainer(BaseTrainer):
    name = 'atc'
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
        # forward
        x = batch['obs']
        target_x = batch['next_obs']
        game_id = batch['game_id']
        done = batch['done'].contiguous()

        # augmentation
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        target_x = rearrange(target_x, 'n t f c h w -> n (t f c) h w')
        target_x = self.aug_func(target_x)
        target_x = rearrange(target_x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online backbone/neck/head
        x, _ = self.model.backbone(x)
        x, _ = self.model.neck(x, game_id) # game-wise spatial embedding
        pred_x, _ = self.model.head(x, game_id)
        pred_x = rearrange(pred_x, 'n t d -> (n t) d')

        # momentum backbone/neck
        with torch.no_grad():
            target_x, _ = self.target_backbone(target_x)
            target_x, _ = self.target_neck(target_x, game_id)
            target_x = rearrange(target_x, 'n t d -> (n t) d')

        # DDP gather
        if self.cfg.pretrain.distributed:
            pred_x_list = [torch.empty_like(pred_x) for _ in range(self.cfg.num_gpus_per_node)]
            target_x_list = [torch.empty_like(target_x) for _ in range(self.cfg.num_gpus_per_node)]
            done_list = [torch.empty_like(done) for _ in range(self.cfg.num_gpus_per_node)]
            # Share variables for loss computation across all GPUs
            dist.all_gather(tensor=pred_x, tensor_list=pred_x_list)
            dist.all_gather(tensor=target_x, tensor_list=target_x_list)
            dist.all_gather(tensor=done, tensor_list=done_list)
            # all_gather() purely copies the value, and the resulting tensor will have no computational graph.
            # To maintain the gradient graph, each GPU must plaster their own tensor (with computational graph) in the corresponding place.
            pred_x_list[self.logger.rank] = pred_x
            pred_x = torch.cat(pred_x_list, dim=0).requires_grad_()
            target_x = torch.cat(target_x_list, dim=0)
            done = torch.cat(done_list, dim=0)
        
        similarity = pred_x @ target_x.t()
        similarity = similarity / self.cfg.pretrain.temperature

        gt_label = torch.arange(similarity.shape[0], device=similarity.device)
        invalid_data = torch.any(done, dim=1)
        gt_label[invalid_data] = -100 # ignore loss on invalid data (done=True)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(similarity, gt_label)
        self_pred_cnt = (similarity.argmax(dim=1) == gt_label).sum().item()
        self_pred_acc = self_pred_cnt / float(similarity.shape[0] - invalid_data.sum().item())

        log_data = {
            "loss": loss.item(),
            "self_pred_acc": self_pred_acc,
        }

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