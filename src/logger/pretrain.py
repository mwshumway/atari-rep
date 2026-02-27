import wandb
from dataclasses import asdict
import numpy as np
import torch

from src.env.atari import *

class PretrainLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = cfg.rank
        if self.cfg.wandb.enabled and self.rank == 0:
            run_id = self.cfg.wandb.run_id if self.cfg.wandb.run_id else None
            wandb.init(
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                entity=self.cfg.wandb.entity,
                group=self.cfg.wandb.group,
                id=run_id,
                resume="allow",
                config=asdict(self.cfg)
            )
        self.logger = TrainerLogger()
    
    def update_log(self, **kwargs):
        self.logger.update_log(**kwargs)
    
    def write_log(self, step):
        log_data = self.logger.fetch_log()

        if self.rank == 0 and self.cfg.wandb.enabled:
            wandb.log(log_data, step=step)

    def save_dict(self, save_dict, name):
        # 3. Only save files on Rank 0
        if self.rank == 0:
            path = f'./{self.cfg.pretrain.checkpoint_dir}/{self.cfg.project_name}/{self.cfg.group_name}/{self.cfg.exp_name}_{self.run_id}/{name}.pth'
            _dir = os.path.dirname(path)
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            
            torch.save(save_dict, path)
            print("Checkpoint saved successfully at", path)


class TrainerLogger(object):
    def __init__(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_set.update(k, v)
            else:
                self.media_set[k] = v

    def fetch_log(self):
        log_data = {}
        log_data.update(self.average_meter_set.averages())
        log_data.update(self.media_set)
        self.reset()
        
        return log_data

    def reset(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)