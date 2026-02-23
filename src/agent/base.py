from abc import ABCMeta, abstractmethod
import torch.optim as optim
from dataclasses import asdict

from src.utils.schedulers import LinearScheduler, CosineScheduler, ExponentialScheduler


class BaseAgent(metaclass=ABCMeta):
    name = "base_agent"

    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 model,
                 buffer,
                 logger,
                 aug_func):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.model = model.to(device)
        self.logger = logger
        self.buffer = buffer
        self.aug_func = aug_func.to(device)

        self.game_id = self.train_env.game_id

        # Schedulers
        self.prior_weight_scheduler = self._build_scheduler(self.cfg.prior_weight_scheduler)
        self.eps_scheduler = self._build_scheduler(self.cfg.eps_scheduler)
        self.gamma_scheduler = self._build_scheduler(self.cfg.gamma_scheduler)
        self.n_step_scheduler = self._build_scheduler(self.cfg.n_step_scheduler)

        # Optimizer
        self.optimizer = self._build_optimizer(self.model.parameters(), asdict(self.cfg.optimizer))
    
    @classmethod
    def get_name(cls):
        return cls.name

    def _build_scheduler(self, scheduler_cfg):
        scheduler_type = scheduler_cfg.type
        if scheduler_type == "linear":
            return LinearScheduler(scheduler_cfg.initial_value, scheduler_cfg.final_value, scheduler_cfg.max_step)
        elif scheduler_type == "cosine":
            return CosineScheduler(scheduler_cfg.initial_value, scheduler_cfg.final_value, scheduler_cfg.max_step)
        elif scheduler_type == "exponential":
            return ExponentialScheduler(scheduler_cfg.initial_value, scheduler_cfg.final_value, scheduler_cfg.max_step, reverse=scheduler_cfg.reverse)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    def _build_optimizer(self, param_group, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(param_group, 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(param_group, 
                               **optimizer_cfg)
        elif optimizer_type == 'sgd':
            return optim.SGD(param_group, 
                              **optimizer_cfg)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(param_group, 
                                 **optimizer_cfg)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    @abstractmethod
    def train(self):
        pass