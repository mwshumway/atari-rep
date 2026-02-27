from .base import BaseTrainer
from src.utils.augmentation import Augmentation
from src.utils.class_utils import all_subclasses, import_all_subclasses

import_all_subclasses(__file__, __name__, BaseTrainer)

TRAINERS = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseTrainer)}


def build_trainer(cfg,
                  device,
                  train_loader,
                  train_sampler,
                  eval_loader,
                  eval_sampler,
                  logger,
                  model):

    if len(cfg.aug_types) == 0:
        cfg.aug_types = []

    aug_func = Augmentation(obs_shape=cfg.obs_shape, 
                            aug_types=cfg.aug_types)

    
    trainer_type = cfg.pretrain.type
    trainer = TRAINERS[trainer_type]
    return trainer(cfg=cfg,
                   device=device,
                   train_loader=train_loader,
                   train_sampler=train_sampler,
                   eval_loader=eval_loader,
                   eval_sampler=eval_sampler,
                   logger=logger,
                   aug_func=aug_func,
                   model=model)