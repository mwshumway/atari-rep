from .base import BaseAgent
from .rainbow import RainbowAgent
from .buffer import BaseBuffer, PERBuffer

from src.utils.class_utils import all_subclasses
from src.utils.augmentation import Augmentation

from dataclasses import asdict

AGENTS = {subclass.get_name():subclass
          for subclass in all_subclasses(BaseAgent)}

BUFFERS = {subclass.get_name():subclass
           for subclass in all_subclasses(BaseBuffer)}

def build_agent(cfg, device, train_env, eval_env, logger, model):
    agent_type = cfg.agent.type
    agent = AGENTS[agent_type]

    if len(cfg.aug_types) == 0:
        cfg.aug_types = []
    aug_func = Augmentation(obs_shape=cfg.obs_shape, aug_types=cfg.aug_types)

    buffer_cfg = asdict(cfg.buffer)
    buffer_cfg['obs_shape'] = cfg.obs_shape
    buffer_cfg['action_size'] = cfg.action_size
    buffer_type = buffer_cfg.pop('type')
    buffer = BUFFERS[buffer_type](device=device, **buffer_cfg)

    return agent(cfg=cfg, 
                 device=device,
                 train_env=train_env,
                 eval_env=eval_env,
                 model=model,
                 buffer=buffer,
                 logger=logger,
                 aug_func=aug_func)