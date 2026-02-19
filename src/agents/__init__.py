import torch

from .base import BaseAgent
from .buffers import *
from dotmap import DotMap
from omegaconf import OmegaConf
from src.common.augmentation import Augmentation
from src.common.class_utils import all_subclasses, import_all_subclasses
from .ppo import PPOAgent
import_all_subclasses(__file__, __name__, BaseAgent)

AGENTS = {subclass.get_name():subclass
          for subclass in all_subclasses(BaseAgent)}

BUFFERS = {subclass.get_name():subclass
           for subclass in all_subclasses(BaseBuffer)}

def build_agent(cfg,
                device,
                train_env,
                eval_env,
                logger,
                model):
    
    # to_container: omegaconf -> dictionary
    # resolve: interpoation to actual value
    cfg = DotMap(OmegaConf.to_container(cfg, resolve=True))

    # augemntation
    if len(cfg.aug_types) == 0:
        cfg.aug_types = []
    aug_func = Augmentation(obs_shape=cfg.obs_shape, 
                            aug_types=cfg.aug_types)
    
    # Get the shape of the output after the backbone for buffer initialization
    with torch.no_grad():
        dummy_obs = torch.zeros((2, 3, *cfg.obs_shape)).to(device)
        out, _ = model.backbone(dummy_obs)
        backbone_feat_shape = out.shape[2:]

    # buffer
    buffer_cfg = cfg['buffer']
    buffer_type = buffer_cfg.pop('type')
    if buffer_type != str(None):
        buffer = BUFFERS[buffer_type]
        buffer = buffer(device=device, backbone_feat_shape=backbone_feat_shape, **buffer_cfg)
    else:
        buffer = None

    agent_type = cfg.pop('type')
    if agent_type == 'ppo':
        agent = PPOAgent
    else:   
        agent = AGENTS[agent_type]

    return agent(cfg=cfg,
                 device=device,
                 train_env=train_env,
                 eval_env=eval_env,
                 logger=logger,
                 buffer=buffer,
                 aug_func=aug_func,
                 model=model)
