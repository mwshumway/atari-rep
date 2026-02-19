from .backbones import *  
from .necks import *
from .heads import * 
from .base import Model
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from src.common.class_utils import all_subclasses
import torch
from dataclasses import asdict


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

NECKS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseNeck)}

HEADS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseHead)}


def build_model(cfg, device: torch.device):
    if isinstance(cfg, OmegaConf) or isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=False)
    else:
        cfg = asdict(cfg)
    backbone_cfg = cfg['backbone']
    neck_cfg = cfg['neck']
    head_cfg = cfg['head']
    
    backbone_type = backbone_cfg.pop('type')
    print(f'Building model with backbone: {backbone_type}')
    neck_type = neck_cfg.pop('type')
    head_type = head_cfg.pop('type')

    # backbone
    backbone_cfg['in_shape'] = backbone_cfg['in_shape'] or cfg['obs_shape']
    backbone_cfg['action_size'] = backbone_cfg['action_size'] or cfg['action_size']
    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg)
    fake_obs = torch.zeros((2, 3, *backbone_cfg['in_shape']))
    out, _ = backbone(fake_obs)

    print(f'Backbone output shape: {out.shape}')

    # neck
    neck_cfg['in_shape'] = neck_cfg['in_shape'] or out.shape[2:]
    neck_cfg['action_size'] = neck_cfg['action_size'] or cfg['action_size']
    neck = NECKS[neck_type]
    neck = neck(**neck_cfg)
    out, neck_reps = neck(out)

    for k, v in neck_reps.items():
        print(f'Neck representation - {k}: {v.shape}')

    # head
    head_cfg['in_shape'] = head_cfg['in_shape'] or out.shape[2:]
    head_cfg['action_size'] = head_cfg['action_size'] or cfg['action_size']
    head = HEADS[head_type]
    head = head(**head_cfg)

    # Get Parameter Counts
    backbone_params = sum(p.numel() for p in backbone.parameters())
    neck_params = sum(p.numel() for p in neck.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    print(f'Backbone parameters: {backbone_params}')
    print(f'Neck parameters: {neck_params}')
    print(f'Head parameters: {head_params}')
    print(f'Total model parameters: {backbone_params + neck_params + head_params}')
    
    # model
    model = Model(backbone=backbone, neck=neck, head=head)

    return model.to(device)