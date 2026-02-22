from .backbone import *
from .neck import *
from .head import *
from .base import Model

import torch
from dataclasses import asdict

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}
NECKS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseNeck)}
HEADS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseHead)}


def print_model_info(model, b_out, n_out, h_out, neck_reps):
    print("="*50)
    print(f"Model Info:")
    print(f"Backbone: {model.backbone.__class__.__name__}")
    print(f"Neck: {model.neck.__class__.__name__}")
    print(f"Head: {model.head.__class__.__name__}")
    print("="*50)
    print(f"Backbone output shape: {b_out.shape}")
    print(f"Neck output shape: {n_out.shape}")
    print(f"Head output shape: {h_out.shape}")
    for k, v in neck_reps.items():
        print(f"Neck representation - {k}: {v.shape}")
    print("="*50)


def build_model(cfg, device: torch.device):
    cfg_dict = asdict(cfg)
    backbone_cfg, neck_cfg, head_cfg = cfg_dict['backbone'], cfg_dict['neck'], cfg_dict['head']
    backbone_type, neck_type, head_type = backbone_cfg.pop('type'), neck_cfg.pop('type'), head_cfg.pop('type')

    backbone_cfg['in_shape'] = backbone_cfg['in_shape'] or cfg_dict['obs_shape']
    backbone_cfg['action_size'] = backbone_cfg['action_size'] or cfg_dict['action_size']
    backbone = BACKBONES[backbone_type](**backbone_cfg)

    fake_obs = torch.zeros((2, 3, *backbone_cfg['in_shape']))
    b_out, _ = backbone(fake_obs)

    neck_cfg['in_shape'] = neck_cfg['in_shape'] or b_out.shape[2:]
    neck_cfg['action_size'] = neck_cfg['action_size'] or cfg_dict['action_size']
    neck = NECKS[neck_type](**neck_cfg)
    n_out, n_info = neck(b_out)

    head_cfg['in_shape'] = head_cfg['in_shape'] or n_out.shape[2:]
    head_cfg['action_size'] = head_cfg['action_size'] or cfg_dict['action_size']
    head = HEADS[head_type](**head_cfg)
    h_out, _ = head(n_out)

    model = Model(backbone, neck, head)

    print_model_info(model, b_out, n_out, h_out, n_info)

    return model.to(device)