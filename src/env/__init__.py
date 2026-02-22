import copy
from dataclasses import asdict

from .base import BaseEnv
from .atari import AtariEnv
from .vec_env import VecEnv
from src.utils.class_utils import all_subclasses

ENVS = {subclass.get_name():subclass
        for subclass in all_subclasses(BaseEnv)}

def build_env(cfg):
    cfg_dict = asdict(cfg)
    env_type = cfg_dict.pop('type')
    env = ENVS[env_type]

    num_train_envs = cfg_dict.pop('num_train_envs')
    num_eval_envs = cfg_dict.pop('num_eval_envs')

    if num_train_envs > 1:
        raise NotImplementedError('For training, only 1 train_env is supported')
    
    train_env = env(**cfg_dict)

    eval_envs = []
    for idx in range(num_eval_envs):
        _cfg_dict = copy.deepcopy(cfg_dict)
        _cfg_dict['seed'] = cfg_dict['seed'] + idx
        eval_env = env(**_cfg_dict)
        eval_envs.append(eval_env)
    
    eval_env = VecEnv(num_processes = num_eval_envs, envs = eval_envs)

    return train_env, eval_env