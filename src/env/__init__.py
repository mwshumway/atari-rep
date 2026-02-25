import copy
from dataclasses import asdict

from .base import BaseEnv
from .atari import AtariEnv
from .vec_env import VecEnv
from src.utils.class_utils import all_subclasses

ENVS = {subclass.get_name():subclass
        for subclass in all_subclasses(BaseEnv)}

def build_env(cfg):
    env_cfg = asdict(cfg.env)

    # Add any relevant fields from the main cfg to env_cfg if needed
    env_cfg['seed'] = cfg.seed
    env_cfg['frame'] = cfg.frame

    assert len(cfg.games) == 1, "Only one game should be specified in cfg.games"
    env_cfg['game'] = cfg.games[0]

    env_type = env_cfg.pop('type')
    env = ENVS[env_type]

    num_train_envs = env_cfg.pop('num_train_envs')
    num_eval_envs = env_cfg.pop('num_eval_envs')

    if num_train_envs > 1:
        raise NotImplementedError('For training, only 1 train_env is supported')
    
    train_env = env(**env_cfg)

    eval_envs = []
    for idx in range(num_eval_envs):
        _cfg_dict = copy.deepcopy(env_cfg)
        _cfg_dict['seed'] = env_cfg['seed'] + idx
        eval_env = env(**_cfg_dict)
        eval_envs.append(eval_env)
    
    eval_env = VecEnv(num_processes = num_eval_envs, envs = eval_envs)

    return train_env, eval_env