import pytest
import torch

from src.agent import build_agent

@pytest.fixture
def cfg():
    from configs.base import BaseConfig
    cfg = BaseConfig()
    cfg.games = ["pong"]
    total_optimize_steps = (cfg.agent.num_timesteps - cfg.agent.min_buffer_size) * cfg.agent.optimize_per_env_step // cfg.num_train_envs
    cfg.prior_weight_scheduler.max_step = total_optimize_steps
    cfg.eps_scheduler.max_step = total_optimize_steps
    cfg.gamma_scheduler.max_step = total_optimize_steps
    cfg.n_step_scheduler.max_step = total_optimize_steps
    return cfg

@pytest.fixture
def model(cfg):
    from src.model import build_model
    device = torch.device("cpu")
    model = build_model(cfg, device)
    return model

@pytest.fixture
def logger(cfg):
    from src.logger import RainbowLogger
    logger = RainbowLogger(cfg)
    return logger

@pytest.fixture
def train_env(cfg):
    from src.env import build_env
    train_env, _ = build_env(cfg)
    return train_env

@pytest.fixture
def eval_env(cfg):
    from src.env import build_env
    _, eval_env = build_env(cfg)
    return eval_env


def test_build_agent(cfg, model, logger, train_env, eval_env):
    device = torch.device("cpu")
    agent = build_agent(cfg, device, train_env, eval_env, logger, model)
    assert agent is not None

