from src.agent import build_agent
from src.env import build_env
from src.model import build_model
from src.logger import RainbowLogger
from configs import BaseConfig
from src.utils.seed import set_global_seeds

import torch
import tyro
import sys
import wandb

def main(cfg):
    total_optimize_steps = (cfg.agent.num_timesteps - cfg.agent.min_buffer_size) * cfg.agent.optimize_per_env_step // cfg.num_train_envs
    cfg.prior_weight_scheduler.max_step = total_optimize_steps
    cfg.eps_scheduler.max_step = int(total_optimize_steps * 0.1)
    cfg.gamma_scheduler.max_step = total_optimize_steps
    cfg.n_step_scheduler.max_step = total_optimize_steps

    # Set random seed
    set_global_seeds(cfg.seed)
    
    # Build envs
    train_env, eval_env = build_env(cfg)

    cfg.action_size = train_env.action_space.n

    # Build model
    device = torch.device(cfg.device)
    model = build_model(cfg, device)

    # Build logger
    logger = RainbowLogger(cfg)

    # Build agent
    agent = build_agent(cfg, device, train_env, eval_env, logger, model)

    # Train agent
    agent.train()
    
    if cfg.wandb.enabled:
        wandb.finish()
    
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    cfg = tyro.cli(BaseConfig)
    sys.exit(main(cfg))