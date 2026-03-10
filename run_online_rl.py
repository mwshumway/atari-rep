from src.agent import build_agent
from src.env import build_env
from src.model import build_model
from src.logger import RainbowLogger
from configs import BaseConfig
from src.utils.seed import set_global_seeds
from src.data import build_dataloader, download_data

import torch
import tyro
import sys
import wandb


def apply_spr_baseline_compat(cfg):
    """Apply SPR-like online RL defaults without depending on rlpyt."""
    cfg.backbone.type = "nature"
    cfg.neck.type = "identity"

    cfg.optimizer.type = "adam"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.eps = 1.5e-4
    cfg.optimizer.betas = [0.9, 0.999]
    cfg.optimizer.weight_decay = 0.0

    cfg.agent.rep = False
    cfg.agent.batch_size = 32
    cfg.agent.min_buffer_size = 2_000
    cfg.agent.optimize_per_env_step = 2
    cfg.agent.target_tau = 0.0
    cfg.agent.exploration_model = "online"
    cfg.agent.update_buffer = False
    cfg.agent.eval_eps = 0.001

    cfg.buffer.prior_exp = 0.5
    cfg.n_step = 10
    cfg.n_step_scheduler.initial_value = 10
    cfg.n_step_scheduler.final_value = 10
    cfg.gamma = 0.99
    cfg.gamma_scheduler.initial_value = 0.99
    cfg.gamma_scheduler.final_value = 0.99

    cfg.eps_scheduler.initial_value = 1.0
    cfg.eps_scheduler.final_value = 0.0

    cfg.agent.rollout_freq = 10_000
    cfg.agent.eval_freq = -1

    cfg.eval_env.repeat_action_probability = 0.0

def main(cfg):
    train_env = None
    eval_env = None
    return_code = 0

    if cfg.agent.spr_baseline_compat:
        apply_spr_baseline_compat(cfg)

    total_optimize_steps = (cfg.agent.num_timesteps - cfg.agent.min_buffer_size) * cfg.agent.optimize_per_env_step // cfg.num_train_envs
    cfg.prior_weight_scheduler.max_step = total_optimize_steps
    if cfg.agent.spr_baseline_compat:
        cfg.eps_scheduler.max_step = min(total_optimize_steps, 2_001)
    else:
        cfg.eps_scheduler.max_step = int(total_optimize_steps * 0.1)
    cfg.gamma_scheduler.max_step = total_optimize_steps
    cfg.n_step_scheduler.max_step = total_optimize_steps

    try:
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

        # Get off policy probing dataset (optional)
        if cfg.agent.probe_off_policy_freq > 0:
            download_data(cfg) # downloads if not already done, else skips
            _, _, eval_dataloader, _ = build_dataloader(cfg)
        else:
            eval_dataloader = None

        # Build agent
        agent = build_agent(cfg, device, train_env, eval_env, logger, model, eval_dataloader)

        # Train agent
        agent.train()
    except KeyboardInterrupt:
        return_code = 130
    except Exception:
        return_code = 1
        raise
    finally:
        if cfg.wandb.enabled:
            try:
                wandb.finish(exit_code=return_code)
            except TypeError:
                wandb.finish()

        if train_env is not None:
            train_env.close()

        if eval_env is not None:
            eval_env.close()

    return return_code


if __name__ == "__main__":
    cfg = tyro.cli(BaseConfig)
    sys.exit(main(cfg))