import tyro
import torch
import tqdm
import os
import sys

from einops import rearrange

from src.utils.seed import set_global_seeds
from src.env import build_env
from src.model import build_model
from src.agent import build_agent
from src.logger import RainbowLogger
from src.data import download_data, build_dataloader
from configs import BaseConfig

from src.probe.probe_utils import create_probe_dataset, _collate
from src.probe.value import train_value_probe
import random


def analyze_dataset_statistics(dataset_list, reward_values):
    if len(dataset_list) == 0:
        raise ValueError("Dataset is empty. Cannot analyze reward sparsity or value distribution.")

    value_targets = torch.tensor([float(x[2]) for x in dataset_list], dtype=torch.float32)
    reward_tensor = torch.tensor(reward_values, dtype=torch.float32)

    quantile_levels = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float32)
    quantiles = torch.quantile(value_targets, quantile_levels)

    reward_sparsity = float("nan")
    if reward_tensor.numel() > 0:
        reward_sparsity = (reward_tensor == 0).float().mean().item()

    return {
        "num_samples": len(dataset_list),
        "num_rewards": int(reward_tensor.numel()),
        "reward_sparsity": reward_sparsity,
        "value_mean": value_targets.mean().item(),
        "value_std": value_targets.std(unbiased=False).item(),
        "value_min": value_targets.min().item(),
        "value_p10": quantiles[0].item(),
        "value_p25": quantiles[1].item(),
        "value_p50": quantiles[2].item(),
        "value_p75": quantiles[3].item(),
        "value_p90": quantiles[4].item(),
        "value_max": value_targets.max().item(),
    }


def print_dataset_analysis(analysis):
    print("Dataset Analysis:")
    print(f"  Num Samples: {analysis['num_samples']}")
    print(f"  Num Rewards: {analysis['num_rewards']}")
    print(f"  Reward Sparsity (fraction of zero rewards): {analysis['reward_sparsity']}")
    print("  Value Distribution:")
    print(f"    Mean: {analysis['value_mean']}")
    print(f"    Std: {analysis['value_std']}")
    print(f"    Min: {analysis['value_min']}")
    print(f"    P10: {analysis['value_p10']}")
    print(f"    P25: {analysis['value_p25']}")
    print(f"    P50: {analysis['value_p50']}")
    print(f"    P75: {analysis['value_p75']}")
    print(f"    P90: {analysis['value_p90']}")
    print(f"    Max: {analysis['value_max']}")


def validate_probe_checkpoint_setup(cfg, checkpoint_dir):
    backbone_path = os.path.join(checkpoint_dir, "backbone.pt")
    has_backbone_checkpoint = os.path.exists(backbone_path)
    backbone_frozen = "backbone" in cfg.load_model.freeze_layers

    if not backbone_frozen and not has_backbone_checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} has no backbone.pt, but cfg.load_model.freeze_layers={cfg.load_model.freeze_layers}. "
            "This implies probing will try to load a backbone checkpoint that does not exist. "
            "Use --load_model.freeze_layers backbone and provide the same pretrained backbone settings used during RL training."
        )

    if backbone_frozen and not cfg.load_model.enable:
        raise ValueError(
            "Backbone is frozen during probing but cfg.load_model.enable is False. "
            "This means the backbone stays randomly initialized and probe performance (including R2) can collapse. "
            "Enable --load_model.enable and set --load_model.model_path to the pretrained checkpoint used for policy training."
        )

    if backbone_frozen and has_backbone_checkpoint:
        print(
            f"[Warning] {checkpoint_dir} includes backbone.pt, but probing is configured with frozen backbone. "
            "The saved backbone will be ignored and probing will rely on cfg.load_model.model_path backbone weights."
        )


def initialize_agent(cfg):
    total_optimize_steps = (cfg.agent.num_timesteps - cfg.agent.min_buffer_size) * cfg.agent.optimize_per_env_step // cfg.num_train_envs
    cfg.prior_weight_scheduler.max_step = total_optimize_steps
    cfg.eps_scheduler.max_step = int(total_optimize_steps * 0.1)
    cfg.gamma_scheduler.max_step = total_optimize_steps
    cfg.n_step_scheduler.max_step = total_optimize_steps
    
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

    return agent


def online_probe(cfg, checkpoint_dir):
    ### Load the policy ###
    validate_probe_checkpoint_setup(cfg, checkpoint_dir)
    agent = initialize_agent(cfg)
    agent.load_progress(agent.model, checkpoint_dir)
    outer_step = checkpoint_dir.split("/")[-1][4:]  # Extract outer step from filename (after "step...")

    ### Collect trajectories with the loaded policy ### 
    env_trajectories = [[] for _ in range(cfg.num_eval_envs)]

    obs = agent.eval_env.reset()
    game_id = torch.full((cfg.num_eval_envs, 1), agent.game_id, dtype=torch.long, device=agent.device)

    for _ in tqdm.tqdm(range(cfg.agent.max_rollout_steps), desc="Collecting trajectories for probing"):
        obs_tensor = agent.buffer.encode_obs(obs, prediction=True).to(agent.device)
        n, t, num_envs, f, c, h, w = obs_tensor.shape
        obs_tensor = rearrange(obs_tensor, "n t num_envs f c h w -> (n num_envs) t f c h w")

        with torch.no_grad():
            backbone_feat, _ = agent.model.backbone(obs_tensor)
            if cfg.agent.rep:
                _, neck_info = agent.model.neck(backbone_feat, game_id=game_id)
                neck_feat = neck_info[cfg.agent.rep_candidate]
            else:
                neck_feat, _ = agent.model.neck(backbone_feat, game_id=game_id)

        action = agent.predict(agent.model, backbone_feat, eps=cfg.agent.eval_eps, n=n * num_envs, t=t)

        next_obs, reward, done, info = agent.eval_env.step(action.reshape(-1))
        agent.logger.step(obs, reward, done, info, mode="probe")

        neck_cpu = neck_feat.cpu().view(num_envs, -1)
        actions_list = action.reshape(-1).tolist()

        for i in range(num_envs):
            env_trajectories[i].append((neck_cpu[i], actions_list[i], float(reward[i]), bool(done[i])))
        
        if agent.logger.is_traj_done(mode="probe"):
            break

        obs = next_obs
    
    dataset = create_probe_dataset(env_trajectories, cfg)
    # Sample 12435 points to match offline dataset size
    if len(dataset) > 12435:
        dataset = random.sample(dataset, 12435)
    reward_values = [float(transition[2]) for traj in env_trajectories for transition in traj]

    if cfg.probe.analyze_only:
        analysis = analyze_dataset_statistics(dataset_list=dataset, reward_values=reward_values)
        print_dataset_analysis(analysis)
        agent.logger.probe_logger.reset()
        return analysis

    train_mse, test_mse, train_r2, test_r2 = train_value_probe(cfg=cfg, dataset_list=dataset, outer_step=outer_step, device=agent.device, log_wandb=False, plot_curves=True)
    agent.logger.probe_logger.reset()

    return {"train_mse": train_mse, "test_mse": test_mse, "train_r2": train_r2, "test_r2": test_r2}


def offline_probe(cfg, checkpoint_dir):
    ### Since the representations move with the policy, we still load the policy checkpoints to access the neck ###
    validate_probe_checkpoint_setup(cfg, checkpoint_dir)
    agent = initialize_agent(cfg)
    agent.load_progress(agent.model, checkpoint_dir)
    outer_step = checkpoint_dir.split("/")[-1][4:]  # Extract outer step from filename (after "step...")

    # Load the offline dataset for probing
    download_data(cfg)
    _, _, eval_dataloader, _ = build_dataloader(cfg)

    dataset_list = []
    reward_values = []
    for batch in tqdm.tqdm(eval_dataloader, desc="Processing offline dataset for probing"):
        batch = _collate(batch, f=cfg.frame)
        obs = batch["obs"].to(agent.device) # (n, t+f-1, c, h, w)
        game_id = batch["game_id"].to(agent.device)
        rtg = batch["rtg"].to(agent.device)
        action = batch["act"].to(agent.device)
        reward = batch["rew"].to(agent.device)
        # game_id = rearrange(game_id, "n t -> (n t)")
        rtg = rearrange(rtg, "n t -> (n t)")
        action = rearrange(action, "n t -> (n t)")
        reward = rearrange(reward, "n t -> (n t)")
        reward_values.extend(reward.cpu().tolist())

        with torch.no_grad():
            backbone_feat, _ = agent.model.backbone(obs)
            if cfg.agent.rep:
                _, neck_info = agent.model.neck(backbone_feat, game_id=game_id)
                neck_feat = neck_info[cfg.agent.rep_candidate]
            else:
                neck_feat, _ = agent.model.neck(backbone_feat, game_id=game_id)
        neck_cpu = neck_feat.cpu().view(neck_feat.shape[0] * neck_feat.shape[1], -1)

        if neck_cpu.shape[0] != action.shape[0] or neck_cpu.shape[0] != rtg.shape[0]:
            raise ValueError(
                f"Offline probe sample mismatch: neck={neck_cpu.shape[0]}, action={action.shape[0]}, rtg={rtg.shape[0]}. "
                "Expected all to match after flattening to (n*t)."
            )

        for i in range(neck_cpu.shape[0]):
            dataset_list.append((neck_cpu[i], action[i].cpu(), rtg[i].cpu()))

    if cfg.probe.analyze_only:
        analysis = analyze_dataset_statistics(dataset_list=dataset_list, reward_values=reward_values)
        print_dataset_analysis(analysis)
        return analysis
    
    train_mse, test_mse, train_r2, test_r2 = train_value_probe(cfg=cfg, dataset_list=dataset_list, outer_step=outer_step, device=agent.device, log_wandb=False, plot_curves=True)
    return {"train_mse": train_mse, "test_mse": test_mse, "train_r2": train_r2, "test_r2": test_r2}
        

def main(cfg):
    set_global_seeds(cfg.seed)
    if cfg.probe.type == "online":
        policy_ckpt_dirs = [os.path.join(cfg.probe.policy_ckpt_dir, d) for d in os.listdir(cfg.probe.policy_ckpt_dir) if d.startswith("step")]
        policy_ckpt_dirs.sort(key=lambda x: int(x.split("/")[-1][4:]))  # Sort by outer step
        results = []
        for ckpt_dir in policy_ckpt_dirs:
            print(f"Probing checkpoint: {ckpt_dir}")
            cfg.probe.checkpoint_path = ckpt_dir
            result = online_probe(cfg, checkpoint_dir=ckpt_dir)
            results.append((ckpt_dir, result))

        if cfg.probe.analyze_only:
            avg_analysis = {key: sum(metrics[key] for _, metrics in results) / len(results) for key in results[0][1].keys()}
            print("Average Dataset Analysis:")
            print_dataset_analysis(avg_analysis)

            if cfg.probe.save_dir:
                os.makedirs(cfg.probe.save_dir, exist_ok=True)
                with open(os.path.join(cfg.probe.save_dir, "online_probe_analysis.txt"), "a") as f:
                    f.write("Checkpoints:\n")
                    for checkpoint_path, checkpoint_metrics in results:
                        f.write(f"  {checkpoint_path}: {checkpoint_metrics}\n")
                    f.write(f"Average Analysis:\n")
                    f.write(f"  {avg_analysis}\n")
                    f.write("-" * 50 + "\n")
            return
        
        # Average results across checkpoints
        avg_train_mse = sum(metrics["train_mse"] for _, metrics in results) / len(results)
        avg_test_mse = sum(metrics["test_mse"] for _, metrics in results) / len(results)
        avg_train_r2 = sum(metrics["train_r2"] for _, metrics in results) / len(results)
        avg_test_r2 = sum(metrics["test_r2"] for _, metrics in results) / len(results)

        print(f"Average Results:")
        print(f"  Train MSE: {avg_train_mse}")
        print(f"  Test MSE: {avg_test_mse}")
        print(f"  Train R2: {avg_train_r2}")
        print(f"  Test R2: {avg_test_r2}")

        # Optionally, log results to a file or visualization tool here
        if cfg.probe.save_dir:
            os.makedirs(cfg.probe.save_dir, exist_ok=True)
            with open(os.path.join(cfg.probe.save_dir, "online_probe_results.txt"), "a") as f:
                f.write("Checkpoints:\n")
                for checkpoint_path, checkpoint_metrics in results:
                    f.write(f"  {checkpoint_path}: {checkpoint_metrics}\n")
                f.write(f"Average Results:\n")
                f.write(f"  Train MSE: {avg_train_mse}\n")
                f.write(f"  Test MSE: {avg_test_mse}\n")
                f.write(f"  Train R2: {avg_train_r2}\n")
                f.write(f"  Test R2: {avg_test_r2}\n")
                f.write("-" * 50 + "\n")
    
    elif cfg.probe.type == "offline":
        policy_ckpt_dirs = [os.path.join(cfg.probe.policy_ckpt_dir, d) for d in os.listdir(cfg.probe.policy_ckpt_dir) if d.startswith("step")]
        policy_ckpt_dirs.sort(key=lambda x: int(x.split("/")[-1][4:]))  # Sort by outer step
        results = []
        for ckpt_dir in policy_ckpt_dirs:
            print(f"Probing checkpoint: {ckpt_dir}")
            cfg.probe.checkpoint_path = ckpt_dir
            result = offline_probe(cfg, checkpoint_dir=ckpt_dir)
            results.append((ckpt_dir, result))

        if cfg.probe.analyze_only:
            avg_analysis = {key: sum(metrics[key] for _, metrics in results) / len(results) for key in results[0][1].keys()}
            print("Average Dataset Analysis:")
            print_dataset_analysis(avg_analysis)

            if cfg.probe.save_dir:
                os.makedirs(cfg.probe.save_dir, exist_ok=True)
                with open(os.path.join(cfg.probe.save_dir, "offline_probe_analysis.txt"), "a") as f:
                    f.write("Checkpoints:\n")
                    for checkpoint_path, checkpoint_metrics in results:
                        f.write(f"  {checkpoint_path}: {checkpoint_metrics}\n")
                    f.write(f"Average Analysis:\n")
                    f.write(f"  {avg_analysis}\n")
                    f.write("-" * 50 + "\n")
            return
        
        # Average results across checkpoints
        avg_train_mse = sum(metrics["train_mse"] for _, metrics in results) / len(results)
        avg_test_mse = sum(metrics["test_mse"] for _, metrics in results) / len(results)
        avg_train_r2 = sum(metrics["train_r2"] for _, metrics in results) / len(results)
        avg_test_r2 = sum(metrics["test_r2"] for _, metrics in results) / len(results)

        print(f"Average Results:")
        print(f"  Train MSE: {avg_train_mse}")
        print(f"  Test MSE: {avg_test_mse}")
        print(f"  Train R2: {avg_train_r2}")
        print(f"  Test R2: {avg_test_r2}")

        # Optionally, log results to a file or visualization tool here
        if cfg.probe.save_dir:
            os.makedirs(cfg.probe.save_dir, exist_ok=True)
            with open(os.path.join(cfg.probe.save_dir, "offline_probe_results.txt"), "a") as f:
                f.write("Checkpoints:\n")
                for checkpoint_path, checkpoint_metrics in results:
                    f.write(f"  {checkpoint_path}: {checkpoint_metrics}\n")
                f.write(f"Average Results:\n")
                f.write(f"  Train MSE: {avg_train_mse}\n")
                f.write(f"  Test MSE: {avg_test_mse}\n")
                f.write(f"  Train R2: {avg_train_r2}\n")
                f.write(f"  Test R2: {avg_test_r2}\n")
                f.write("-" * 50 + "\n")

if __name__ == "__main__":
    cfg = tyro.cli(BaseConfig)
    main(cfg)