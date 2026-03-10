import argparse
import os
import statistics
import subprocess
from typing import Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None


VARIANT_FLAGS: Dict[str, List[str]] = {
    "legacy": [],
    "spr_compat": ["--agent.spr_baseline_compat"],
}


def make_run_cmd(game: str, seed: int, variant: str, project: str, entity: Optional[str], prefix: str) -> str:
    flags = VARIANT_FLAGS[variant]
    name = f"{prefix}_{variant}_{game}_seed{seed}"

    cmd = [
        "python", "run_online_rl.py",
        "--games", game,
        "--seed", str(seed),
        "--pretrain.type", "baseline",
        "--wandb.enabled",
        "--wandb.project", project,
        "--wandb.group", f"{prefix}_{variant}",
        "--wandb.name", name,
        "--agent.probe_on_policy_freq", "-1",
        "--agent.probe_off_policy_freq", "-1",
        "--agent.rollout_freq", "10000",
        "--agent.eval_freq", "-1",
        "--agent.save_freq", "-1",
        "--agent.num_timesteps", "100000",
        "--backbone.type", "nature",
        "--neck.type", "identity",
        "--env.repeat_action_probability", "0.0",
        "--eval_env.repeat_action_probability", "0.0",
    ] + flags

    if entity:
        cmd.extend(["--wandb.entity", entity])

    return " \\\n    ".join(cmd)


def make_qsub_script(command: str, job_name: str, log_dir: str, sge_project: str) -> str:
    return f"""#!/bin/bash -l
#$ -P {sge_project}
#$ -N {job_name}
#$ -cwd
#$ -o {log_dir}/
#$ -e {log_dir}/
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=V100
#$ -l h_rt=12:00:00

set -euo pipefail

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

{command}
"""


def launch_runs(args):
    os.makedirs(args.job_script_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    for game in args.games:
        for seed in args.seeds:
            for variant in args.variants:
                run_cmd = make_run_cmd(game, seed, variant, args.project, args.entity, args.group_prefix)
                job_name = f"gap_{variant[:6]}_{seed}"
                script = make_qsub_script(run_cmd, job_name, os.path.abspath(args.log_dir), args.sge_project)

                script_path = os.path.join(
                    args.job_script_dir,
                    f"submit_{args.group_prefix}_{variant}_{game}_seed{seed}.sh",
                )
                with open(script_path, "w", encoding="utf-8") as file:
                    file.write(script)

                if args.mode == "print":
                    print(f"\n=== {variant} | {game} | seed {seed} ===")
                    print(run_cmd)
                elif args.mode == "local":
                    print(f"Running local: {variant} | {game} | seed {seed}")
                    subprocess.run(run_cmd, shell=True, check=True)
                else:
                    print(f"Submitting: {variant} | {game} | seed {seed}")
                    result = subprocess.run(["qsub"], input=script, text=True, capture_output=True, check=False)
                    if result.returncode == 0:
                        print(result.stdout.strip())
                    else:
                        print(result.stderr.strip())


def collect_metric_from_runs(runs, metric: str) -> List[Tuple[str, float]]:
    values = []
    for run in runs:
        if metric in run.summary:
            game = "unknown"
            games = run.config.get("games", None)
            if isinstance(games, list) and len(games) > 0:
                game = str(games[0])
            values.append((game, float(run.summary[metric])))
    return values


def summarize_runs(args):
    if wandb is None:
        raise ImportError("wandb is required for summarize mode: pip install wandb")

    entity = args.entity or os.environ.get("WANDB_ENTITY", "")
    if not entity:
        raise ValueError("Entity is required for summarize mode. Pass --entity or set WANDB_ENTITY.")

    api = wandb.Api()
    path = f"{entity}/{args.project}"

    grouped_results: Dict[str, List[Tuple[str, float]]] = {}
    for variant in args.variants:
        group_name = f"{args.group_prefix}_{variant}"
        runs = api.runs(path=path, filters={"group": group_name})
        grouped_results[variant] = collect_metric_from_runs(runs, args.metric)

    print("\n=== Baseline Gap Summary ===")
    print(f"Project: {path}")
    print(f"Metric: {args.metric}")

    for variant in args.variants:
        entries = grouped_results[variant]
        vals = [v for _, v in entries]
        if len(vals) == 0:
            print(f"- {variant}: no runs found")
            continue
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"- {variant}: n={len(vals)}, mean={mean:.4f}, std={std:.4f}")

    if "legacy" in grouped_results and "spr_compat" in grouped_results:
        legacy_vals = [v for _, v in grouped_results["legacy"]]
        compat_vals = [v for _, v in grouped_results["spr_compat"]]
        if len(legacy_vals) > 0 and len(compat_vals) > 0:
            legacy_mean = statistics.mean(legacy_vals)
            compat_mean = statistics.mean(compat_vals)
            delta = compat_mean - legacy_mean
            rel = (delta / (abs(legacy_mean) + 1e-8)) * 100.0
            print(f"\nDelta (spr_compat - legacy): {delta:.4f} ({rel:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Ablate atari-rep baseline gap: legacy vs SPR-compat")
    parser.add_argument("--action", choices=["launch", "summarize"], default="launch")
    parser.add_argument("--mode", choices=["qsub", "local", "print"], default="qsub")
    parser.add_argument("--variants", nargs="+", default=["legacy", "spr_compat"], choices=list(VARIANT_FLAGS.keys()))
    parser.add_argument("--games", nargs="+", default=["seaquest"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--project", type=str, default="spr_baseline_gap")
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--group-prefix", type=str, default="gap_ablation")
    parser.add_argument("--metric", type=str, default="eval/mean_traj_game_scores")
    parser.add_argument("--sge-project", type=str, default="ds598xz")
    parser.add_argument("--job-script-dir", type=str, default="job_scripts")
    parser.add_argument("--log-dir", type=str, default="job_logs")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "launch":
        launch_runs(args)
    else:
        summarize_runs(args)


if __name__ == "__main__":
    main()
