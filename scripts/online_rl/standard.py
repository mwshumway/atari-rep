import os
import subprocess

CKPT_NUM = 100

PRETRAINED_PATHS = {
    # "cql": f"./data_storage/pretrained_models/pretrain_cql_seaquest/cql/cql-dist_resnet_seaquest_None/epoch100.pth",
    # "atc": f"./data_storage/pretrained_models/pretrain_atc_seaquest/atc/atc_resnet_seaquest_None/epoch100.pth",
    "spr": f"./data_storage/pretrained_models/pretrain_spr_seaquest/spr/spr_resnet_seaquest_None/epoch25.pth",
    # "nature-e2e": "",
    # "baseline": ""
}

SEEDS = [0]
GAMES = ["seaquest"]

REPEAT_ACTION_PROBABILITY = 0.0

SPR_BASELINE_COMPAT = False

def make_cmd(path, seed, game, pretrain_type, log_dir):
    cmd = f"""#!/bin/bash -l

# Set SCC project
#$ -P ds598xz

# Name the job in the queue
#$ -N {pretrain_type}_{seed}

# Execute the job from the current working directory
#$ -cwd

# Specify where to save the .o and .e log files
#$ -o {log_dir}/
#$ -e {log_dir}/

# Request 8 cores
#$ -pe omp 8

# Request gpu(s)
#$ -l gpus=1

# GPU type
#$ -l gpu_type=V100

# Runtime
#$ -l h_rt=12:00:00

set -euo pipefail

cleanup() {{
    status=$?
    jobs -pr | xargs -r kill -TERM || true
    sleep 2
    jobs -pr | xargs -r kill -KILL || true
    wait || true
    exit $status
}}
trap cleanup EXIT TERM INT

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8 
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=60
    
timeout --signal=TERM --kill-after=120s 42600s python -u run_online_rl.py \\
    --games {game} \\
    --seed {seed} \\
    --pretrain.type {pretrain_type} \\
    --agent.pretrain_ckpt {CKPT_NUM} \\
    --wandb.enabled \\
    --wandb.project probing_testbed \\
    --wandb.group {pretrain_type} \\
    --wandb.name {pretrain_type}_seed{seed}_nonlinear256 \\
    --agent.probe_on_policy_freq 10000 \\
    --agent.probe_off_policy_freq 10000 \\
    --agent.random_probe \\
    --agent.rollout_freq 10000 \\
    --agent.num_timesteps 100000 \\
    --agent.eval_freq -1 \\
    --agent.save_freq 10000 \\
    --agent.optimize_per_env_step 2 \\
    --env.repeat_action_probability {REPEAT_ACTION_PROBABILITY} \\
    --eval_env.repeat_action_probability {REPEAT_ACTION_PROBABILITY} \\
    --data.runs 1 2 3 4 5 \\
    --data.checkpoints 49 \\
    --data.dataset_name seaquest_expert \\
    --data.eval_ratio 0.99 \\
    --probe.hidden_sizes 256"""

    # Conditionally append the load_model flags
    if pretrain_type not in ("baseline", "nature-e2e"):
        cmd += f" \\\n    --load_model.enable \\"
        cmd += f"\n    --load_model.model_path {path} \\"
        cmd += f"\n    --load_model.freeze_layers backbone"
        
    if SPR_BASELINE_COMPAT:
        cmd += " \\\n    --agent.spr_baseline_compat"

    cmd += "\n" # Add a final newline to be safe
    return cmd


if __name__ == "__main__":
    os.makedirs("job_scripts", exist_ok=True)
    
    # Create the cluster logs directory and get its absolute path
    os.makedirs("job_logs", exist_ok=True)
    log_dir = os.path.abspath("job_logs")
    
    for pretrain_type, path in PRETRAINED_PATHS.items():
        for seed in SEEDS:
            for game in GAMES:
                cmd = make_cmd(path, seed, game, pretrain_type, log_dir)
                
                # Save the script to disk for debugging/record-keeping
                script_name = f"job_scripts/submit_{game}_{pretrain_type}_{seed}.sh"
                with open(script_name, "w") as f:
                    f.write(cmd)
                
                # Submit to SGE queue by piping the string to qsub
                print(f"Submitting {pretrain_type} | seed {seed} | game {game}...")
                process = subprocess.run(
                    ["qsub"], 
                    input=cmd, 
                    text=True, 
                    capture_output=True
                )
                
                if process.returncode == 0:
                    print(f"  Success: {process.stdout.strip()}")
                else:
                    print(f"  Error: {process.stderr.strip()}")