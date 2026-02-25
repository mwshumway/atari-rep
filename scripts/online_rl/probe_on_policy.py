import os
import subprocess

PRETRAINED_PATHS = {
    "cql": "./data_storage/pretrained_models/pretrain_cql/cql/cql-dist_resnet/epoch90.pth",
    "atc": "./data_storage/pretrained_models/pretrain_atc/atc/atc_resnet/epoch50.pth",
    "spr": "./data_storage/pretrained_models/pretrain_spr/spr/spr_resnet/epoch25.pth",
    "baseline": ""
}

SEEDS = [0, 1]
GAMES = ["seaquest"]

NONLINEAR_PROBE_HIDDEN_SIZES = ()

def make_cmd(path, seed, game, pretrain_type, log_dir):
    cmd = f"""#!/bin/bash -l

# Set SCC project
#$ -P replearn

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
#$ -l gpu_c=7.0

# Runtime
#$ -l h_rt=12:00:00

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUDA_LAUNCH_BLOCKING=1
export TORCH_COMPILE_DISABLE=1
    
python run_online_rl.py \\
    --games {game} \\
    --seed {seed} \\
    --wandb.enabled \\
    --wandb.project {game}_probe_on_policy \\
    --wandb.group {pretrain_type} \\
    --wandb.name {pretrain_type}_seed{seed} \\
    --agent.probe_on_policy_freq 10000 \\
    --agent.rollout_freq 20000 \\
    --agent.eval_freq -1 \\
    --agent.save_freq -1"""

    # Conditionally append the load_model flags
    if pretrain_type != "baseline":
        cmd += f" \\\n    --load_model.enable \\"
        cmd += f"\n    --load_model.model_path {path} \\"
        cmd += f"\n    --load_model.freeze_layers backbone"
        
    cmd += f"\n    --probe.hidden_sizes {' '.join(map(str, NONLINEAR_PROBE_HIDDEN_SIZES))}"

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