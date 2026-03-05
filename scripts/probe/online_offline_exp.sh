#!/bin/bash -l

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

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8 
    
python run_probe.py \
    --games seaquest \
    --seed 0 \
    --pretrain.type spr \
    --agent.pretrain_ckpt 25 \
    --probe.type offline \
    --probe.policy_ckpt_dir data_storage/policies/seaquest/spr_ckpt25_seed0 \
    --data.dataset_name seaquest_expert_10ksamples_nstep10 \
    --data.runs 1 2 3 4 5 \
    --data.checkpoints 49 \
    --data.eval_ratio 0.25 \
    --n_step 10 \
    --load_model.enable \
    --load_model.model_path ./data_storage/pretrained_models/pretrain_spr_seaquest/spr/spr_resnet_seaquest_None/epoch25.pth \
    --load_model.freeze_layers backbone \
    --probe.patience 500 \
