#!/bin/bash -l

# Set SCC project
#$ -P ds598xz

# Name the job in the queue
#$ -N spr_seed0

# Execute the job from the current working directory
#$ -cwd

# Specify where to save the .o and .e log files
#$ -o job_logs/
#$ -e job_logs/

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
    --agent.pretrain_ckpt 100 \
    --probe.type offline \
    --probe.policy_ckpt_dir data_storage/policies/seaquest/spr_seed0_nonlinear256_ckpt100 \
    --data.dataset_name seaquest_expert \
    --data.runs 1 2 3 4 5 \
    --data.checkpoints 49 \
    --data.eval_ratio 0.25 \
    --n_step 10 \
    --load_model.enable \
    --load_model.model_path ./data_storage/pretrained_models/pretrain_spr_seaquest/spr/spr_resnet_seaquest_None/epoch25.pth \
    --load_model.freeze_layers backbone \
    --probe.patience 500 \
    --wandb.enabled \
    --wandb.project diagnosing_train_test_difference \
    --wandb.name spr_seed0_nonlinear256_ckpt100_off_off \
    --eval_env.repeat_action_probability 0.0 \
    --probe.hidden_sizes 256 \
    # --probe.analyze_only \
