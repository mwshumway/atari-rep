#!/bin/bash -l

# Set SCC project
#$ -P replearn

# Request 8 cores
#$ -pe omp 8

# Request 3 gpus
#$ -l gpus=3

# Minimum compute capability
#$ -l gpu_type=L40S

# Runtime
#$ -l h_rt=72:00:00

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8 

python run_pretrain.py \
    --num_gpus_per_node 3 \
    --data.distributed \
    --pretrain.distributed \
    --wandb.enabled \
    --wandb.project 'pretrain_atc_seaquest' \
    --wandb.name 'atc_resnet_seaquest' \
    --wandb.group 'atc' \
    --head.type 'mh_linear' \
    --head.action_size 512 \
    --neck.type 'mh_mlp' \
    --neck.hidden_dims 1024 512 \
    --pretrain.type 'atc' \
    --optimizer.type 'adamw' \
    --optimizer.lr 0.0003 \
    --lr_scheduler.max_lr 0.0003 \
    --optimizer.weight_decay 0.00001 \
    --optimizer.betas 0.9 0.999 \
    --optimizer.eps 0.00000001 \
    --games "seaquest" \
    --data.dataset_name 'seaquest_5runs_10ckpts_10ksamples' \
    --data.runs 1 2 3 4 5 \
    --pretrain.target_update_every -1 \