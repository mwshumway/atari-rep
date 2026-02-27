#!/bin/bash -l

# Set SCC project
#$ -P ds598xz

# Request 8 cores
#$ -pe omp 8

# Request gpu(s)
#$ -l gpus=1

# GPU type
#$ -l gpu_type=L40S

# Runtime
#$ -l h_rt=12:00:00

module load miniconda
conda activate atari-rep-bench
module load cuda/12.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8 

python run_online_rl.py \
    --games pong \
    --seed 0 \
    --wandb.enabled \
    
    