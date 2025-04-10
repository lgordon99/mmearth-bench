#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time 1-00:00
#SBATCH --partition davies_gpu,gpu,seas_gpu
#SBATCH --mem 60G
#SBATCH --gres gpu:1
#SBATCH --output bash-outputs/sweep_%j.out
#SBATCH --account davies_lab

SWEEP_ID=${1}
source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"
wandb agent --count 1 $SWEEP_ID
