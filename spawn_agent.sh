#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time 0-2:00
#SBATCH --partition gpu,seas_gpu
#SBATCH --mem 30G
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output bash-outputs/sweep_%j.out
#SBATCH --account davies_lab

SWEEP_ID=${1}
source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"
wandb agent --count 1 $SWEEP_ID
