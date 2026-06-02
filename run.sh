task=${1}
architecture=${2}
adaptation_mode=${3}
train_percent=${4}
seed=${5}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")

if [[ "$adaptation_mode" == *TTT* ]]; then
    TIME="0-6:00"
elif [[ "$adaptation_mode" == "FT" || "$adaptation_mode" == "JT" || "$adaptation_mode" == "LP" ]]; then
    TIME="2-00:00"
fi

if [[ "$task" == "species" ]]; then
    MEM="140G"
elif [[ "$task" == "biomass" ]]; then
    MEM="130G"
elif [[ "$task" == "soil_organic_carbon" || "$task" == "soil_pH" ]]; then
    MEM="40G"
else
    MEM="30G"
fi

if [[ "$architecture" == "AnySat" ]]; then
    NUM_GPUS=2
    MEM="$((${MEM%[A-Za-z]} * NUM_GPUS))${MEM##*[0-9]}"
else
    NUM_GPUS=1
fi

bash_file="${data_dir_path}/experiments/${task}_${architecture}_${adaptation_mode}_${train_percent}_${seed}_run.sh"

cat > $bash_file <<EOF
#!/bin/bash
#SBATCH --job-name ${task}_${architecture}_${adaptation_mode}_${train_percent}_${seed}
#SBATCH --time $TIME
#SBATCH --partition seas_gpu
#SBATCH --mem $MEM
#SBATCH --ntasks-per-node $NUM_GPUS
#SBATCH --gres gpu:nvidia_h200:$NUM_GPUS
#SBATCH --output ${data_dir_path}/experiments/output-files/${task}/${task}_${architecture}_${adaptation_mode}_${train_percent}_${seed}.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate $data_dir_path/mmearth-bench-env
echo "Task: ${task}"
echo "Architecture: ${architecture}"
echo "Adaptation mode: ${adaptation_mode}"
echo "Train percent: ${train_percent}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
export WANDB_CACHE_DIR=/tmp/${task}_${architecture}_${adaptation_mode}_${train_percent}_${seed}/wandb_cache
srun python train.py +task=${task} +architecture=${architecture} +adaptation_mode=${adaptation_mode} +train_percent=${train_percent} seed=${seed}
EOF

sbatch $bash_file
rm $bash_file
