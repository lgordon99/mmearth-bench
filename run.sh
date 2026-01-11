task=${1}
architecture=${2}
adaptation_mode=${3}
train_percent=${4}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")

if [[ "$adaptation_mode" == *TTT* ]]; then
    TIME="0-6:00"
elif [[ "$architecture" == *AnySat* ]]; then
    TIME="4-00:00"
elif [[ "$adaptation_mode" == "FT" || "$adaptation_mode" == "TMD" || "$adaptation_mode" == "JT" || "$adaptation_mode" == "JT_weighted_gradients" || "$adaptation_mode" == "MT3_metabatch" ]]; then
    TIME="1-00:00"
elif [[ "$adaptation_mode" == *mt3* || "$adaptation_mode" == *sln* ]]; then
    TIME="7-00:00"
else
    TIME="2-00:00"
fi

if [[ "$task" == "species" ]]; then
# if [[ "$architecture" == "AnySat" || "$task" == "species" ]]; then
    MEM="140G"
elif [[ "$task" == "biomass" ]]; then
    MEM="130G"
elif [[ "$task" == "soil_organic_carbon" || "$task" == "soil_pH" ]]; then
    MEM="40G"
else
    MEM="30G"
fi

bash_file="${data_dir_path}/experiments/${task}_${architecture}_${adaptation_mode}_${train_percent}_run.sh"

cat > $bash_file <<EOF
#!/bin/bash
#SBATCH --job-name ${task}_${architecture}_${adaptation_mode}_${train_percent}
#SBATCH --time $TIME
#SBATCH --partition seas_gpu,gpu_h200
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_h200:1
#SBATCH --output ${data_dir_path}/experiments/output-files/${task}/${task}_${architecture}_${adaptation_mode}_${train_percent}.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate $data_dir_path/mmearth-bench-env
echo "Task: ${task}"
echo "Architecture: ${architecture}"
echo "Adaptation mode: ${adaptation_mode}"
echo "Train percent: ${train_percent}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
export WANDB_CACHE_DIR=/tmp/${task}_${architecture}_${adaptation_mode}_${train_percent}/wandb_cache
python train.py +task=${task} +architecture=${architecture} +adaptation_mode=${adaptation_mode} +train_percent=${train_percent}
EOF

sbatch $bash_file
rm $bash_file
