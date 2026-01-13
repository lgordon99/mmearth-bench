task=${1}
architecture=${2}
adaptation_mode=${3}
lr=${4}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")

TIME="0-0:20"

if [[ "$task" == "species" ]]; then
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
#SBATCH --partition seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_h200:1
#SBATCH --output ${data_dir_path}/experiments/output-files/${task}/${task}_${architecture}_${adaptation_mode}_${train_percent}.out
#SBATCH --account tambe_lab

source ~/.bashrc
conda activate $data_dir_path/mmearth-bench-env
echo "Task: ${task}"
echo "Architecture: ${architecture}"
echo "Adaptation mode: ${adaptation_mode}"
echo "Train percent: ${train_percent}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
export WANDB_CACHE_DIR=/tmp/${task}_${architecture}_${adaptation_mode}_${train_percent}/wandb_cache
python train.py +task=${task} +architecture=${architecture} +adaptation_mode=${adaptation_mode} +train_percent=100 model.inner_loop_lr=${lr} logger.tags=[n1,b8,lr${lr}]
EOF

sbatch $bash_file
rm $bash_file
