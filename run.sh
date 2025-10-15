task=${1}
architecture=${2}
adaptation_mode=${3}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")

if [[ "$adaptation_mode" == "standard" || "$adaptation_mode" == "multimodal" || "$adaptation_mode" == *ttt* || "$adaptation_mode" == *-10* ]]; then
    DAYS="1"
elif [[ "$adaptation_mode" == *mt3* || "$adaptation_mode" == *sln* ]]; then
    DAYS="7"
else
    DAYS="2"
fi

if [ "$task" == "biomass" ]; then
    MEM="90G"
elif [ "$task" == "species" ]; then
    MEM="130G"
else
    MEM="30G"
fi

bash_file="${data_dir_path}/experiments/${task}_${architecture}_${adaptation_mode}_run.sh"

cat > $bash_file <<EOF
#!/bin/bash
#SBATCH --job-name ${task}_${architecture}_${adaptation_mode}
#SBATCH --time $DAYS-00:00
#SBATCH --partition seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_h200:1
#SBATCH --output ${data_dir_path}/experiments/output-files/${task}/${task}_${architecture}_${adaptation_mode}.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate $data_dir_path/mmearth-bench-env
echo "Task: ${task}"
echo "Architecture: ${architecture}"
echo "Adaptation mode: ${adaptation_mode}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
python train.py +task=${task} +architecture=${architecture} +adaptation_mode=${adaptation_mode}
EOF

sbatch $bash_file
rm $bash_file
