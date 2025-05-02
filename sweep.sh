task=${1}
split_type=${2}
model_type=${3}
adaptation_mode=${4}

max_lr=(1e-2 1e-1 1e-0)
weight_decay=(0 1e-3 1e-2 1e-1)

num_runs=$(( ${#max_lr[@]} * ${#weight_decay[@]} ))

function format_array_to_string() {
    local -n array=$1
    local result="["

    for i in "${!array[@]}"; do
        if [ "$i" -eq 0 ]; then
            result+="${array[$i]}"
        else
            result+=", ${array[$i]}"
        fi
    done

    result+="]"
    echo "$result"
}

max_lr_string=$(format_array_to_string max_lr)
weight_decay_string=$(format_array_to_string weight_decay)

touch sweep_2.yaml
cat >sweep_2.yaml <<EOF
project: mmearth-bench
entity: luciagordon-harvard-university
name: ${task}_${split_type}_${model_type}_${adaptation_mode}
program: train.py
method: grid
metric:
  name: "Val RMSE"
  goal: minimize
parameters:
  model.max_lr:
    values: ${max_lr_string}
  model.weight_decay:
    values: ${weight_decay_string}
command:
  - python
  - train.py
  - +task=${task}
  - +split_type=$([ "$split_type" == "r" ] && echo "random" || echo "geographic")
  - +model_type=${model_type}
  - +adaptation_mode=${adaptation_mode}
EOF

wandb sweep sweep_2.yaml &> sweep_output.txt
rm sweep_2.yaml
sweep_id=$(cat sweep_output.txt | grep "agent" | tail -1 | awk '{print $NF}')
rm sweep_output.txt

echo "Sweep ID: $sweep_id"
echo "Number of runs: $num_runs"

if [ "$task" == "biomass" ]; then
    MEM="60G"
elif [ "$task" == "species" ]; then
    MEM="30G"
else
    MEM="20G"
fi

touch spawn_agent.sh

cat > spawn_agent.sh <<EOF
#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time 0-2:00
#SBATCH --partition gpu,seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output bash-outputs/sweep_%j.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
wandb agent --count 1 $sweep_id
EOF

for _ in $(seq 1 $num_runs); do
    sbatch spawn_agent.sh
done

rm spawn_agent.sh
