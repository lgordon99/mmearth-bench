task=${1}
model_type=${2}
adaptation_mode=${3}

max_lr=($(jq -r ".[\"${adaptation_mode}\"].max_lr | .[]" sweep_hyperparameters.json))
echo "max_lr: ${max_lr[@]}"

weight_decay=($(jq -r ".[\"${adaptation_mode}\"].weight_decay | .[]" sweep_hyperparameters.json))
echo "weight_decay: ${weight_decay[@]}"

if [ "$adaptation_mode" == "llrd" ]; then
  decay_factor=($(jq -r ".[\"${adaptation_mode}\"].decay_factor | .[]" sweep_hyperparameters.json))
  echo "decay_factor: ${decay_factor[@]}"
fi

if [ "$adaptation_mode" == "llrd" ]; then
  num_runs=$(( ${#max_lr[@]} * ${#weight_decay[@]} * ${#decay_factor[@]} ))
else
  num_runs=$(( ${#max_lr[@]} * ${#weight_decay[@]} ))
fi

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

if [ "$adaptation_mode" == "llrd" ]; then
    decay_factor_string=$(format_array_to_string decay_factor)
fi

sweep_log_file="sweep_log.csv"

if [ ! -f "$sweep_log_file" ]; then
    echo "date,name,sweep_ID" > "$sweep_log_file"
fi

sweep_name="${task}_${model_type}_${adaptation_mode}"

touch sweep_2.yaml

yaml_content="project: mmearth-bench
entity: luciagordon-harvard-university
name: ${sweep_name}
program: train.py
method: grid
metric:
  name: \"Val RMSE\"
  goal: minimize
parameters:
  model.max_lr:
    values: ${max_lr_string}
  model.weight_decay:
    values: ${weight_decay_string}"

if [ "$adaptation_mode" == "llrd" ]; then
  yaml_content="${yaml_content}
  model.decay_factor:
    values: ${decay_factor_string}"
fi

yaml_content="${yaml_content}
command:
  - python
  - train.py
  - +task=${task}
  - +model_type=${model_type}
  - +adaptation_mode=${adaptation_mode}"

echo "$yaml_content" > sweep_2.yaml

wandb sweep sweep_2.yaml &> sweep_output.txt
rm sweep_2.yaml
sweep_id=$(cat sweep_output.txt | grep "agent" | tail -1 | awk '{print $NF}')
rm sweep_output.txt

echo "Sweep ID: $sweep_id"
echo "Number of runs: $num_runs"

if [ -n "$sweep_id" ]; then
    current_date=$(date "+%Y-%m-%d %H:%M:%S")
    echo "$current_date,$sweep_name,$sweep_id" >> "$sweep_log_file"
fi

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
#SBATCH --time 0-3:00
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
