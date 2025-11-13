#!/bin/bash
#SBATCH --job-name=length-bias
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue


export PYTHONPATH=.:$PYTHONPATH
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/tmp"
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/shared/datasets/huggingface"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"

# export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 # For cmmlu dataset
export MASTER_ADDR MASTER_PORT WORLD_SIZE


# ===== User variables =====
model_name_or_path="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Qwen/Qwen1.5-0.5B"
task_name="ceval-valid"
length=512
seed=1231

export NCCL_DEBUG=warn

echo "Running job:"
echo "Model:         $model_name_or_path"
echo "Task:          $task_name"
echo "Length:        $length"
echo "Seed:          $seed"
echo

# ===== Command =====
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 29512 \
    -m lm_eval \
        --model hf \
        --model_args pretrained="${model_name_or_path}" \
        --gen_kwargs max_new_tokens=${length} \
        --tasks ${task_name} \
        --batch_size 16 \
        --output_path "logs/${task_name}_${length}_seed${seed}_samples.json" 
