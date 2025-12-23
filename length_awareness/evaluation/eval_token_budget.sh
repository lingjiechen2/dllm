
export PYTHONPATH=.:$PYTHONPATH
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/LLaDA-8B-SFT/multi_granularity_gsm8k_1"
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


num_gpu=4
common_args="--model llada --apply_chat_template"
model_name_or_path="checkpoint-final"
task_name="gsm8k_cot"
length_list=(32 64 128 256 512 1024)
seed=42
limit=80

mkdir -p "multi_granularity/${task_name}"

for length in "${length_list[@]}"; do
  echo "Checking for existing output for length=${length}"
  base_dir="multi_granularity/${task_name}/$(basename "$model_name_or_path")"
  mkdir -p "$base_dir"


  base_name="${task_name}_${length}_seed${seed}_samples"
  prefix="${base_dir}/${base_name}"
  existing_file=$(find "$base_dir" -type f -name "${base_name}_*.json" | head -n 1)

  if [[ -n "$existing_file" ]]; then
    echo "✅ Skipping: Output already exists -> $existing_file"
    continue
  fi

  echo "🚀 Submitting job for length=${length}"

  sbatch \
    --job-name=eval-${length} \
    --partition=mllm_safety \
    --quotatype=spot \
    --gres=gpu:${num_gpu} \
    --ntasks-per-node=${num_gpu} \
    --cpus-per-task=${num_gpu} \
    --time=04:00:00 \
    --output="/mnt/petrelfs/fanyuyu/fyy/dllm/multi_granularity/${task_name}/%x-%j.out" \
    --error="/mnt/petrelfs/fanyuyu/fyy/dllm/multi_granularity/${task_name}/%x-%j.err" \
    --requeue \
    eval_dllm.sh llada "${task_name}" "${BASE_MODELS_DIR}/${model_name_or_path}" True 1 False "${limit}" \
      --max_new_tokens "${length}" \
      --steps "${length}" \
      --block_length "${length}" \
      --seed "${seed}" \
      --output_path "${prefix}.json"
  sleep 0.5
  # break
done



# sbatch \
#   --job-name=model-eval \
#   --partition=mllm_safety \
#   --quotatype=spot \
#   --gres=gpu:${num_gpu} \
#   --ntasks-per-node=${num_gpu} \
#   --cpus-per-task=${num_gpu} \
#   --time=04:00:00 \
#   --output=/mnt/petrelfs/fanyuyu/fyy/dllm/length_awareness/%x-%j.out \
#   --error=/mnt/petrelfs/fanyuyu/fyy/dllm/length_awareness/%x-%j.err \
#   --requeue \
#   eval.sh llada gsm8k_cot ${BASE_MODELS_DIR}/${model_name_or_path} True 1 False 20 \
#     --max_new_tokens 256 \
#     --steps 256 \
#     --block_length 256 \
#     --seed 42

# srun  -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=04:00:00 \
# accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
#     --tasks "${task_name}" --num_fewshot 8 ${common_args} \
#     --model_args "pretrained=${BASE_MODELS_DIR}/${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0,confidence_eos_eot_inf=False" \
#     --output_path "length_awareness/$(basename "$model_name_or_path")_${task_name}_samples.json" \
#     --log_samples \
#     --limit 20




# model_name_or_path="/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
# task_name="gsm8k_cot_llama"
# # length_list=(32 64 128 256 512)
# length_list=(1024)
# seed=42
# limit=16

# mkdir -p "length_awareness/${task_name}/$(basename "$model_name_or_path")"

# for length in "${length_list[@]}"; do
#   echo "Running max_new_tokens=${length}"

#   srun  -p mllm_safety --quotatype=spot --gres=gpu:4 --time=04:00:00 \
#     accelerate launch --multi_gpu --num_processes 4 -m lm_eval \
#           --model hf \
#           --model_args pretrained="${model_name_or_path}" \
#           --gen_kwargs max_new_tokens=${length} \
#           --tasks ${task_name} \
#           --batch_size 4 \
#           --limit ${limit} \
#           --apply_chat_template \
#           --output_path "length_awareness/${task_name}/$(basename "$model_name_or_path")/${task_name}_${length}_seed${seed}_samples.json" \
#           --log_samples&
#   sleep 1
#   break
# done




# srun  -p mllm_safety --quotatype=spot --gres=gpu:1 --time=04:00:00 lm_eval \
#   --model hf \
#   --model_args pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Qwen/Qwen1.5-0.5B-Chat" \
#   --tasks lambada \
#   --batch_size 4 \
#   --apply_chat_template