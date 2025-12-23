#!/usr/bin/env bash

# ===== Environment =====
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ===== Arguments =====
model_name_or_path="/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/"
num_gpu=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== BASE mode =====
common_args="--model llada"

# ===== main_process_port control =====
BASE_PORT=26700
PORT_OFFSET=0

next_port () {
  local port=$((BASE_PORT + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
  echo "${port}"
}

# ===== task -> limit =====
get_limit () {
  case "$1" in
    gsm8k|humaneval|mbpp)
      echo 100 ;;
    bbh|minerva_math)
      echo 20 ;;
    *)
      echo 0 ;;
  esac
}

# ===== Generative tasks =====
tasks=(
  "gsm8k 5"
  "bbh 3"
  "minerva_math 4"
  "humaneval 0"
  "mbpp 3"
)

run_job () {
  local task=$1
  local fewshot=$2
  local hp_tag=$3
  local model_args=$4

  local limit
  limit=$(get_limit "${task}")

  local out_dir="tmp/${task}_${limit}/${hp_tag}"

  if [[ -d "${out_dir}" ]] && [[ -n "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
    echo "[SKIP] ${out_dir} exists and is non-empty"
    return
  fi

  mkdir -p "${out_dir}"
  local port
  port=$(next_port)

  accelerate launch \
    --num_processes "${num_gpu}" \
    --main_process_port "${port}" \
    dllm/pipelines/llada/eval.py \
    --tasks "${task}" \
    --num_fewshot "${fewshot}" \
    --output_path "${out_dir}" \
    ${common_args} \
    --model_args "${model_args}" \
    --confirm_run_unsafe_code \
    --limit "${limit}"

  sleep 1
}

# ==================================================
# 1. Baseline
# ==================================================
hp_tag="baseline"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024"
done

# ==================================================
# 2. len1024_step512
# ==================================================
hp_tag="len1024_step512"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=1024,steps=512,block_size=1024"
done

# ==================================================
# 3. len512_step512
# ==================================================
hp_tag="len512_step512"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=1024"
done

# ==================================================
# 4. block32
# ==================================================
hp_tag="block32"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=32"
done

# ==================================================
# 5. temp0.5
# ==================================================
hp_tag="temp0.5"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,temperature=0.5"
done

# ==================================================
# 6. remask_random
# ==================================================
hp_tag="remask_random"
for t in "${tasks[@]}"; do
  set -- $t
  run_job "$1" "$2" "${hp_tag}" \
    "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,remasking=random"
done
