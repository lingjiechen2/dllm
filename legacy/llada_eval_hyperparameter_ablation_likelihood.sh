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

# ===== BASE mode only =====
common_args="--model llada"

# ===== Tasks (log-likelihood only) =====
tasks=(
  "gpqa_main_n_shot 5"
  "truthfulqa_mc2 0"
  "arc_challenge 0"
  "hellaswag 0"
  "winogrande 5"
  "piqa 0"
  "mmlu 5"
  "cmmlu 5"
  "ceval-valid 5"
)

# =========================
# 1. Baseline
# =========================
for t in "${tasks[@]}"; do
  set -- $t
  task=$1
  fewshot=$2

  accelerate launch \
    --num_processes "${num_gpu}" \
    dllm/pipelines/llada/eval.py \
    --tasks "${task}" \
    --num_fewshot "${fewshot}" \
    ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.5"
done

# =========================
# 2. CFG ablation (cfg = 0.1)
# =========================
# for t in "${tasks[@]}"; do
#   set -- $t
#   task=$1
#   fewshot=$2
#
#   accelerate launch \
#     --num_processes "${num_gpu}" \
#     dllm/pipelines/llada/eval.py \
#     --tasks "${task}" \
#     --num_fewshot "${fewshot}" \
#     ${common_args} \
#     --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.1"
# done

# =========================
# 3. MC_NUM ablation (mc_num = 32)
# =========================
# for t in "${tasks[@]}"; do
#   set -- $t
#   task=$1
#   fewshot=$2
#
#   accelerate launch \
#     --num_processes "${num_gpu}" \
#     dllm/pipelines/llada/eval.py \
#     --tasks "${task}" \
#     --num_fewshot "${fewshot}" \
#     ${common_args} \
#     --model_args "pretrained=${model_name_or_path},mc_num=32,cfg=0.5"
# done
