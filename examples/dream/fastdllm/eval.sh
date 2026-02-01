#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Input Arguments =====
model_name_or_path="Dream-org/Dream-v0-Instruct-7B"
instruct=True
num_gpu=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Conditional Configurations =====
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    common_args="--model dream --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model dream"
fi

# =======================
# GSM8K Task Evaluation
# =======================

# Baseline （25.14s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=False,escape_until=False"

# Prefix cache (7.49s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=False,escape_until=False"

# Parallel (9.98s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=256,steps=8,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

# Prefix cache + Parallel (1.97s)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

# Dual cache + Parallel (1.97s)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"


# ===========================
# Humaneval Task Evaluation
# ===========================

# Baseline
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=768,steps=768,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# Prefix cache (6.64s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# Parallel (3.17s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=256,steps=8,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel (1.65s)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# Dual cache + Parallel (1.49s)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/fastdllm/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=256,steps=256,block_size=32,temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code
