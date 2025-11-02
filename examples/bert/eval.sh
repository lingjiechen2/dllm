#!/usr/bin/env bash
# ===== Input Arguments =====
model_path=$1      # e.g., ModernBERT-base/checkpoint-final

# =====  Environmental Variables =====
export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

# ===== Basic Settings =====
num_gpu=4

# ===== Common arguments =====
common_args="--model bert --seed 1234 --device cuda --batch_size 1 --apply_chat_template"

# =======================
# BERT Instruct (Chat) Tasks
# =======================

accelerate launch --num_processes ${num_gpu} dllm/eval/eval_bert.py \
    --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/eval/eval_bert.py \
    --tasks mmlu_generative --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/eval/eval_bert.py \
    --tasks mmlu_pro --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256"

accelerate launch --num_processes ${num_gpu} dllm/eval/eval_bert.py \
    --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/eval/eval_bert.py \
    --tasks winogrande --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"
