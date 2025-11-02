#!/usr/bin/env bash
# ===== Input Arguments =====
model_path=$1      # e.g., Dream-org/Dream-v0-Base-7B or Dream-org/Dream-v0-Instruct-7B
use_instruct=$2    # True or False

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

# ===== Conditional Configurations =====
if [ "$use_instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    common_args="--model dream --seed 42 --device cuda --batch_size 1 --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model dream --seed 42 --device cuda --batch_size 1"
fi


# =======================
# Generation / Instruct Tasks
# =======================

if [ "$use_instruct" = "True" ]; then
    # Instruct Tasks
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks mmlu_generative --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks mmlu_pro --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks gsm8k_cot --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=256,max_length=256,steps=256,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks minerva_math --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.0,top_p=1.0,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=768,max_length=768,steps=768,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks mbpp_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=1024,max_length=1024,steps=1024,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks ifeval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=1280,max_length=1280,steps=1280,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

else
    # Base Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=256,max_length=256,steps=256,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"
fi


# =======================
# Likelihood Tasks (Base Only)
# =======================

if [ "$use_instruct" != "True" ]; then
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks arc_easy --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_dream.py \
        --tasks race --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"
fi
