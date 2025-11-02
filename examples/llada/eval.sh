#!/usr/bin/env bash
# ===== Input Arguments =====
model_path=$1      # e.g., GSAI-ML/LLaDA-8B-Base or GSAI-ML/LLaDA-8B-Instruct
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
    common_args="--model llada --seed 1234 --device cuda --batch_size 1 --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model llada --seed 1234 --device cuda --batch_size 1"
fi


# =======================
# Generation Tasks
# =======================

if [ "$use_instruct" = "True" ]; then
    # Instruct Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks mbpp_llada_instruct --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

else
    # Base Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks gsm8k --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"
fi


# =======================
# Likelihood Tasks
# =======================

if [ "$use_instruct" = "True" ]; then
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks mmlu_generative --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks mmlu_pro --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=5,steps=5,block_length=5,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks gpqa_n_shot_gen --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=32,steps=32,block_length=32,cfg=0.0"

else
    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks truthfulqa_mc2 --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=2.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks cmmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/eval/eval_llada.py \
        --tasks ceval-valid --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"
fi
