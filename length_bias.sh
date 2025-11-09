
export PYTHONPATH=.:$PYTHONPATH
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface"
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/shared/datasets/huggingface"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"


export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 # For cmmlu dataset
export MASTER_ADDR MASTER_PORT WORLD_SIZE


num_gpu=4
common_args="--model llada --apply_chat_template"
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"

srun  -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=04:00:00 \
accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0" \
    --output_path 


# srun  -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=04:00:00 \
# accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
#     --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
#     --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"