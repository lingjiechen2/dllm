export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/home/lingjie7/datasets/huggingface"
export BASE_MODELS_DIR="/home/lingjie7/models/huggingface/"
export HF_DATASETS_CACHE="/home/lingjie7/datasets/huggingface"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true  # For CMMLU dataset

export CUDA_VISIBLE_DEVICES=4,5,6,7
main_port=29515
pretrained="/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16"
common_args="--model llada_dist --seed 1234 --apply_chat_template --limit 200" #  --limit None
base_path="dllm/eval/eval_llada.py"
num_gpu=4

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks winogrande \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128" \
#     --device cuda \
#     --num_fewshot 5

PYTHONBREAKPOINT=0 \
accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    ${base_path} \
    --tasks gsm8k_cot \
    --batch_size 1 \
    ${common_args} \
    --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
    --device cuda