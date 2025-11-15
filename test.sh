srun -p mllm_safety --quotatype=spot --gres=gpu:1 python \
    extract_probe_data.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_name "openai/gsm8k" \
    --dataset_config "main" \
    --dataset_split "test" \
    --model_tag "llada_instruct" \
    --steps 128 \
    --max_new_tokens 128 \
    --num_splits 1024 \
    --split_idx 0