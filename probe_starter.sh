#!/bin/bash


STEPS=${1:-128}
MAX_NEW_TOKENS=${2:-128}

MAX_JOBS=32
NUM_SPLITS=128
MODEL="GSAI-ML/LLaDA-8B-Instruct"
DATASET="openai/gsm8k"
DATASET_CONFIG="main"
DATASET_SPLIT="test"
MODEL_TAG="llada_instruct"

job_count=0

for ((i=0; i<NUM_SPLITS; i++)); do

    if (( i >= MAX_JOBS )); then
        echo "Reached MAX_JOBS=$MAX_JOBS → stop submitting."
        break
    fi

    # Check any layer file exists
    layer1_file="/mnt/lustrenew/mllm_safety-shared/fanyuyu/probe_data/${MODEL_TAG}/$(basename ${DATASET})/layer01/L${MAX_NEW_TOKENS}_step${STEPS}_split$(printf "%02d" $i).pt"

    if [[ -f "$layer1_file" ]]; then
        echo "[Skip] split $i: already processed."
        continue
    fi

    echo "[Submit] split $i / $NUM_SPLITS → submitting sbatch job"

    sbatch probe_job.sh \
        "$i" \
        "$NUM_SPLITS" \
        "$DATASET" \
        "$DATASET_CONFIG" \
        "$DATASET_SPLIT" \
        "$STEPS" \
        "$MAX_NEW_TOKENS" \
        "$MODEL_TAG" \
        "$MODEL"

    job_count=$((job_count + 1))
    sleep 0.2
done

echo "=== All splits processed or submitted ==="
