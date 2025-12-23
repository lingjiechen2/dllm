#!/bin/bash
#SBATCH --job-name=length-bias
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

SPLIT_IDX=$1
NUM_SPLITS=$2
DATASET=$3
DATASET_CONFIG=$4
DATASET_SPLIT=$5
STEPS=$6
MAX_NEW_TOKENS=$7
MODEL_TAG=$8
MODEL=$9
NUM_EXTRACT_STEPS=${10:-4}

echo "Running split $SPLIT_IDX / $NUM_SPLITS"

python "${SCRIPT_DIR}/extract_probe_data.py" \
    --dataset_name "$DATASET" \
    --dataset_config "$DATASET_CONFIG" \
    --dataset_split "$DATASET_SPLIT" \
    --steps "$STEPS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --split_idx "$SPLIT_IDX" \
    --num_splits "$NUM_SPLITS" \
    --model_tag "$MODEL_TAG" \
    --model_name_or_path "$MODEL" \
    --num_extract_steps "$NUM_EXTRACT_STEPS"
