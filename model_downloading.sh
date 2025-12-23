#!/usr/bin/env bash
set -e

MODEL_DIR_BASE="/home/lingjie7/models/huggingface"
DATASET_DIR_BASE="/home/lingjie7/datasets/huggingface"

download_model() {
    local model_name="$1"
    local target_dir="${MODEL_DIR_BASE}/${model_name}"
    echo "🚀 Downloading model: ${model_name}"
    echo "📦 Target path: ${target_dir}"
    mkdir -p "${target_dir}"
    hf download "${model_name}" --local-dir "${target_dir}"
}

download_dataset() {
    local dataset_name="$1"
    local target_dir="${DATASET_DIR_BASE}/${dataset_name}"
    echo "📚 Downloading dataset: ${dataset_name}"
    echo "📦 Target path: ${πtarget_dir}"
    mkdir -p "${target_dir}"
    hf download "${dataset_name}" --repo-type dataset --local-dir "${target_dir}"
}

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  bash model_downloading.sh model <model_name>"
    echo "  bash model_downloading.sh dataset <dataset_name>"
    exit 1
fi

TYPE="$1"
NAME="$2"

if [ "$TYPE" = "model" ]; then
    download_model "$NAME"
elif [ "$TYPE" = "dataset" ]; then
    download_dataset "$NAME"
else
    echo "❌ Unknown type: $TYPE (use 'model' or 'dataset')"
    exit 1
fi


