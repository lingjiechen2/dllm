# BERT Chat

<!-- > 🤗 **Checkpoints**  
> ModernBERT-large-chat-v1, ModernBERT-base-chat-v1. -->
### 🤗 BERT Chat Checkpoints
* `ModernBERT-large-chat-v1`
* `ModernBERT-base-chat-v1`

This directory provides the exact training / inference / evaluation scripts for the two ModernBERT models finetuned for instruction following.

See [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) **Report**](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Generative-BERT--VmlldzoxNDg0MzExNg) for experimental results, lessons learned and more reproduction scripts.

<details>
<summary>🎬 Click to show Chat Demo</summary>

<p align="center" style="margin-top: 15px;">
    <img src="/examples/bert/assets/chat.gif" alt="chat" width="70%">
</p>
<p align="center">
  <em>
    Chat with <a href="[TODO]"><code>ModernBERT-large-chat-v1</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
  </em>
</p>
</details>

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>

## Files overview
```
# example entry points for training / inference
examples/bert
├── chat.py                         # Interactive inference example
├── generate.py                     # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```

## Training
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024"

accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-base" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024"
```

## Inference


## Evaluation
> [!IMPORTANT]  
> If you find missing files inside the `lm-evaluation-harness/` submodule, reinitialize it properly with:
> ```bash
> git submodule update --init --recursive
> ```

For example, to evaluate [ModernBERT-large-chat-v1](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v1) on [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) using 4 GPUs, run:
```shell
# Use model_args to adjust the generation arguments for evalution.
accelerate launch  --num_processes 4 \
    dllm/pipelines/bert/eval.py \
    --tasks mmlu_pro \
    --batch_size 1 \
    --model bert \
    --seed 1234 \
    --device cuda \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-collection/ModernBERT-large-chat-v1,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256"
```

To perform full evaluations on all benchmark datasets with consistent generation parameters for both [ModernBERT-base-chat-v1](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v1) and [ModernBERT-large-chat-v1](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v1), use the preconfigured script:
```shell
bash examples/bert/eval.sh <model_path>
# <model_path>: Local path or huggingface model ID
# BERT model default to use --apply_chat_template argument
```