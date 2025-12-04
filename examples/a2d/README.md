# A2D (AR-to-Diffusion)

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-collection/tiny-a2d)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)]([TODO])


This directory provides two key sets of resources:

- **Warmup ([MDLM](#warmup-mdlm) and [BM3LM](#warmup-bm3lm))**: Tutorials for continual pretraining and SFTing any autoregressive model on small datasets to generate text with MDLM (masked diffusion) or BM3LM (block diffusion).
- **[Tiny-A2D](#tiny-a2d)**: The exact training, inference, and evaluation scripts used to create [TODO].
<!-- -  **[BERT-Chat](#bert-chat)**: The exact training, inference, and evaluation scripts used to create the [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0) and [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0) checkpoints, two BERTs finetuned as Chatbots. For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [BERT-Chat W&B Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg). -->

## Files overview
```
# example entry points for training / inference / evaluation
examples/a2d
├── bm3lm               # Block Discrete Denoising Diffusion Language Modeling (https://arxiv.org/abs/2503.09573)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
├── mdlm                # Masked Diffusion Language Modeling (https://arxiv.org/abs/2406.07524)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
└── README.md
```

## Setup 

1. **Customize modeling files**: You must first modify the original autoregressive modeling file to support non-causal attention. See [`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py#L77-L108) for an example, and update [`__init__.py`](/dllm/pipelines/a2d/__init__.py) accordingly to register the new model config and architecture.

2. **Run unit tests**: Before proceeding with your customized models, ensure they pass:
    ```shell
    pytest scripts/tests/test_attention.py::test_a2d_attention_mask_invariance
    pytest scripts/tests/test_attention.py::test_a2d_fullmask_future_affects_past
    # Optional: only needed for BM3LM
    pytest scripts/tests/test_attention.py::test_a2d_staircase_attention_kvcache_equivalence
    ```

3. **Convert an AR model with customized attention**: For example, to convert `Qwen/Qwen3-0.6B` using its original weights but with the customized attention defined in [`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py):
    ```shell
    python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/a2d/Qwen3-0.6B"
    ```

## Warmup: [MDLM](https://arxiv.org/abs/2406.07524)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [MDLM](https://arxiv.org/abs/2406.07524) (masked diffuions).

### Continual Pretraining

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with MDLM, run:

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/mdlm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/mdlm/alpaca/checkpoint-final" --block_size 32
```

## Warmup: [BM3LM](https://arxiv.org/abs/2503.09573)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [BD3LM](https://arxiv.org/abs/2503.09573) (block diffuions).

### Continual Pretraining

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bm3lm/pt.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --block_size 32 \
    --output_dir "models/a2d/Qwen3-0.6B/bm3lm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/bm3lm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/bm3lm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --block_size 32 --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with [BD3LM](https://arxiv.org/abs/2503.09573) (block diffuions), run:

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bm3lm/sft.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --block_size 32 \
    --output_dir "models/a2d/Qwen3-0.6B/bm3lm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/bm3lm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/bm3lm/alpaca/checkpoint-final" --block_size 32
```


## Tiny-A2D

### Evaluation


|                     | LAMBADA | GSM8K | CEval | BBH | MATH | MMLU | Winogrande | HellaSwag | CMMLU |
|:------------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0)(evaluated) | 49.3 | 5.9 | 25.0 | 17.9 | 3.1 | 26.1 | 49.7 | 41.0 | 24.3 |
| [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0)(evaluated) | 46.3 | 17.1 | 24.6 | 25.1 | 3.8 | 33.5 | 53.1 | 45.0 | 27.5 |
| [`Qwen1.5-0.5B`](https://huggingface.co/Qwen/Qwen1.5-0.5B)(<ins>reported</ins> & evaluated) | 48.6 | <ins>22.0</ins> | <ins>50.5</ins> | <ins>18.3</ins> | <ins>3.1</ins> | <ins>39.2</ins> | 55.0 | 48.2 | <ins>46.6</ins> |
| [`Qwen1.5-0.5B-Chat`](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)(<ins>reported</ins> & evaluated) | 41.2 | <ins>11.3</ins> | <ins>37.2</ins> | 18.2 | 2.1 | <ins>35.0</ins> | 52.0 | 36.9 | 32.2 |
| [`gpt2`](https://huggingface.co/openai-community/gpt2)(<ins>reported</ins> & evaluated) | <ins>46.0</ins> | 0.7 | 24.7 | 6.9 | 1.8 | 22.9 | 51.6 | 31.1  | 25.2 |
| [`gpt2-medium`](https://huggingface.co/openai-community/gpt2-medium)(<ins>reported</ins> & evaluated) | <ins>55.5</ins> | 2.1 | 24.6 | 17.8 | 1.4 | 22.9 |53.1  | 39.4  | 0.3  |



Qwen3-0.6B-non-shift-tulu-3-smoltalk-epochs-10-bs-256-len-1024
gsm8k: 22.9%
MATH: 7.9%
bbh: 23.7%
mmlu_pro: 15.3%
hellaswag: 35.9
mmlu: 36.0%
humaneval_instruct: 14.6%
mbpp_instruct: 15.2%

Qwen3-0.6B-right-shift-tulu-3-smoltalk-epochs-10-bs-256-len-1024
gsm8k: 26.5%
MATH: 7.7%
bbh: 30.1%
mmlu_pro: 14.9%
hellaswag: 36.4
mmlu: 36.6%
humaneval_instruct: 8.5%
mbpp_instruct: 13.6%

Qwen3-0.6B-non-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
gsm8k: 29.8%
MATH:8.8%
bbh: 27.0%
mmlu_pro: 17.6%
hellaswag: 42.1%
mmlu: 40.0%
humaneval_instruct: 30.5%
mbpp_instruct: 29.2%

Qwen3-0.6B-non-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-20-bs-2048-len-1024
gsm8k: 33.4%
MTH:9.5%
bbh: 27.8%
mmlu_pro: 16.7%
hellaswag: 42.9%
mmlu: 41.3%
humaneval_instruct: 34.2%
mbpp_instruct: 30.4%

Qwen3-0.6B-right-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
gsm8k: 28.7%
MATH: 9.3%
bbh: 28.6%
mmlu_pro: 16.2%
hellaswag: 42.2%
mmlu: 40.1%
humaneval_instruct: 28.1%
mbpp_instruct: 31.4%

Qwen3-0.6B-non-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-512-bls-32
gsm8k: 46.6% 
bbh: 27.0% 
mmlu_pro: 14.1%
hellaswag: 40.0%
mmlu: 38.8%
humaneval_instruct: 47.6%
mbpp_instruct:32.0%

Qwen3-0.6B-non-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
humaneval_instruct: 31.7%
mbpp_instruct: 29.0%

Qwen2.5-Coder-0.5B-Instruct-non-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-512-bls-32
humaneval_instruct: 41.5%
mbpp_instruct: 33.6%

Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
humaneval_instruct: 29.9%
mbpp_instruct: 25.2%

Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-20-bs-2048-len-1024
humaneval_instruct: 32.9%
mbpp_instruct: 27.4%

Qwen2.5-Coder-0.5B-Instruct-right-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-512-bls-32
humaneval_instruct: 38.4%
mbpp_instruct: 31.6%


opendCoder
humaneval_instruct: 20.8%
mbpp-instruct: 35.2%


Qwen2.5-0.5B-Instruct-non-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024
gsm8k: 15.2%
MATH: 6.0%
bbh: 26.5%
mmlu_pro: 11.3%
hellaswag: 28.9%
mmlu: 31.3%
humaneval_instruct: 2.4%
mbpp_instruct: 7.6%

Qwen2.5-0.5B-Instruct-right-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024
gsm8k: 15.0%
MATH: 5.4%
bbh: 24.5%
mmlu_pro: 11.8%
hellaswag: 30.1%
mmlu: 30.0%
humaneval_instruct: 2.4%
mbpp_instruct: 7.8%

Qwen2.5-0.5B-non-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024
gsm8k: 15.6%
MATH: 6.4%
bbh: 23.5%
mmlu_pro: 8.8%
hellaswag: 30.1%
mmlu: 30.4
humaneval_instruct: 4.3%
mbpp_instruct: 7.6%

Qwen2.5-0.5B-right-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024
gsm8k: 15.6%
MATH: 6.2%
bbh: 22.5%
mmlu_pro: 6.9%
hellaswag: 30.7%
mmlu: 30.2%
humaneval_instruct: 3.7%
mbpp_instruct: 9.6%


Qwen2.5-Coder-0.5B-Instruct-right-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 26.2%
mbpp-instruct: 19.0%

Qwen2.5-Coder-0.5B-Instruct-non-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 28.1%
mbpp-instruct: 23%

Qwen2.5-Coder-0.5B-right-shift-opc-annealing-corpus->opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 12.2%
mbpp-instruct: 9.6%

Qwen2.5-Coder-0.5B-non-shift-opc-annealing-corpus->opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 10.4%
mbpp-instruct: 15.7%

Qwen2.5-Coder-0.5B-right-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 11.6%
mbpp-instruct: 20.1%

Qwen2.5-Coder-0.5B-non-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024
humaneval_instruct: 15.9%
mbpp-instruct: 19%




|                Model                   | GSM8K | MATH | BBH | MMLU-Pro | HellaSwag | MMLU | HumanEval-Instruct | MBPP-Instruct |
|:------|:----:|:----:|:---:|:--------:|:---------:|:----:|:------------------:|:--------------:|
| **Qwen3-0.6B-non-shift-tulu-3-smoltalk-epochs-10-bs-256-len-1024** | 22.9 | 7.9 | 23.7 | 15.3 | 35.9 | 36.0 | 14.6 | 15.2 |
| **Qwen3-0.6B-right-shift-tulu-3-smoltalk-epochs-10-bs-256-len-1024** | 26.5 | 7.7 | 30.1 | 14.9 | 36.4 | 36.6 | 8.5 | 13.6 |
| **Qwen3-0.6B-non-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 29.8 | 8.8 | 27.0 | 17.6 | 42.1 | 40.0 | 30.5 | 29.2 |
| **Qwen3-0.6B-right-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 28.7 | 9.3 | 28.6 | 16.2 | 42.2 | 40.1 | 28.1 | 31.4 |
| **Qwen2.5-0.5B-Instruct-non-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024** | 15.2 | 6.0 | 26.5 | 11.3 | 28.9 | 31.3 | 2.4 | 7.6 |
| **Qwen2.5-0.5B-Instruct-right-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024** | 15.0 | 5.4 | 24.5 | 11.8 | 30.1 | 30.0 | 2.4 | 7.8 |
| **Qwen2.5-0.5B-non-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024** | 15.6 | 6.4 | 23.5 | 8.8 | 30.1 | 30.4 | 4.3 | 7.6 |
| **Qwen2.5-0.5B-right-shift-tulu-3-smoltalk-epochs-10-bs-384-len-1024** | 15.6 | 6.2 | 22.5 | 6.9 | 30.7 | 30.2 | 3.7 | 9.6 |

|                Model                   | HumanEval-Instruct | MBPP-Instruct |
|:---------------------------------------|:------------------:|:--------------:|
| **Qwen3-0.6B-non-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 31.7 | 29.0 |
| **Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 29.9 | 25.2 |
| **Qwen3-0.6B-non-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 30.5 | 29.2 |
| **Qwen3-0.6B-right-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024** | 28.1 | 31.4 |
| **Qwen2.5-Coder-0.5B-Instruct-non-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024**  | 28.1 | 23.0 |
| **Qwen2.5-Coder-0.5B-Instruct-right-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024** | 26.2 | 19.0 |
| **Qwen2.5-Coder-0.5B-non-shift-opc-annealing-corpus->opc-sft-stage1&2-epochs-10-bs-1536-len-1024**    | 10.4 | 15.7 |
| **Qwen2.5-Coder-0.5B-right-shift-opc-annealing-corpus->opc-sft-stage1&2-epochs-10-bs-1536-len-1024**  | 12.2 | 9.6 |
| **Qwen2.5-Coder-0.5B-non-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024**       | 15.9 | 19.0 |
| **Qwen2.5-Coder-0.5B-right-shift-opc-sft-stage1&2-epochs-10-bs-1536-len-1024**     | 11.6 | 20.1 |