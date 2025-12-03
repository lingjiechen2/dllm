```shell
srun -p $PARTITION --quotatype=spot --gres=gpu:1 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen2.5-0.5B" --output_dir "models/a2d/Qwen2.5-0.5B"
```

```shell
srun -p $PARTITION --quotatype=spot --gres=gpu:8 --cpus-per-task=24 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 \
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/sft.py \
        --model_name_or_path "models/a2d/Qwen2.5-0.5B" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --save_steps 0.1 \
        --right_shift_logits True \
        --output_dir "models/a2d/Qwen2.5-0.5B/alpaca/right-shift"

srun -p $PARTITION --quotatype=spot --gres=gpu:8 --cpus-per-task=24 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 \
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/sft.py \
        --model_name_or_path "models/a2d/Qwen2.5-0.5B" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --save_steps 0.1 \
        --right_shift_logits False \
        --output_dir "models/a2d/Qwen2.5-0.5B/alpaca/non-right-shift"
```



### Evaluation results

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

opendCoder
humaneval_instruct: 20.8%
mbpp-instruct: 35.2%


Qwen3-0.6B-right-shift-tulu-3-smoltalk-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
gsm8k: 28.7%
MATH: 9.3%
bbh: 28.6%
mmlu_pro: 16.2%
hellaswag: 42.2%
mmlu: 40.1%
humaneval_instruct: 28.1%
mbpp_instruct: 31.4%

Qwen3-0.6B-non-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
humaneval_instruct: 31.7%
mbpp_instruct: 29.0%

Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
humaneval_instruct: 29.9%
mbpp_instruct: 25.2%

Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024
humaneval_instruct: 32.9%
mbpp_instruct: 27.4%

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