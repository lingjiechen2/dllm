---
library_name: transformers
license: apache-2.0
license-link: https://huggingface.co/answerdotai/ModernBERT-large/blob/main/LICENSE
pipeline_tag: text-generation
---

<center> <div style="text-align: center;"> <img src="https://raw.githubusercontent.com/ZHZisZZ/dllm/rnd/assets/logo.gif" width="400" />
 </div> </center>

# ModernBERT-large-chat

ModernBERT-large-chat is a generative variant of [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), fine-tuned using the [dLLM](https://github.com/ZHZisZZ/dllm) framework.
This model demonstrates that BERT can serve as a diffusion-based chatbot trained purely with supervised instruction data — no autoregressive pretraining required.

## Model Overview

ModernBERT-large-chat has the following features:

- **Type**: Diffusion-based Generative BERT
- **Architecture**: Transformer encoder with 8192-token context
- **Training Objective**: Supervised fine-tuning on instruction–response pairs
- **Framework**: [dLLM](https://github.com/ZHZisZZ/dllm)
- **Base Model**: [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)
- **Datasets**: [TULU-3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)

For detailed training methodology, see the [dLLM–BERT report on W&B](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-BERT--VmlldzoxNDg0MzExNg).

## Installation

```bash
pip install torch transformers accelerate
```

For diffusion-based generation support:

```bash
pip install git+https://github.com/ZHZisZZ/dllm
```

## Quick Start

```python
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=64, temperature=0.0, cfg_scale=0., remasking='random'):
    mask_id = tokenizer.mask_token_id
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


device = 'cuda'
model = AutoModelForMaskedLM.from_pretrained('dllm-collection/ModernBERT-large-chat-v0', dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('dllm-collection/ModernBERT-large-chat-v0')

prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
m = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": prompt}
]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

text = generate(model, input_ids, steps=128, gen_length=128, block_length=64, temperature=0.0, cfg_scale=0.0, remasking='random')
print(tokenizer.batch_decode(text[:, input_ids.shape[1]:], skip_special_tokens=False)[0])
```

## Generation Parameters

| Parameter        | Description                                                                                    | Default  |
| ---------------- | ---------------------------------------------------------------------------------------------- | -------- |
| `max_new_tokens` | Number of tokens to generate                                                                   | 128      |
| `steps`          | Number of diffusion denoising iterations                                                       | 128      |
| `temperature`    | Sampling temperature; set to `0.0` for deterministic generation                                | 0.0      |
| `block_length`   | Token block size used during iterative denoising                                               | 64       |
| `cfg_scale`      | Classifier-free guidance scale controlling instruction adherence (higher = more deterministic) | 0.0      |
| `remasking`      | Strategy for re-masking during each denoising step (`random`, `none`, or `confidence`)         | `random` |

## Command-Line Interface

Folowing the Github repo's demo script [examples/bert/chat.py](https://github.com/ZHZisZZ/dllm/blob/rnd/examples/bert/chat.py)

```bash
python -u examples/bert/chat.py \
  --model_name_or_path dllm-collection/ModernBERT-large-chat-v0 \
  --chat True
```

## Technical Details

ModernBERT-large-chat is trained using the dLLM framework, which extends Masked Language Modeling (MLM) to Masked Diffusion Language Modeling (MDLM) — sampling mask ratios from 0–100% during training.
This enables iterative denoising-based generation rather than token-by-token decoding.

Fine-tuning on the TULU-3 and Smoltalk datasets grants instruction-following and conversational fluency. Despite lacking autoregressive training, the model produces coherent, contextually relevant dialogue responses.

## Citation

If you use ModernBERT-large-chat or dLLM, please cite:

```bibtex
@misc{dllm,
  author = {Zhanhui Zhou and Lingjie Chen and Hanghang Tong and Dawn Song},
  title = {dLLM: Simple Diffusion Language Modeling},
  year = {2025},
  howpublished = {\url{https://github.com/ZHZisZZ/dllm}},
}





# 6204965 - 6204971 6205021
sbatch --gres=gpu:8 --ntasks-per-node=8 scripts/eval.slurm.sh dream mmlu Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream arc_easy Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream arc_challenge Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream hellaswag Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream piqa Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream winogrande Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream race Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot Dream-org/Dream-v0-Base-7B False 1 False


# 6205055 6205066 6205067 6205077 6205078
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval_dream Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gsm8k_cot Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp Dream-org/Dream-v0-Base-7B False 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream bbh Dream-org/Dream-v0-Base-7B False 1 False


sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math Dream-org/Dream-v0-Base-7B False 1 False
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream bbh Dream-org/Dream-v0-Base-7B False 1 False

# 6214257 6214258
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_pro Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative_dream Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval_instruct Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct_dream Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream ifeval Dream-org/Dream-v0-Instruct-7B True 1 True
sbatch --gres=gpu:8 --ntasks-per-node=8 scripts/eval.slurm.sh dream bbh Dream-org/Dream-v0-Instruct-7B True 1 True



export model_name="Dream/Dream-7B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5-gbl/checkpoint-final"
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_pro ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative_dream ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval_instruct ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct_dream ${model_name} True 1 True
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream ifeval ${model_name} True 1 True
