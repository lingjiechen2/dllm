# Evaluation

We provide a **unified evaluation framework** built on top of **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**, serving as the standardized backbone for evaluating the [LLaDA series,](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) [Dream series](https://huggingface.co/collections/Dream-org/dream-7b), and [BERT-diffusion series](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v1).
It supports diverse model architectures and evaluation paradigms through a **configuration-driven**, **modular**, and **extensible** design.



## Table of Contents
1. [Setup](#setup)
   - [Environment Variables](#environment-variables)
   - [Dependencies](#dependencies)
2. [Evaluation](#evaluation)
   - [Run Command](#run-command)
   - [Plausible Tasks](#plausible-tasks)
   - [Example Evaluation Results](#example-evaluation-results)
3. [Framework and Further Extension](#framework-and-further-extension)
   - [File Structure](#file-structure)


---

## Setup

> [!IMPORTANT]
> Before running evaluations, you **must** make sure all files are fetched from github repo, check whether `/lm-evaluation-harness/lm-eval/tasks` exists. 

<!-- The following part is only for eval.slurm.sh
### Environment Variables

Before running evaluations, export the following environment variables to specify where datasets, pretrained models, and caches are stored:

```shell
export BASE_DATASETS_DIR=<path_to_huggingface_datasets>
export BASE_MODELS_DIR=<path_to_local_or_shared_models>
export HF_DATASETS_CACHE=<path_to_hf_dataset_cache>
export HF_EVALUATE_CACHE=<path_to_hf_evaluate_cache>
export PYTHONPATH=.:$PYTHONPATH
``` -->


### Dependencies

Install the core dependencies:

```shell
pip install -e lm-evaluation-harness
pip install accelerate transformers datasets
pip install -e ".[ifeval,math]"
```

Make sure to initialize submodules before installation:
```shell
git submodule update --init --recursive
```


## Evaluation

### Run Command

> [!NOTE]
> All configuration parameters (e.g., few-shot settings, max length, temperature, etc.) are aligned with the model’s original repository.

**Example commands:**
For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) using 4 GPUs, run:
```shell
# Use model_args to adjust the generation arguments for evalution.
accelerate launch  --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks mmlu_pro \
    --batch_size 1 \
    --model llada \
    --seed 1234 \
    --device cuda \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"
```
You can also run one-line evaluations using model-specific scripts under the [examples/](examples) directory.
Each script automatically loads its corresponding configurations and launches evaluation — no need to manually specify `model_class`, `task_name`, `model_path` or `model_args`.
Detailed instructions are provided in the [examples/](examples)'s README.md.

### Plausible Tasks

| Category | Tasks |
|----------|-------|
| **Instruct** | `mmlu_generative`, `mmlu_pro`, `gsm8k_cot`, `minerva_math`, `gpqa_main_n_shot`, `humaneval_instruct`, `mbpp_instruct`, `ifeval` |
| **Base** | `humaneval`, `gsm8k_cot`, `mbpp`, `minerva_math`, `bbh`, `mmlu`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `gpqa_main_n_shot`, `winogrande`, `race` |

> [!NOTE]
> Certain dataset configurations of lm-eval were refined to match the model templates and ensure optimal evaluation performance.


### Example Evaluation Results

<details>
<summary><strong>LLaDA-Base results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



<details>
<summary><strong>LLaDA-Instruct results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



<details>
<summary><strong>Dream-Base results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>


<details>
<summary><strong>Dream-Instruct results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



## Framework and Further Extension

> [!NOTE]
> Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements model-specific generation and likelihood computation methods.

Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements:

- **`generate_until()`** — defines model-specific text generation (e.g., diffusion or autoregressive).
- **`loglikelihood()`** — computes NLL or masked likelihood (Monte Carlo, autoregressive, etc.).
- **`apply_chat_template()`** — formats multi-turn inputs when `--apply_chat_template=True`.

This modular design allows adding new model architectures while keeping the evaluation pipeline unified.

### Customizing Tasks

> [!NOTE]
> Customize evaluation behavior by editing YAML configuration files — no code changes required.

To customize or extend tasks, edit the configuration files in:

```
lm-evaluation-harness/lm_eval/tasks/<task_name>/<task_name>.yaml
```

For example:

```
lm-evaluation-harness/lm_eval/tasks/mbpp/mbpp.yaml
```

Each YAML file defines:

- Dataset sources and splits
- Prompt templates and context formatting
- Metric computation and postprocessing
- Stop sequences and answer extraction rules

By editing these YAMLs, you can modify task behavior or introduce new benchmarks without rebuilding the framework.

### Adding New Models

> [!NOTE]
> New model types can be integrated while maintaining **full compatibility** with the unified evaluation system.

To integrate a new model type:

1. **Create a new evaluation file**, e.g. `dllm/eval/eval_newmodel.py`.

2. **Register it with lm-eval**:

   ```python
   from lm_eval.api.registry import register_model

   @register_model("newmodel")
   class NewModel(LM):
       ...
   ```

3. **Implement `generate_until()` and `loglikelihood()`** for the model's decoding logic.

4. **Add corresponding entries to `eval_configs.sh`** for task configurations.

> [!NOTE]
> This approach supports both custom and standard model backends, making the framework highly extensible.

## Acknowledgments
We sincerely thank [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their outstanding contributions to the open evaluation ecosystem