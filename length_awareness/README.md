# Length Awareness

Centralizes length-related tooling: probe extraction, dataset builders, token-budget evaluation, and saved artifacts.

## Layout
- `probes/` — Slurm helpers and extractor to dump hidden states/labels (`probe_starter.sh`, `probe_job.sh`, `extract_probe_data.py`).
- `dataset_builders/` — Generation-based dataset scripts (`build_multigranularity_dataset.py`, `build_QAT_dataset.py`).
- `evaluation/` — Token-budget evaluation sweep (`eval_token_budget.sh`).
- `outputs/` — Data drop zone. Existing eval logs/samples live under `outputs/evals/` (`gsm8k_cot`, `gsm8k_cot_llama`, `humaneval_instruct`, `minerva_math`). `outputs/datasets/` and `outputs/probes/` are reserved for new artifacts.

## Quickstart
- **Env**: run from repo root with `PYTHONPATH=.` (or `source env.sh` if you use the shared BASE_* paths). Most scripts assume Slurm for job submission.
- **Probes**: submit all splits with `bash length_awareness/probes/probe_starter.sh [steps] [max_new_tokens]`. It enqueues `probe_job.sh` jobs and skips splits already saved under `/mnt/lustrenew/mllm_safety-shared/fanyuyu/probe_data/...`. For a single split:  
  `sbatch length_awareness/probes/probe_job.sh <split_idx> <num_splits> <dataset> <dataset_config> <dataset_split> <steps> <max_new_tokens> <model_tag> <model_path> [num_extract_steps]`
- **Datasets**: generate multi-granularity answers from GSM8K:  
  `python length_awareness/dataset_builders/build_multigranularity_dataset.py --start 0 --end 100 --output length_awareness/outputs/datasets/multi_granularity_gsm8k_0_100.jsonl`  
  Build answer-first QAT samples:  
  `python length_awareness/dataset_builders/build_QAT_dataset.py > length_awareness/outputs/datasets/answer_first_dataset.jsonl`
- **Token-budget eval**: run `bash length_awareness/evaluation/eval_token_budget.sh` (uses `sbatch` and `eval_dllm.sh`). Adjust `BASE_MODELS_DIR`, `task_name`, and `length_list` in the script; outputs land under `multi_granularity/<task>/<model>/`.

## Notes
- Keep bulky results under `length_awareness/outputs/` and consider ignoring them in version control.
- Existing eval logs/samples were moved into `length_awareness/outputs/evals/` for safekeeping.***
