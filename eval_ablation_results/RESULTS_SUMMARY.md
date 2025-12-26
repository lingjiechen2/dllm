# Eval Ablation Results (Extracted)

Source directory: `eval_ablation_results`
Generated (UTC): `2025-12-26T00:24:12.664286+00:00`
Total result files: 18

## Missing result files
- `eval_ablation_results/bbh_100`: no `results_*.json` found
- `eval_ablation_results/minerva_math_100`: no `results_*.json` found

## `gsm8k_100`

### `gsm8k`

| variant | exact_match,flexible-extract | exact_match,strict-match | exact_match_stderr,flexible-extract | exact_match_stderr,strict-match | date_utc | model | pretrained | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.73 | 0.75 | 0.0446 | 0.0435 | 2025-12-16T06:35:36.092787+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/baseline/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T03-09-01.507525.json` |
| block32 | 0.69 | 0.73 | 0.0465 | 0.0446 | 2025-12-17T06:52:48.342049+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/block32/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T02-12-47.612964.json` |
| len1024_step512 | 0.65 | 0.7 | 0.0479 | 0.0461 | 2025-12-16T20:10:30.688451+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/len1024_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T14-50-35.909704.json` |
| len512_step512 | 0.67 | 0.73 | 0.0473 | 0.0446 | 2025-12-17T02:28:43.156096+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/len512_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T20-59-21.420247.json` |
| remask_random | 0.32 | 0.28 | 0.0469 | 0.0451 | 2025-12-18T06:54:38.596661+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/remask_random/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-18T02-13-57.773246.json` |
| temp0.5 | 0.73 | 0.75 | 0.0446 | 0.0435 | 2025-12-17T18:53:57.255124+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/gsm8k_100/temp0.5/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T14-13-56.077522.json` |

## `humaneval_100`

### `humaneval`

| variant | pass@1,create_test | pass@1_stderr,create_test | date_utc | model | pretrained | file |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.08 | 0.0273 | 2025-12-16T18:04:16.534557+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/baseline/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T12-52-47.154656.json` |
| block32 | 0.12 | 0.0327 | 2025-12-17T16:51:44.050514+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/block32/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T11-40-05.682264.json` |
| len1024_step512 | 0.05 | 0.0219 | 2025-12-17T01:26:52.520890+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/len1024_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T19-51-16.398640.json` |
| len512_step512 | 0.1 | 0.0302 | 2025-12-17T06:10:52.015165+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/len512_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T00-26-18.550409.json` |
| remask_random | 0.15 | 0.0359 | 2025-12-18T16:49:23.504314+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/remask_random/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-18T11-37-24.378355.json` |
| temp0.5 | 0.08 | 0.0273 | 2025-12-18T04:52:29.970945+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/humaneval_100/temp0.5/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T23-40-49.874506.json` |

## `mbpp_100`

### `mbpp`

| variant | pass_at_1,none | pass_at_1_stderr,none | date_utc | model | pretrained | file |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.43 | 0.0498 | 2025-12-16T18:53:06.091446+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/baseline/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T14-10-12.423744.json` |
| block32 | 0.4 | 0.0492 | 2025-12-17T17:40:22.777233+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/block32/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T12-53-40.416191.json` |
| len1024_step512 | 0.32 | 0.0469 | 2025-12-17T01:51:34.091456+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/len1024_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-16T20-28-25.790555.json` |
| len512_step512 | 0.43 | 0.0498 | 2025-12-17T06:26:36.253066+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/len512_step512/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-17T00-52-29.658042.json` |
| remask_random | 0.18 | 0.0386 | 2025-12-18T17:37:41.567811+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/remask_random/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-18T12-50-21.489922.json` |
| temp0.5 | 0.43 | 0.0498 | 2025-12-18T05:41:07.243481+00:00 | llada | /home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base/ | `eval_ablation_results/mbpp_100/temp0.5/__home__lingjie7__models__huggingface__GSAI-ML__LLaDA-8B-Base__/results_2025-12-18T00-54-21.641987.json` |

