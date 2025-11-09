export BASE_DATASETS_DIR=/mnt/lustrenew/mllm_safety-shared/datasets/huggingface
export BASE_MODELS_DIR=/mnt/lustrenew/mllm_aligned/shared/models/tmp
export HF_DATASETS_CACHE=/mnt/lustrenew/mllm_safety-shared/datasets/huggingface
export HF_EVALUATE_CACHE=/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/.cache/
export PYTHONPATH=.:$PYTHONPATH

# Check whether two model files are equivalent
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"

# CKPT_DIR="/mnt/lustrenew/mllm_aligned/shared/models/tmp/Dream/Dream-7B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5-gbl"
# FINAL="$CKPT_DIR/checkpoint-final"
# CKPT5355="$CKPT_DIR/checkpoint-5355"

# echo "Comparing checkpoint-final vs checkpoint-5355"
# echo

# for f in "$FINAL"/model-*.safetensors; do
#     fname=$(basename "$f")
#     f1="$FINAL/$fname"
#     f2="$CKPT5355/$fname"

#     if [[ ! -f "$f2" ]]; then
#         echo "Missing in 5355: $fname"
#         continue
#     fi

#     md1=$(md5sum "$f1" | awk '{print $1}')
#     md2=$(md5sum "$f2" | awk '{print $1}')

#     if [[ "$md1" == "$md2" ]]; then
#         echo "$fname: EQUAL"
#     else
#         echo "$fname: DIFF"
#         echo "  final:  $md1"
#         echo "  5355:   $md2"
#     fi
# done


model_name="Dream/Dream-7B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5-gbl/checkpoint-final"
use_instruct="True"
LIMIT=20


# 6214291 - 6214297 6214318 6214327
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_pro ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative_dream ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval_instruct ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct_dream ${model_name} True 1 True ${LIMIT}
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream ifeval ${model_name} True 1 True ${LIMIT}

# 6216308 - 6216312
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada gsm8k_cot  ${model_name} True 1 False ${LIMIT}
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada bbh  ${model_name} True 1 False ${LIMIT}
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada minerva_math  ${model_name} True 1 False ${LIMIT}
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada humaneval_instruct  ${model_name} True 1 False ${LIMIT}
sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mbpp_llada_instruct  ${model_name} True 1 False ${LIMIT}


# # 6214497 - 6214510
# sbatch --gres=gpu:8 --ntasks-per-node=8 scripts/eval.slurm.sh bert mmlu "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert ceval-valid "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert cmmlu "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False 10
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert gsm8k_cot "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert gsm8k "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 True 10
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert minerva_math "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert bbh "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert hellaswag "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False 100
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert winogrande "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None

# # 6215149 - 6215156
# sbatch --gres=gpu:8 --ntasks-per-node=8 scripts/eval.slurm.sh bert mmlu "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert ceval-valid "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert cmmlu "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False 10
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert gsm8k_cot "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert gsm8k "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert minerva_math "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert bbh "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert hellaswag "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh bert winogrande "ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final" True 1 False None


# Unexecuted
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada gsm8k GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada bbh GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada minerva_math GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada humaneval GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mbpp GSAI-ML/LLaDA-8B-Base False 1 False

# Executed
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada gpqa_main_n_shot GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada truthfulqa_mc2 GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada arc_challenge GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada hellaswag GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada winogrande GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada piqa GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mmlu GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada cmmlu GSAI-ML/LLaDA-8B-Base False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada ceval-valid GSAI-ML/LLaDA-8B-Base False 1 False

# Unexecuted
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada gsm8k_cot GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada bbh GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada minerva_math GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada humaneval_instruct GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mbpp_llada_instruct GSAI-ML/LLaDA-8B-Instruct True 1 False

# Executed
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mmlu_generative GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada mmlu_pro GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada hellaswag_gen GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada arc_challarc_challenge_chatenge GSAI-ML/LLaDA-8B-Instruct True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh llada gpqa_n_shot_gen GSAI-ML/LLaDA-8B-Instruct True 1 False


# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gsm8k_cot Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream bbh Dream-org/Dream-v0-Base-7B False 1 False

# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream arc_easy Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream arc_challenge Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream hellaswag Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream piqa Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream winogrande Dream-org/Dream-v0-Base-7B False 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream race Dream-org/Dream-v0-Base-7B False 1 False

# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_generative Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mmlu_pro Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gsm8k_cot Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream minerva_math Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream gpqa_main_n_shot Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream humaneval_instruct Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream mbpp_instruct Dream-org/Dream-v0-Instruct-7B True 1 False
# sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/eval.slurm.sh dream ifeval Dream-org/Dream-v0-Instruct-7B True 1 False
