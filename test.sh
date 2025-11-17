#!/bin/bash

# Total samples and number of GPUs
TOTAL_SAMPLES=1000
NUM_GPUS=50
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))

# Create output directory
OUTPUT_DIR="./multi_granularity_outputs"
mkdir -p ${OUTPUT_DIR}

# Launch jobs for each GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * SAMPLES_PER_GPU))
    END=$(((i + 1) * SAMPLES_PER_GPU))
    
    # Define output file path (adjust the pattern to match your actual output file naming)
    OUTPUT_FILE="${OUTPUT_DIR}/multi_granularity_gsm8k_${START}_${END}.jsonl"
    
    # Check if output file already exists
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "Job $i: Output file ${OUTPUT_FILE} already exists. Skipping..."
        continue
    fi
    
    echo "Launching job $i: samples ${START} to ${END}"
    
    srun -p mllm_safety --quotatype=spot --gres=gpu:1 --time=03:00:00 \
        python build_multigranularity_dataset.py \
        --start ${START} \
        --end ${END} &
    
    # Small delay to avoid overwhelming the scheduler
    sleep 0.5
    # break
done

echo "All jobs submitted. Waiting for completion..."
wait
echo "All jobs completed!"

# Optional: Combine all outputs into a single file
# echo "Combining outputs..."
# cat multi_granularity_gsm8k_*.jsonl > ${OUTPUT_DIR}/multi_granularity_gsm8k_full.jsonl
# echo "Combined output saved to ${OUTPUT_DIR}/multi_granularity_gsm8k_full.jsonl"
