#!/bin/bash

# --- Configuration ---
# Path to the directory containing your trained checkpoints (e.g., ckpts/checkpoint-100)
CKPT_DIR="/opt/tiger/ckpts"

# Path to the original, un-trained Llama-3.2-1B model
BASELINE_MODEL_PATH="/opt/tiger/Llama-3.2-1B" 

# Directory to store all evaluation results (CSVs and plots)
OUTPUT_DIR="./degeneration_results"

# Comma-separated list of checkpoint steps to evaluate
CKPT_STEPS="100,200,300,400,500"

# Comma-separated list of loop counts to test for each checkpoint
LOOP_COUNTS="0,1,2,3,4,5"

# Random seed for reproducibility
SEED=42

# --- Script Logic ---
echo "Starting model degeneration evaluation..."
mkdir -p "$OUTPUT_DIR"
# Create a directory for logs
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Logs will be stored in $LOG_DIR"

# Check if baseline model path is valid
if [ ! -d "$BASELINE_MODEL_PATH" ]; then
    echo "Error: Baseline model path not found at '$BASELINE_MODEL_PATH'"
    echo "Please update the BASELINE_MODEL_PATH variable in this script."
    exit 1
fi

# Convert comma-separated string to array
IFS=',' read -r -a steps_array <<< "$CKPT_STEPS"

# Array to hold background process IDs
pids=()

# --- Run Baseline Evaluation (GPU 0) ---
current_gpu=0
echo "-----------------------------------------------------"
echo "Launching baseline evaluation on GPU $current_gpu... Log: $LOG_DIR/baseline_eval.log"
echo "-----------------------------------------------------"
CUDA_VISIBLE_DEVICES=$current_gpu python LoopLLM/degeneration_test/run_baseline_eval.py \
    --model_path "$BASELINE_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda:0" \
    --seed $SEED > "$LOG_DIR/baseline_eval.log" 2>&1 &
pids+=($!)

# --- Run LoopLlama Checkpoint Evaluations (GPUs 1, 2, ...) ---
for step in "${steps_array[@]}"; do
    current_gpu=$((current_gpu + 1))
    ckpt_path="$CKPT_DIR/checkpoint-$step"

    if [ ! -d "$ckpt_path" ]; then
        echo "Warning: Checkpoint directory not found, skipping: $ckpt_path"
        continue
    fi
    
    log_file="$LOG_DIR/checkpoint-${step}_eval.log"
    echo "-----------------------------------------------------"
    echo "Launching evaluation for checkpoint-$step on GPU $current_gpu... Log: $log_file"
    echo "-----------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$current_gpu python LoopLLM/degeneration_test/run_single_ckpt_eval.py \
        --ckpt_path "$ckpt_path" \
        --base_model_path "$BASELINE_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --loop_counts "$LOOP_COUNTS" \
        --device "cuda:0" \
        --seed $SEED > "$log_file" 2>&1 & # Inside the container, it's always cuda:0
    pids+=($!)
done

# --- Wait for all background jobs to finish ---
echo -e "\nWaiting for all evaluation scripts to complete... Process IDs: ${pids[*]}"
wait

echo -e "\n-----------------------------------------------------"
echo "All evaluations have finished. You can check the logs in the '$LOG_DIR' directory."
echo "-----------------------------------------------------"

# --- Plotting Results ---
echo "Generating plots from all result files..."
python LoopLLM/degeneration_test/plot_results.py \
    --results_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo -e "\nEvaluation and plotting complete. Results are in '$OUTPUT_DIR'." 