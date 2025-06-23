# Model Degeneration Analysis Toolkit for LoopLlama

This toolkit is designed to analyze whether the looped layers in a `LoopLlama` model degenerate into an identity mapping after continuous pre-training. It follows a modular, multi-GPU approach where each model checkpoint is evaluated in a separate process.

## Toolkit Components

1.  **`run_baseline_eval.py`**: Evaluates the original, non-looped Llama model to establish a baseline. It uses hooks to capture hidden states.
2.  **`run_single_ckpt_eval.py`**: Evaluates a single `LoopLlama` checkpoint. It uses the `output_hidden_states` flag to get intermediate layers without hooks.
3.  **`run_all_evals.sh`**: The main orchestration script. It launches the baseline and checkpoint evaluation scripts in parallel across multiple GPUs, redirecting their output to log files.
4.  **`plot_results.py`**: Gathers all the individual result CSVs, merges them, and generates final plots for analysis.

## How It Works

1.  **Execution (`run_all_evals.sh`)**:
    *   You configure paths and parameters at the top of the shell script.
    *   The script creates a `logs` subdirectory within your output directory.
    *   It first launches `run_baseline_eval.py` on GPU 0, redirecting its console output to a log file (e.g., `logs/baseline_eval.log`).
    *   It then iterates through your specified checkpoint steps (e.g., 100, 200, ...), launching an instance of `run_single_ckpt_eval.py` for each on a new GPU (GPU 1, GPU 2, ...), with output also redirected to a corresponding log file (e.g., `logs/checkpoint-100_eval.log`).
    *   All evaluation processes run in the background. The script waits for all of them to complete.
2.  **Evaluation (Python Scripts)**:
    *   Each script loads its assigned model and a tokenizer.
    *   It runs a forward pass on sample data.
    *   It captures `h_in` (hidden state before layer 6) and `h_out` (hidden state after layer 8 for various loop counts).
    *   It computes three metrics: Cosine Similarity, Mean Squared Error (MSE), and Norm Difference.
    *   Each script saves its results to a unique CSV file in the output directory (e.g., `baseline_metrics.csv`, `checkpoint-100_metrics.csv`).
3.  **Aggregation and Visualization (`plot_results.py`)**:
    *   After all evaluations are done, the shell script calls `plot_results.py`.
    *   This script finds and merges all `*_metrics.csv` files.
    *   It then generates a series of plots comparing all checkpoints across the different loop counts.

## How to Use

### Prerequisites

-   Ensure you have the necessary packages installed: `torch`, `transformers`, `pandas`, `matplotlib`, `seaborn`.
-   You have access to multiple CUDA-enabled GPUs. The number of GPUs should ideally be at least `1 + (number of checkpoints)`.

### Step-by-Step Guide

1.  **Configure the Main Script**:

    Open `LoopLLM/degeneration_test/run_all_evals.sh` and edit the configuration section at the top:
    ```sh
    # Path to the original, un-trained Llama-3.2-1B model
    BASELINE_MODEL_PATH="/path/to/your/un-trained/Llama-3.2-1B" 

    # Directory containing your trained checkpoints (e.g., ckpts/checkpoint-100)
    CKPT_DIR="./ckpts"
    
    # Comma-separated list of checkpoint steps to evaluate
    CKPT_STEPS="100,200,300,400,500"
    ```
    **Important**: You must provide a valid path to your baseline model.

2.  **Make the Script Executable**:

    In your terminal, run:
    ```bash
    chmod +x LoopLLM/degeneration_test/run_all_evals.sh
    ```

3.  **Run the Evaluation**:

    Execute the script from the root of your project (`loop_llm_exp`):
    ```bash
    ./LoopLLM/degeneration_test/run_all_evals.sh
    ```
    The script will show the progress of launching each evaluation and then wait for them to finish.

4.  **View the Results and Logs**:

    Once the script completes, the `degeneration_results` directory (or your configured output directory) will contain:
    *   A `_metrics.csv` file for each evaluated checkpoint.
    *   A `logs/` subdirectory containing detailed output from each process, useful for debugging.
    *   The final analysis plots:
        *   `cosine_similarity_vs_loop_count.png`
        *   `mse_vs_loop_count.png`
        *   `norm_diff_vs_loop_count.png`

## Interpreting the Results

-   **Ideal (No Degeneration)**: On the plots, a healthy trained model should show lines that deviate from the "0 loop" case as the loop count increases. This means Cosine Similarity should decrease while MSE and Norm Difference increase, indicating that the layers are performing meaningful work with each pass.
-   **Sign of Degeneration**: If a highly-trained checkpoint (e.g., `checkpoint-500`) has lines that stay very flat and close to the "0 loop" values (Cosine Sim ≈ 1, MSE ≈ 0), it suggests the model has learned to make the looped block act like an identity function to avoid instability, which is a sign of degeneration. 