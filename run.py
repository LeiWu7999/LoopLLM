import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import random
import numpy as np
from hydra.core.hydra_config import HydraConfig

# Assuming LoopLlamaForCausalLM and LoopLlamaConfig are in the same directory or properly installed
# If run.py is in LoopLLM directory, and other .py files are also in LoopLLM:
from loop_llama_model import LoopLlamaForCausalLM
from loop_llama_config import LoopLlamaConfig

def calculate_ppl_loss(model, tokenizer, encodings, device, window_size, stride):
    """
    Calculates Perplexity (PPL) and Negative Log-Likelihood (NLL) for a given model
    using a sliding window approach, based on the user's provided logic.
    """
    model.eval()
    seq_len = encodings.input_ids.size(1)
    
    nll_sum = 0.0
    total_loss_tokens = 0
    
    prev_end_loc = 0
    # Ensure input_ids is on CPU for slicing, then move chunk to device
    # Assuming batch_size=1 for encodings initially from tokenizer(full_text, ...)
    input_ids_cpu = encodings.input_ids.squeeze(0) 

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL/NLL"):
        end_loc = min(begin_loc + window_size, seq_len)
        
        current_input_ids_chunk = input_ids_cpu[begin_loc:end_loc]
        if current_input_ids_chunk.numel() == 0: # Skip empty chunks
            if end_loc == seq_len: break
            continue

        current_input_ids = current_input_ids_chunk.unsqueeze(0).to(device)
        
        # trg_len is the length of the new segment in the current window, relative to prev_end_loc
        trg_len = end_loc - prev_end_loc
        
        if trg_len <= 0:
            # This condition means the current window does not extend beyond prev_end_loc.
            # This can happen if stride causes begin_loc to advance such that
            # begin_loc + window_size <= prev_end_loc, or if we've reached the end.
            # We only want to calculate loss on new segments.
            if end_loc == seq_len: # Processed up to the end.
                break
            # If not at the end, but trg_len is not positive, it means this chunk offers no new tokens
            # for evaluation under the current prev_end_loc.
            # This implies an overlap situation where the "new" part is zero or negative.
            # We must ensure prev_end_loc is updated to allow progress.
            # The user's original logic: prev_end_loc = end_loc. This update happens *after* processing.
            # So if trg_len <=0 here, it means prev_end_loc might be ahead or equal to end_loc,
            # which suggests the loop or prev_end_loc update might need care.
            # However, with `prev_end_loc = end_loc` at the end of the loop, `trg_len` should mostly be `stride` or `window_size` (first pass).
            # For safety, if no new tokens, just ensure prev_end_loc advances if possible and continue.
            if end_loc > prev_end_loc : # Possible if window is very small or stride is tricky
                 prev_end_loc = end_loc
            continue

        target_labels = current_input_ids.clone()
        current_chunk_len = current_input_ids.size(1)
        
        # Mask tokens that are NOT part of the new segment (trg_len).
        # The number of tokens to mask at the start of the current_chunk is:
        # current_chunk_len - trg_len
        num_to_mask_at_start = current_chunk_len - trg_len
        if num_to_mask_at_start < 0: 
            num_to_mask_at_start = 0 # Should ideally not happen with correct trg_len
        
        target_labels[:, :num_to_mask_at_start] = -100

        # If all tokens in target_labels are masked, no loss can be computed.
        if (target_labels == -100).all().item():
            if end_loc == seq_len:
                 break
            prev_end_loc = end_loc # Update prev_end_loc to move forward
            continue
            
        with torch.no_grad():
            outputs = model(current_input_ids, labels=target_labels)
            neg_log_likelihood = outputs.loss # This is mean NLL

        if neg_log_likelihood is not None:
            batch_size = target_labels.size(0) # Should be 1
            num_valid_target_tokens = (target_labels != -100).sum().item()
            
            num_loss_calc_tokens = 0
            if num_valid_target_tokens > batch_size : 
                num_loss_calc_tokens = num_valid_target_tokens - batch_size
            
            if num_loss_calc_tokens > 0 :
                nll_sum += neg_log_likelihood.item() * num_loss_calc_tokens
                total_loss_tokens += num_loss_calc_tokens
        
        prev_end_loc = end_loc # Crucial update for the next iteration
        if end_loc == seq_len:
            break
            
    if total_loss_tokens == 0:
        print("\nWarning: No tokens were processed for PPL/NLL calculation (total_loss_tokens is 0).")
        print(f"seq_len: {seq_len}, window_size: {window_size}, stride: {stride}")
        print("This might be due to very short sequence, or window/stride settings.")
        return float('inf'), float('inf')

    avg_nll = nll_sum / total_loss_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item() # Ensure input to exp is tensor
    return ppl, avg_nll

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Ensure matplotlib uses a non-interactive backend if running in a headless environment
    matplotlib.use('Agg') 

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # --- Set Random Seed --- 
    if cfg.get("random_seed") is not None:
        seed = int(cfg.random_seed)
        print(f"Setting random seed to: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For potentially more deterministic behavior on CUDA (at a performance cost)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    # --- Determine loop layers string for subfolder naming (once per run) ---
    py_loop_layers_idx_for_naming = []
    # Use .get to safely access loop_layers_idx, which might not be present
    raw_loop_layers_idx = cfg.model.get('loop_layers_idx') 
    # Adjusted isinstance check for wider omegaconf version compatibility
    if raw_loop_layers_idx is not None:
        # If it's not a Python list, try to convert it.
        # This handles OmegaConf's list-like structures.
        if not isinstance(raw_loop_layers_idx, list):
            try:
                current_list_candidate = list(raw_loop_layers_idx)
                if isinstance(current_list_candidate, list): # Check if conversion was successful
                     py_loop_layers_idx_for_naming = current_list_candidate
                else: # Conversion didn't result in a list, treat as not a valid list for this purpose
                    print(f"Warning: loop_layers_idx was present but not a list or convertible to list: {raw_loop_layers_idx}")
            except TypeError: # Not iterable or other issue during list() conversion
                print(f"Warning: loop_layers_idx was present but could not be converted to a list: {raw_loop_layers_idx}")
        else: # It's already a Python list
            py_loop_layers_idx_for_naming = raw_loop_layers_idx
    
    loop_layers_descriptive_str = "layers_unspecified" # Default if key missing or not a list
    if len(py_loop_layers_idx_for_naming) >= 1:
        s_idx = min(py_loop_layers_idx_for_naming)
        e_idx = max(py_loop_layers_idx_for_naming)
        if s_idx <= e_idx:
            loop_layers_descriptive_str = f"L{s_idx}-L{e_idx}"
        else: # Should ideally not happen with min/max logic
            loop_layers_descriptive_str = "layers_config_error"
    elif raw_loop_layers_idx is None: # Key 'loop_layers_idx' exists and is explicitly None
        loop_layers_descriptive_str = "layers_explicitly_none"
    # If raw_loop_layers_idx was not found by .get(), it remains "layers_unspecified"
    
    # The output directory is managed by Hydra.
    # The `run_loop_test.sh` script sets `hydra.run.dir` to a specific path for each run.
    # We get this path from Hydra's config API to be robust against `hydra.job.chdir` settings.
    final_results_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(final_results_dir, exist_ok=True) # Ensure it exists.
    print(f"Results for this configuration will be saved in: {final_results_dir}")

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() and cfg.gpu_id is not None else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer for base model: {cfg.model.base_model_name_or_path}")
    # Add trust_remote_code=True if model is custom/not fully in HF standard
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")


    # 2. Load Dataset
    print(f"Loading dataset: {cfg.dataset.name} ({cfg.dataset.subset}), split: {cfg.dataset.split}")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.subset, split=cfg.dataset.split)
    text_column = cfg.dataset.text_column
    
    # Concatenate all text from the dataset
    # Ensure this step doesn't consume too much memory for very large datasets. Wikitext-2 is fine.
    full_text = "\n\n".join(d[text_column] for d in dataset if d[text_column]) # Filter out None/empty
    if not full_text:
        print("Error: Dataset is empty or text column yielded no text.")
        return

    print(f"Tokenizing dataset...")
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    print(f"Dataset tokenized. Total tokens: {encodings.input_ids.size(1)}")

    results_data = []

    # --- Overall Progress Bar --- 
    total_evaluation_steps = 1 + len(cfg.evaluation.loop_n_times_list)
    with tqdm(total=total_evaluation_steps, desc="Overall Experiment Progress") as pbar:

        # 3.a. Baseline LlamaForCausalLM (Original non-loop model)
        pbar.set_description(f"Baseline Eval: {os.path.basename(str(cfg.model.base_model_name_or_path))}")
        print(f"\n--- Evaluating Baseline Model: {cfg.model.base_model_name_or_path} ---")
        try:
            baseline_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model_name_or_path,
                trust_remote_code=True # Important for some models
            )
            baseline_model.to(device)
            
            baseline_ppl, baseline_nll = calculate_ppl_loss(
                baseline_model, 
                tokenizer, 
                encodings, 
                device, 
                min(cfg.evaluation.window_size, baseline_model.config.max_position_embeddings), # Use model's actual max length if smaller
                cfg.evaluation.stride
            )
            print(f"Baseline PPL: {baseline_ppl:.4f}, Baseline NLL: {baseline_nll:.4f}")
            results_data.append({"loop_count": "baseline", "ppl": baseline_ppl, "nll": baseline_nll, "loop_layers_config": loop_layers_descriptive_str})
            
            del baseline_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during baseline evaluation: {e}")
            results_data.append({"loop_count": "baseline", "ppl": float('inf'), "nll": float('inf'), "loop_layers_config": loop_layers_descriptive_str})
        pbar.update(1)

        # 3.b. LoopLlamaForCausalLM for different loop_counts
        for loop_n_times_val in cfg.evaluation.loop_n_times_list:
            pbar.set_description(f"LoopLlama Eval (Loops: {loop_n_times_val}, Layers: {loop_layers_descriptive_str})")
            print(f"\n--- Evaluating LoopLlama with loop_n_times = {loop_n_times_val} ---")
            try:
                # Determine loop_layers tuple from cfg.model.loop_layers_idx
                final_loop_layers_tuple = None
                # Adjusted check for cfg.model.loop_layers_idx
                loop_layers_idx_from_cfg = cfg.model.get('loop_layers_idx')
                
                py_loop_layers_idx = []
                if loop_layers_idx_from_cfg is not None:
                    if not isinstance(loop_layers_idx_from_cfg, list):
                        try:
                            candidate_list = list(loop_layers_idx_from_cfg)
                            if isinstance(candidate_list, list):
                                py_loop_layers_idx = candidate_list
                            else:
                                print(f"Warning: model.loop_layers_idx was present but not a list or convertible: {loop_layers_idx_from_cfg}")
                        except TypeError:
                             print(f"Warning: model.loop_layers_idx could not be converted to list: {loop_layers_idx_from_cfg}")
                    else: # It's already a Python list
                        py_loop_layers_idx = loop_layers_idx_from_cfg

                if py_loop_layers_idx and len(py_loop_layers_idx) >= 1:
                    start_idx = min(py_loop_layers_idx)
                    end_idx = max(py_loop_layers_idx)
                    
                    if start_idx <= end_idx: # Basic validation
                        final_loop_layers_tuple = (start_idx, end_idx)
                    else:
                        print(f"警告: 配置中的 model.loop_layers_idx {py_loop_layers_idx} 导致无效的层区间 ({start_idx}, {end_idx})。loop_layers 将保持为 None。")

                # Initialize LoopLlamaConfig, passing parameters directly if possible.
                # from_pretrained for configs passes **kwargs to the config's __init__.
                loop_config = LoopLlamaConfig.from_pretrained(
                    cfg.model.base_model_name_or_path,
                    loop_layers=final_loop_layers_tuple,  # Pass interpreted loop_layers
                    loop_count=loop_n_times_val,          # Pass loop_count (from loop_n_times_val)
                    loop_strategy="fixed_count",
                    kv_cache_mode="virtual_layers",
                    virtual_layer_count=loop_n_times_val,
                    min_loop_count=loop_n_times_val,
                    virtual_attention_mode="serial",
                    trust_remote_code=True
                )
                
                print(f"Initializing LoopLlamaForCausalLM from {cfg.model.base_model_name_or_path}")
                print(f"LoopLlamaConfig effective: loop_layers={loop_config.loop_layers}, loop_count={loop_config.loop_count}, loop_strategy='{loop_config.loop_strategy}'")
                
                model = LoopLlamaForCausalLM.from_pretrained(
                    cfg.model.base_model_name_or_path,
                    config=loop_config,
                    trust_remote_code=True 
                )
                model.to(device)

                actual_window_size = min(cfg.evaluation.window_size, model.config.max_position_embeddings)
                if hasattr(model.config, 'n_positions'): # Some configs use n_positions
                     actual_window_size = min(cfg.evaluation.window_size, model.config.n_positions)
                
                print(f"Using window_size: {actual_window_size} (Configured: {cfg.evaluation.window_size}, Model max: {model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else model.config.n_positions})")

                ppl, nll = calculate_ppl_loss(
                    model, 
                    tokenizer, 
                    encodings, 
                    device, 
                    actual_window_size,
                    cfg.evaluation.stride
                )
                print(f"Loop Count {loop_n_times_val} - PPL: {ppl:.4f}, NLL: {nll:.4f}")
                results_data.append({"loop_count": loop_n_times_val, "ppl": ppl, "nll": nll, "loop_layers_config": loop_layers_descriptive_str})
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during LoopLlama evaluation (loop_n_times={loop_n_times_val}): {e}")
                results_data.append({"loop_count": loop_n_times_val, "ppl": float('inf'), "nll": float('inf'), "loop_layers_config": loop_layers_descriptive_str})
            pbar.update(1)

    # 4. Save Results and Plot
    if not results_data:
        print("No results were generated. Skipping saving and plotting.")
        return

    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(final_results_dir, "results.csv") # Save in new subdir
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")
    print(results_df)

    # Plotting (only if there are numeric results beyond baseline)
    numeric_results_df = results_df[results_df['loop_count'] != 'baseline'].copy()
    if not numeric_results_df.empty:
        numeric_results_df['loop_count'] = pd.to_numeric(numeric_results_df['loop_count'])
        numeric_results_df = numeric_results_df.sort_values(by='loop_count')

        baseline_entry = results_df[results_df['loop_count'] == 'baseline']
        
        plt.figure(figsize=(14, 6))

        # PPL Plot
        plt.subplot(1, 2, 1)
        plt.plot(numeric_results_df['loop_count'], numeric_results_df['ppl'], marker='o', linestyle='-', label='LoopLlama PPL')
        if not baseline_entry.empty and pd.notna(baseline_entry['ppl'].iloc[0]):
            plt.axhline(y=baseline_entry['ppl'].iloc[0], color='r', linestyle='--', label=f"baseline PPL ({baseline_entry['ppl'].iloc[0]:.2f})")
        plt.xlabel("Loop Count")
        plt.ylabel("PPL")
        plt.title("PPL vs. Loop Count")
        plt.legend()
        plt.grid(True)

        # NLL Plot
        plt.subplot(1, 2, 2)
        plt.plot(numeric_results_df['loop_count'], numeric_results_df['nll'], marker='s', linestyle='-', label='LoopLlama NLL')
        if not baseline_entry.empty and pd.notna(baseline_entry['nll'].iloc[0]):
            plt.axhline(y=baseline_entry['nll'].iloc[0], color='g', linestyle='--', label=f"baseline NLL ({baseline_entry['nll'].iloc[0]:.2f})")
        plt.xlabel("Loop Count")
        plt.ylabel("Avg. NLL")
        plt.title("NLL vs. Loop Count")
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f"LoopLlama Performance Evaluation ({loop_layers_descriptive_str})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        
        plot_path = os.path.join(final_results_dir, "ppl_nll_vs_loop_count.png") # Save in new subdir
        plt.savefig(plot_path)
        print(f"Plots saved to {plot_path}")
    else:
        print("No numeric results for LoopLlama to plot.")

if __name__ == "__main__":
    main()
