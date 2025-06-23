import torch
import torch.nn.functional as F
import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import random
import numpy as np
from transformers import AutoTokenizer

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from loop_llama_model import LoopLlamaForCausalLM
from loop_llama_config import LoopLlamaConfig


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def calculate_metrics(h_in, h_out):
    """Calculates Cosine Similarity, MSE, and Norm Difference."""
    h_in, h_out = h_in.float(), h_out.float()
    h_in_flat = h_in.view(-1, h_in.shape[-1])
    h_out_flat = h_out.view(-1, h_out.shape[-1])

    cos_sim = F.cosine_similarity(h_in_flat, h_out_flat, dim=-1).mean().item()
    mse = F.mse_loss(h_in_flat, h_out_flat).item()
    norm_diff = (torch.linalg.norm(h_out_flat, dim=-1) - torch.linalg.norm(h_in_flat, dim=-1)).mean().item()
    
    return cos_sim, mse, norm_diff

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single LoopLlama checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the LoopLlama checkpoint directory.")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.2-1B", help="Path to the base model for tokenizer.")
    parser.add_argument("--output_dir", type=str, default="./degeneration_results", help="Directory to save results.")
    parser.add_argument("--loop_start_layer", type=int, default=6, help="Start layer index of the loop block.")
    parser.add_argument("--loop_end_layer", type=int, default=8, help="End layer index of the loop block.")
    parser.add_argument("--loop_counts", type=str, default="0,1,2,3,4,5", help="Comma-separated list of loop counts to test.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length for evaluation data.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the evaluation on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_name = Path(args.ckpt_path).name
    print(f"Using device: {device} for checkpoint: {ckpt_name}")

    # Load Model and Tokenizer
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    config = LoopLlamaConfig.from_pretrained(
        args.ckpt_path,
        loop_layers=[[args.loop_start_layer, args.loop_end_layer]],  # Pass interpreted loop_layers
        loop_count=[args.loop_counts],          # Pass loop_count (from loop_n_times_val)
        loop_strategy="fixed_count",
        kv_cache_mode="merge_strategy",
        virtual_layer_count=[args.loop_counts],
        min_loop_count=[args.loop_counts],
        virtual_attention_mode="serial",
        trust_remote_code=True
    )
    
    model = LoopLlamaForCausalLM.from_pretrained(args.ckpt_path, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare sample data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming many industries.",
        "Large language models require significant computational resources.",
        "PyTorch provides a flexible framework for deep learning research."
    ] * (args.batch_size // 4 + 1)
    sample_texts = sample_texts[:args.batch_size]
    
    inputs = tokenizer(
        sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.seq_length
    ).to(device)

    all_results = []
    loop_counts_to_test = [int(c) for c in args.loop_counts.split(',')]

    for loop_count in tqdm(loop_counts_to_test, desc=f"Evaluating {ckpt_name}"):
        with torch.no_grad():
            # Run forward pass, requesting hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True,
                loop_count=loop_count
            )
        
        # In Hugging Face, `outputs.hidden_states` is a tuple where
        # hidden_states[0] = input embeddings
        # hidden_states[i] = output of layer i-1 (i.e., input to layer i)
        # So, input to layer 6 is always at index 6.
        h_in = outputs.hidden_states[args.loop_start_layer].detach().cpu()
        
        # The output of the final loop block is the input to the next layer.
        # The index of this state depends on how many loops were executed, as each
        # loop adds `num_layers_in_block` states to the list.
        num_layers_in_block = args.loop_end_layer - args.loop_start_layer + 1
        
        # The input to the first layer of the block is at `loop_start_layer`.
        # After that, `loop_count * num_layers_in_block` states are added.
        # The state we want is the one right after all these.
        h_out_index = args.loop_start_layer + (loop_count * num_layers_in_block)
        
        h_out = outputs.hidden_states[h_out_index].detach().cpu()
        
        cos_sim, mse, norm_diff = calculate_metrics(h_in, h_out)
        all_results.append({
            "checkpoint": ckpt_name,
            "loop_count": loop_count,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "norm_diff": norm_diff,
        })

    # Save results to a file specific to this checkpoint
    results_df = pd.DataFrame(all_results)
    output_file = Path(args.output_dir) / f"{ckpt_name}_metrics.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results for {ckpt_name} saved to {output_file}")

if __name__ == "__main__":
    main() 