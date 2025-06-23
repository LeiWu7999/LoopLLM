import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np

# This dictionary will be used by hooks to store hidden states.
hidden_states_storage = {}

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def get_h_in_hook(layer_name):
    """Factory for pre-forward hooks to capture layer input."""
    def hook(model, input_tuple):
        hidden_states_storage[f'{layer_name}_in'] = input_tuple[0].detach().cpu()
    return hook

def get_h_out_hook(layer_name):
    """Factory for forward hooks to capture layer output."""
    def hook(model, input_tuple, output_tuple):
        hidden_states_storage[f'{layer_name}_out'] = output_tuple[0].detach().cpu()
    return hook

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
    parser = argparse.ArgumentParser(description="Evaluate baseline Llama model degeneration.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the baseline Llama model.")
    parser.add_argument("--output_dir", type=str, default="./degeneration_results", help="Directory to save results.")
    parser.add_argument("--start_layer", type=int, default=6, help="Start layer index of the block.")
    parser.add_argument("--end_layer", type=int, default=8, help="End layer index of the block.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length for evaluation data.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the evaluation on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model and Tokenizer
    print(f"Loading baseline model from: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare sample data (can be replaced with a real dataset)
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

    # Attach hooks
    start_layer = model.model.layers[args.start_layer]
    end_layer = model.model.layers[args.end_layer]
    hook_handle_in = start_layer.register_forward_pre_hook(get_h_in_hook('block'))
    hook_handle_out = end_layer.register_forward_hook(get_h_out_hook('block'))
    
    print(f"Attached hooks to layers {args.start_layer} and {args.end_layer}.")

    global hidden_states_storage
    hidden_states_storage = {}

    with torch.no_grad():
        model(**inputs)

    h_in = hidden_states_storage.get('block_in')
    h_out = hidden_states_storage.get('block_out')

    if h_in is not None and h_out is not None:
        print("Calculating metrics...")
        cos_sim, mse, norm_diff = calculate_metrics(h_in, h_out)
        
        results = [{
            "checkpoint": "baseline",
            "loop_count": 0,  # Baseline is equivalent to 0 loops
            "cosine_similarity": cos_sim,
            "mse": mse,
            "norm_diff": norm_diff,
        }]
        
        results_df = pd.DataFrame(results)
        output_file = Path(args.output_dir) / "baseline_metrics.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Baseline results saved to {output_file}")
        
    else:
        print("Error: Could not retrieve hidden states from hooks.")

    hook_handle_in.remove()
    hook_handle_out.remove()

if __name__ == "__main__":
    main() 