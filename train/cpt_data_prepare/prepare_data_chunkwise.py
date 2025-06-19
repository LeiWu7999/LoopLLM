import os
import json
import warnings
import tempfile
import huggingface_hub
import random
import fnmatch
import math
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
DATA_CACHE_DIR = "./datasets_cache_chunkwise"
OUTPUT_DIR = "./output_data_chunkwise"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_WORKERS = 64 # You can adjust this based on your machine's performance

# Define the target size for SlimPajama
SLIMPAJAMA_TOTAL_TOKENS_APPROX = 627 * 1_000_000_000
TARGET_SLIMPAJAMA_TOKENS_APPROX = 20 * 1_000_000_000

# --- Main Logic ---

def prepare_finemath(tmp_dir):
    """Downloads and processes the FineMath dataset first."""
    print("--- 1. Processing FineMath Dataset ---")
    
    finemath_ds = load_dataset(
        "HuggingFaceTB/finemath", "finemath-4plus", 
        split='train', cache_dir=DATA_CACHE_DIR
    )
    
    output_path = tmp_dir / "FineMath.jsonl" # Use consistent naming
    with open(output_path, "w") as f:
        for example in tqdm(finemath_ds, desc="FineMath"):
            text = example.get("text")
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            
    print(f"FineMath processing complete. Saved to temporary file.")

def download_and_process_chunk(chunk_file, repo_id):
    """
    Downloads a single chunk, extracts text, and returns it.
    This function is designed to be run in a separate thread.
    """
    processed_texts = []
    try:
        with tempfile.TemporaryDirectory() as dl_tmp:
            local_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=chunk_file,
                repo_type="dataset",
                cache_dir=dl_tmp,
                local_dir=dl_tmp,
                local_dir_use_symlinks=False,
            )
            
            # Setting cache_dir to the temp download dir to avoid global cache conflicts
            chunk_ds = load_dataset("json", data_files=local_path, split="train", cache_dir=os.path.join(dl_tmp, "datasets_cache"))
            
            for example in chunk_ds:
                text = example.get("text")
                if text:
                    processed_texts.append(text)
        return processed_texts
    except Exception as e:
        warnings.warn(f"Failed to process chunk {chunk_file}: {e}")
        return []

def prepare_slimpajama_chunkwise(tmp_dir):
    """Downloads and processes a sample of SlimPajama chunks in parallel."""
    print("\n--- 2. Processing SlimPajama Dataset (Sampled) ---")
    
    repo_id = "cerebras/SlimPajama-627B"
    
    print("Fetching file list for SlimPajama...")
    all_repo_files = huggingface_hub.list_repo_files(repo_id, repo_type="dataset")
    glob_pattern = "train/*/*.jsonl.zst"
    all_files = [f for f in all_repo_files if fnmatch.fnmatch(f, glob_pattern)]
    print(f"Found {len(all_files)} total chunks.")

    # Calculate number of files to download based on the token ratio
    sampling_ratio = TARGET_SLIMPAJAMA_TOKENS_APPROX / SLIMPAJAMA_TOTAL_TOKENS_APPROX
    num_files_to_download = math.ceil(len(all_files) * sampling_ratio)
    print(f"Sampling ratio: ~{sampling_ratio:.4f}. Will download {num_files_to_download} of {len(all_files)} files.")

    # Randomly sample the list of files to download
    files_to_download = random.sample(all_files, num_files_to_download)

    output_path = tmp_dir / "SlimPajama.jsonl"
    with open(output_path, "w") as f_out:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(download_and_process_chunk, chunk, repo_id) for chunk in files_to_download}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks"):
                chunk_texts = future.result()
                if not chunk_texts:
                    continue

                for text in chunk_texts:
                    f_out.write(json.dumps({"text": text}) + "\n")

    print("\nSlimPajama processing complete.")

def combine_and_finalize(tmp_dir, output_dir):
    """Combines all temporary files into the final train/validation splits with shuffling."""
    print("\n--- 3. Combining and Finalizing Dataset ---")
    
    VALIDATION_SPLIT_RATIO = 1000
    train_file_path = os.path.join(output_dir, "train.jsonl")
    valid_file_path = os.path.join(output_dir, "validation.jsonl")
    
    print("Reading all temporary files into memory for shuffling...")
    all_lines = []
    for temp_file in tqdm(list(tmp_dir.glob("*.jsonl")), desc="Reading temp files"):
        with open(temp_file, "r") as f_in:
            all_lines.extend(f_in.readlines())
            
    print(f"Shuffling {len(all_lines):,} examples...")
    random.shuffle(all_lines)
    
    print("Writing to final train/validation files...")
    with open(train_file_path, "w") as f_train, open(valid_file_path, "w") as f_valid:
        for i, line in enumerate(tqdm(all_lines, desc="Finalizing")):
            if (i + 1) % VALIDATION_SPLIT_RATIO == 0:
                f_valid.write(line)
            else:
                f_train.write(line)
                        
    print("\n--- Dataset preparation complete! ---")
    print(f"Final dataset saved to '{output_dir}'")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        prepare_finemath(tmp_path)
        prepare_slimpajama_chunkwise(tmp_path)
        combine_and_finalize(tmp_path, OUTPUT_DIR)

        print("\nDataset preparation finished.") 