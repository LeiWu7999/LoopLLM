import os
import json
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import tempfile
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# We use Mistral's tokenizer as a high-quality, open-access alternative to Llama's.
# You can easily swap this to your model of choice, e.g., "meta-llama/Llama-2-7b-hf",
# but ensure you have access to it.
TOKENIZER_NAME = "meta-llama/Llama-3.1-1B"
# The sequence length for the model.
SEQ_LENGTH = 8192
# Number of CPU workers for parallel processing. Use half of the available CPUs by default.
NUM_WORKERS = max(1, os.cpu_count() // 2)
# Number of lines to process in each chunk. Adjust based on memory.
CHUNK_SIZE = 10_000
# Input files from the previous step.
INPUT_DIR = "./output_data_chunkwise"
# Where to save the processed dataset.
OUTPUT_DIR = "./packed_data"

INPUT_FILES = {
    "train": os.path.join(INPUT_DIR, "train.jsonl"),
    "validation": os.path.join(INPUT_DIR, "validation.jsonl")
}

# --- Parallel Processing Functions ---

def read_in_chunks(file_path, chunk_size):
    """Generator to read a file and yield chunks of lines."""
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def tokenize_chunk(lines_chunk, tokenizer_name):
    """
    Worker function to tokenize a list of text lines.
    This function is executed in a separate process.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_docs = []
    for line in lines_chunk:
        try:
            raw_doc = json.loads(line)
            text = raw_doc.get("text")
            if text:
                tokenized_doc = tokenizer(text, add_special_tokens=True).input_ids
                tokenized_docs.append(tokenized_doc)
        except (json.JSONDecodeError, AttributeError):
            # E.g. if a line is malformed or text is not a string
            continue
    return tokenized_docs

def pack_and_write_streaming(tokenized_docs_iterator, seq_length, output_file_handle):
    """
    Packs tokenized documents from an iterator and writes them to a file handle
    in a streaming fashion to avoid high memory usage.
    """
    buffer = []
    doc_id_buffer = []
    doc_id_counter = 0
    packed_sequence_count = 0

    for tokenized_doc in tokenized_docs_iterator:
        # Can the new document fit entirely in the current buffer?
        if len(buffer) + len(tokenized_doc) <= seq_length:
            buffer.extend(tokenized_doc)
            doc_id_buffer.extend([doc_id_counter] * len(tokenized_doc))
            doc_id_counter += 1
            continue

        # If not, the buffer must be filled and flushed.
        space_left = seq_length - len(buffer)

        # Fill the remaining space with the beginning of the new document.
        if space_left > 0:
            buffer.extend(tokenized_doc[:space_left])
            doc_id_buffer.extend([doc_id_counter] * space_left)

        # The buffer is now full. Write it to the temp file as a JSON line.
        output_file_handle.write(json.dumps({
            "input_ids": buffer,
            "document_ids": doc_id_buffer
        }) + "\n")
        packed_sequence_count += 1
        
        # The rest of the new document is discarded.
        buffer = []
        doc_id_buffer = []
        doc_id_counter += 1

    logging.info(f"Streaming pack complete. Wrote {packed_sequence_count:,} packed sequences.")
    logging.info(f"Discarding {len(buffer):,} tokens from the final incomplete sequence.")
    return packed_sequence_count


if __name__ == '__main__':
    logging.info("--- Starting Tokenization and Packing Process ---")
    
    # --- 1. Setup Directories ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    # --- 2. Log Tokenizer Info (without loading it in the main process repeatedly) ---
    logging.info(f"Using tokenizer: {TOKENIZER_NAME}")
    logging.info(f"Using {NUM_WORKERS} worker processes.")
    
    # --- 3. Process and Save Files ---
    for split, file_path in INPUT_FILES.items():
        if not os.path.exists(file_path):
            logging.warning(f"Input file not found at {file_path}. Skipping '{split}' split.")
            continue

        # Count total lines for progress bar
        logging.info(f"Counting lines in {file_path} for '{split}' split...")
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        total_chunks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        line_chunks_generator = read_in_chunks(file_path, CHUNK_SIZE)

        # Use a temporary file to store intermediate packed results
        with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl", encoding="utf-8") as tmp_file:
            logging.info(f"Using temporary file for packing: {tmp_file.name}")

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                logging.info(f"Starting parallel tokenization for '{split}' split...")
                
                worker_func = partial(tokenize_chunk, tokenizer_name=TOKENIZER_NAME)
                
                tokenized_docs_lists = executor.map(worker_func, line_chunks_generator)
                
                progress_bar = tqdm(tokenized_docs_lists, total=total_chunks, desc=f"Tokenizing {split}")

                def flatten_iterator(iterator_of_lists):
                    for list_of_items in iterator_of_lists:
                        for item in list_of_items:
                            yield item

                # Stream packed data to the temporary file
                pack_and_write_streaming(flatten_iterator(progress_bar), SEQ_LENGTH, tmp_file)
            
            # --- Load from temp file and save to final destination ---
            logging.info(f"Loading packed data from temporary file '{tmp_file.name}'...")
            
            # Define the features for the dataset
            features = Features({
                'input_ids': [Value(dtype='int32')],
                'document_ids': [Value(dtype='int32')]
            })
            
            # Use from_json which is memory-efficient for large files
            packed_dataset = Dataset.from_json(tmp_file.name, features=features)
            
            save_path = os.path.join(OUTPUT_DIR, split)
            logging.info(f"Saving '{split}' dataset to {save_path} in Arrow format...")
            packed_dataset.save_to_disk(save_path)
            logging.info(f"'{split}' dataset saved successfully.")

    logging.info("--- All files processed and saved. ---") 