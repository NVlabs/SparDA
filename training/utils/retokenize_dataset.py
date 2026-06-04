# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from streaming import MDSWriter, StreamingDataset, Stream
from tqdm import tqdm
import signal
import sys
import shutil
import json

# Ensure clean exit of children if main process is killed
def signal_handler(sig, frame):
    print("\nCtrl+C pressed, exiting...")
    # Kill the entire process group
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Disable tokenizer internal parallelism to avoid deadlocks in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define SafeStream to avoid lock issues during read
class SafeStream(Stream):
    """Safe if multiple processes try to decompress the same shard."""
    def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
        unique_extension = "." + str(os.getenv("SPARDA_RUN_ID", "local")) + "-" + str(os.getpid())
        super()._decompress_shard_part(zip_info, zip_filename, raw_filename + unique_extension, compression)
        os.rename(raw_filename + unique_extension, raw_filename)

class RetokenizeCollator:
    def __init__(self, src_model_id, tgt_model_id):
        self.src_model_id = src_model_id
        self.tgt_model_id = tgt_model_id
        # Tokenizers are initialized lazily in the worker process
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def __call__(self, batch):
        # Initialize tokenizers once per worker process
        if self.src_tokenizer is None:
            self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_id, trust_remote_code=True)
            self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_id, trust_remote_code=True)

        # Extract input_ids from batch (list of dicts)
        src_input_ids = [torch.from_numpy(item['input_ids'].copy()).long() for item in batch]
        
        # 1. Decode back to text (strip special tokens like BOS/EOS/PAD from source)
        texts = self.src_tokenizer.batch_decode(src_input_ids, skip_special_tokens=True)
        
        # 2. Encode with new tokenizer
        # We do NOT add special tokens automatically to packed chunks to avoid corruption
        tgt_encodings = self.tgt_tokenizer(texts, add_special_tokens=False)
        
        results = []
        for new_ids in tgt_encodings['input_ids']:
            # Skip empty sequences
            if len(new_ids) == 0:
                continue
            results.append(np.array(new_ids, dtype=np.int32))
            
        return results

def get_dataset_sample_count(data_dir):
    """Get total sample count from MDS index.json"""
    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        return None
    
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
        # Sum samples from all shards
        total_samples = sum(shard.get('samples', 0) for shard in index.get('shards', []))
        return total_samples
    except Exception as e:
        print(f"Error reading index.json: {e}")
        return None

def retokenize_domain(domain_name, input_root, output_root, src_model, tgt_model, args):
    input_dir = os.path.join(input_root, domain_name)
    output_dir = os.path.join(output_root, domain_name)
    
    if not os.path.isdir(input_dir):
        print(f"Skipping {domain_name}: Input directory not found at {input_dir}")
        return

    # Get input sample count
    input_sample_count = get_dataset_sample_count(input_dir)
    if input_sample_count is None:
        print(f"Warning: Could not determine input sample count for {domain_name}")
        input_sample_count = float('inf')  # Proceed anyway
    
    # Check if output exists and is complete
    output_exists = os.path.exists(output_dir)
    needs_processing = True
    
    if output_exists:
        output_sample_count = get_dataset_sample_count(output_dir)
        
        if output_sample_count is not None:
            # We have index.json, check if it's complete
            completion_pct = 100 * output_sample_count / max(input_sample_count, 1)
            print(f"{domain_name}: Found {output_sample_count}/{input_sample_count} samples ({completion_pct:.1f}%)")
            
            if output_sample_count >= input_sample_count:
                if not args.overwrite:
                    print(f"Skipping {domain_name}: Output is complete")
                    return
                else:
                    print(f"Overwriting complete output at {output_dir}")
                    shutil.rmtree(output_dir)
            else:
                # Partial completion
                print(f"Removing incomplete output at {output_dir}")
                shutil.rmtree(output_dir)
        else:
            # Directory exists but no valid index.json
            print(f"Removing corrupt/incomplete output at {output_dir}")
            shutil.rmtree(output_dir)

    print(f"Processing domain: {domain_name}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")

    # Load dataset
    # We use batch_size 1 for the StreamingDataset but will use a DataLoader for batching processing
    dataset = StreamingDataset(
        streams=[SafeStream(local=input_dir, remote=None)],
        shuffle=False,
        batch_size=1 
    )
    
    # Custom collator that runs in worker processes
    collator = RetokenizeCollator(src_model, tgt_model)

    # DataLoader to batch the read samples AND perform tokenization in parallel workers
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=collator,
        prefetch_factor=2 # Buffer batches
    )

    columns = {
        "input_ids": "ndarray:int32", # Standardize on int32
        "length": "int64"
    }

    # Use 'hashes' or 'hash' depending on version (older versions used hash, newer hashes)
    # To be safe, try to inspect or just use 'hashes' as it worked for user previously
    # (User reported ValueError: Invalid Writer argument(s): ['hash'])
    writer_kwargs = {"out": output_dir, "columns": columns, "compression": None}
    try:
        # Try new API first
        with MDSWriter(**writer_kwargs, hashes=None) as out:
            for batch_new_ids in tqdm(loader, desc=f"Retokenizing {domain_name}"):
                # batch_new_ids is a list of numpy arrays from the collator
                for new_ids_np in batch_new_ids:
                    out.write({
                        "input_ids": new_ids_np,
                        "length": len(new_ids_np)
                    })
    except TypeError:
        # Fallback for older API if 'hashes' is unexpected
        with MDSWriter(**writer_kwargs, hash=None) as out:
            for batch_new_ids in tqdm(loader, desc=f"Retokenizing {domain_name}"):
                for new_ids_np in batch_new_ids:
                    out.write({
                        "input_ids": new_ids_np,
                        "length": len(new_ids_np)
                    })

def main():
    parser = argparse.ArgumentParser(description="Retokenize ProLong dataset for MiniCPM")
    parser.add_argument("--input_path", type=str, default="../../datasets/prolong-64k", help="Path to original ProLong dataset")
    parser.add_argument("--output_path", type=str, default="../../datasets/prolong-64k-minicpm", help="Path to save new dataset")
    parser.add_argument("--src_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tokenizer used for original data")
    parser.add_argument("--tgt_model", type=str, default="openbmb/MiniCPM4.1-8B", help="Tokenizer to convert to")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing (sequences per batch)")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers (CPU cores for parallel tokenization)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--limit", type=int, default=None, help="Limit batches per domain for testing")
    
    args = parser.parse_args()

    # Auto-discover all domains in the input directory
    # This ensures we don't miss any domains like dclm-baseline
    domains = []
    for entry in sorted(os.listdir(args.input_path)):
        entry_path = os.path.join(args.input_path, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "index.json")):
            domains.append(entry)
    
    print(f"Discovered {len(domains)} domains to process:")
    for domain in domains:
        print(f"  - {domain}")

    os.makedirs(args.output_path, exist_ok=True)

    for domain in domains:
        retokenize_domain(domain, args.input_path, args.output_path, args.src_model, args.tgt_model, args)

    print("Done! New dataset is ready at:", args.output_path)
    print("Don't forget to update your training script to point to this new path!")

if __name__ == "__main__":
    main()
