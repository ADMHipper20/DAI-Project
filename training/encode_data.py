# DAI Data Encoding Module
# ========================
# Encodes text data into binary tokens for training.
# Supports SFT (Supervised Fine-tuning) format.
# Includes both training_corpus.txt AND chosen responses from dpo_training_data.txt

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import os
import json
import re


def encode_data(
    tokenizer_path: str = "tokenizer/dai_tokenizer.json",
    input_file: str = "data/training_corpus.txt",
    output_file: str = "data/train_data.bin",
    add_eos: bool = True,
    eos_token_id: int = 3,
    fp16: bool = True
):
    """
    Encode text data to binary tokens.
    
    Args:
        tokenizer_path: Path to BPE tokenizer
        input_file: Input text file
        output_file: Output binary file
        add_eos: Add EOS token at end of each line
        eos_token_id: Token ID for EOS (default=3 for [EOS])
        fp16: Use float16 instead of uint16
    
    Returns:
        Number of tokens encoded
    """
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    
    all_ids = []
    total_lines = 0
    
    # =================================================================
    # Encode SFT data (training_corpus.txt)
    # =================================================================
    if os.path.exists(input_file):
        print(f"\nEncoding SFT data: {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="SFT"):
                line = line.strip()
                if line:
                    ids = tokenizer.encode(line).ids
                    if add_eos:
                        ids.append(eos_token_id)
                    all_ids.extend(ids)
                    total_lines += 1
        print(f"  SFT: {total_lines:,} lines")
    else:
        print(f"Warning: {input_file} not found")
    
    # =================================================================
    # Encode DPO chosen responses (for SFT training)
    # =================================================================
    dpo_file = "data/dpo_training_data.txt"
    if os.path.exists(dpo_file):
        print(f"\nEncoding DPO chosen responses: {dpo_file}...")
        dpo_lines = 0
        
        with open(dpo_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="DPO"):
                line = line.strip()
                if not line:
                    continue
                
                # Extract chosen response only (everything before <!-- REJECTED:)
                if "<!-- REJECTED:" in line:
                    chosen_part = line.split("<!-- REJECTED:")[0].strip()
                else:
                    chosen_part = line.strip()
                
                if chosen_part:
                    ids = tokenizer.encode(chosen_part).ids
                    if add_eos:
                        ids.append(eos_token_id)
                    all_ids.extend(ids)
                    dpo_lines += 1
        
        print(f"  DPO: {dpo_lines:,} chosen responses")
    else:
        print(f"Warning: {dpo_file} not found (skipping DPO)")
    
    # Convert to numpy array (use uint32 to support vocab up to 4B)
    arr = np.array(all_ids, dtype=np.uint32)
    
    # Save as binary
    arr.tofile(output_file)
    
    # Stats
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"Encoding complete!")
    print(f"  Total tokens: {len(arr):,}")
    print(f"  Output file: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"{'='*50}")
    
    return len(arr)


def load_encoded_data(
    data_file: str = "data/train_data.bin",
    memory_map: bool = True
):
    """
    Load encoded binary data for training.
    
    Args:
        data_file: Path to binary file
        memory_map: Use memory mapping (recommended for large files)
    
    Returns:
        Numpy array of tokens
    """
    print(f"Loading {data_file}...")
    
    if memory_map:
        # Memory map - doesn't load into RAM, just maps the file
        data = np.memmap(data_file, dtype=np.uint32, mode='r')
    else:
        # Load fully into RAM
        data = np.fromfile(data_file, dtype=np.uint32)
    
    print(f"Loaded {len(data):,} tokens")
    return data


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DAI Data Encoder")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/dai_tokenizer.json")
    parser.add_argument("--input", type=str, default="data/training_corpus.txt")
    parser.add_argument("--output", type=str, default="data/train_data.bin")
    parser.add_argument("--no-eos", action="store_true", help="Don't add EOS tokens")
    parser.add_argument("--no-fp16", action="store_true", help="Use uint16 instead of fp16")
    
    args = parser.parse_args()
    
    # Encode
    n_tokens = encode_data(
        tokenizer_path=args.tokenizer,
        input_file=args.input,
        output_file=args.output,
        add_eos=not args.no_eos,
        fp16=not args.no_fp16
    )
    
    if n_tokens:
        print(f"\nEncoding complete! {n_tokens:,} tokens saved.")
