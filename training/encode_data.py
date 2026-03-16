# data/encode_data.py
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import os

tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
input_file = "data/training_corpus.txt"
output_file = "data/train_data.bin"

print("Encoding data line-by-line to prevent OOM...")
all_ids = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Tokenizing"):
        line = line.strip()
        if line:
            # Encode just one line at a time, keeping RAM usage near 0
            all_ids.extend(tokenizer.encode(line).ids)

print("Converting to 16-bit integers and saving to SSD...")
# Convert to numpy uint16 array and save directly as a raw binary file
arr = np.array(all_ids, dtype=np.uint16)
arr.tofile(output_file)

print(f"Success! Saved {len(arr):,} tokens to {output_file}")
print(f"File size on disk: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")