# Kaggle-optimized training script for DAI
# Upload this file to Kaggle along with the model files

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
from tqdm import tqdm

# =============================================================================
# KAGGLE SETUP - Upload these files to Kaggle:
# =============================================================================
# DAI/
# ├── model/
# │   ├── __init__.py
# │   ├── transformer.py
# │   ├── attention.py  
# │   └── embedding.py
# ├── tokenizer/
# │   └── dai_tokenizer.json  (your trained tokenizer)
# ├── data/
# │   └── training_corpus.txt (your data)
# ├── training/
# │   └── kaggle_train.py     (this file)
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# CONFIG - Adjust these for your GPU
# =============================================================================
# RTX 4060 (8GB): layers=6, embed=512, heads=8, batch=16, block=256
# T4 (16GB): layers=12, embed=768, heads=12, batch=32, block=512  
# P100/V100 (16-32GB): layers=16, embed=1024, heads=16, batch=64, block=512

MODEL_CONFIG = {
    "layers": 12,
    "embed": 768,
    "heads": 12,
    "block": 512,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "iterations": 5000,
    "lr": 1e-4,
    "warmup": 500,
}

# =============================================================================
# LOAD TOKENIZER
# =============================================================================
print("\nLoading tokenizer...")
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Vocab size: {VOCAB_SIZE}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")
data = []
with open("data/training_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        ids = tokenizer.encode(line.strip()).ids
        data.extend(ids + [tokenizer.token_to_id("[EOS]")])

data = torch.tensor(data, dtype=torch.long)
print(f"Total tokens: {len(data):,}")

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =============================================================================
# MODEL
# =============================================================================
print("\nBuilding model...")
sys.path.append(".")
from model.transformer import DAIModel

model = DAIModel(
    vocab_size=VOCAB_SIZE,
    embed_size=MODEL_CONFIG["embed"],
    max_seq_length=MODEL_CONFIG["block"],
    num_layers=MODEL_CONFIG["layers"],
    num_heads=MODEL_CONFIG["heads"]
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params / 1e6:.1f}M")

# =============================================================================
# TRAINING
# =============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"])
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=TRAIN_CONFIG["warmup"]
)

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - MODEL_CONFIG["block"], (TRAIN_CONFIG["batch_size"],))
    x = torch.stack([data_split[i:i+MODEL_CONFIG["block"]] for i in ix])
    y = torch.stack([data_split[i+1:i+MODEL_CONFIG["block"]+1] for i in ix])
    return x.to(device), y.to(device)

print(f"\nTraining for {TRAIN_CONFIG['iterations']} iterations...")
model.train()

for i in tqdm(range(TRAIN_CONFIG["iterations"])):
    xb, yb = get_batch('train')
    
    # Forward
    logits, loss = model(xb, yb)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if i % 500 == 0:
        print(f"\nIter {i}: loss={loss.item():.4f}")

# =============================================================================
# SAVE
# =============================================================================
print("\nSaving model...")
os.makedirs("weights", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {**MODEL_CONFIG, 'vocab_size': VOCAB_SIZE}
}, "weights/dai_kaggle.pt")
print("Saved to weights/dai_kaggle.pt")
