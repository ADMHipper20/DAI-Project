import torch
import os
from tokenizers import Tokenizer
import time
import math
import numpy as np

# Import our custom architecture
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import DAIModel

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETERS (The Original Fast Settings)
# -----------------------------------------------------------------------------
BATCH_SIZE = 16         # Parallel sequences
BLOCK_SIZE = 256        # Context window
MAX_ITERS = 5000        # Hardcoded for a fast run! (Increase later if you want)
EVAL_INTERVAL = 500     # Check loss every 500 steps
LEARNING_RATE = 3e-4    # Max learning rate
MIN_LR = 3e-5           # Min learning rate for cosine decay
WARMUP_ITERS = 500      # Steps to warm up

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model dimensions
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 6

# -----------------------------------------------------------------------------
# 2. LOAD DATA (SSD MEMMAP)
# -----------------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

print("Loading memory-mapped dataset from SSD...")
data = np.memmap("data/train_data.bin", dtype=np.uint16, mode="r")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Tokens total: {len(data):,} | Train: {len(train_data):,} | Val: {len(val_data):,}")

# -----------------------------------------------------------------------------
# 3. DATA LOADER
# -----------------------------------------------------------------------------
def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((dataset[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[i+1:i+BLOCK_SIZE+1]).astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# -----------------------------------------------------------------------------
# 4. LEARNING RATE SCHEDULER
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# -----------------------------------------------------------------------------
# 5. INITIALIZE MODEL & AMP
# -----------------------------------------------------------------------------
print("Initializing DAI Agent Brain...")
model = DAIModel(
    vocab_size=VOCAB_SIZE, 
    embed_size=EMBED_SIZE, 
    max_seq_length=BLOCK_SIZE, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda')

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
print(f"Starting Training Loop for {MAX_ITERS} iterations...")

# -----------------------------------------------------------------------------
# 6. THE TRAINING LOOP
# -----------------------------------------------------------------------------
t0 = time.time()
for iter in range(MAX_ITERS):
    
    # Update Learning Rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Validation phase
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            with torch.amp.autocast('cuda'):
                _, val_loss = model(xb, yb)
        print(f"Step {iter:4d} | Val Loss: {val_loss.item():.4f} | LR: {lr:.6f} | Time: {time.time()-t0:.2f}s")
        model.train()
        t0 = time.time()

    # Get training batch
    xb, yb = get_batch('train')
    
    # Forward & Backward Pass (Lightning Fast)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Save the weights!
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/dai_agent_v1.pt")
print("Training Complete! Weights saved to weights/dai_agent_v1.pt 🧠")