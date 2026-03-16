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
# 1. MODEL CONFIGURATION - Optimized for RTX 4060 8GB VRAM
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "tiny": {
        "embed_size": 384,
        "num_heads": 6,
        "num_layers": 8,
        "batch_size": 32,
        "block_size": 128,
        "gradient_checkpointing": False,
        "description": "~20M params - Fastest"
    },
    "small": {
        "embed_size": 512,
        "num_heads": 8,
        "num_layers": 12,
        "batch_size": 16,
        "block_size": 256,
        "gradient_checkpointing": False,
        "description": "~70M params - Balanced"
    },
    "medium": {
        "embed_size": 768,
        "num_heads": 12,
        "num_layers": 16,
        "batch_size": 8,
        "block_size": 512,
        "gradient_checkpointing": False,
        "description": "~150M params - Higher quality"
    },
}

# Select config
CONFIG_NAME = "small"  # Change to "tiny", "small", or "medium"
config = MODEL_CONFIGS[CONFIG_NAME]

EMBED_SIZE = config["embed_size"]
NUM_HEADS = config["num_heads"]
NUM_LAYERS = config["num_layers"]
BATCH_SIZE = config["batch_size"]
BLOCK_SIZE = config["block_size"]
GRADIENT_CHECKPOINTING = config.get("gradient_checkpointing", False)

# -----------------------------------------------------------------------------
# 2. TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
TARGET_EPOCHS = 2
LEARNING_RATE = 1e-4
MIN_LR = 1e-5
WARMUP_RATIO = 0.1  # 10% of iterations for warmup
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_INTERVAL = 100

# -----------------------------------------------------------------------------
# 3. LOAD DATA
# -----------------------------------------------------------------------------
print(f"Loading Tokenizer...")
tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

import numpy as np

print("Loading memory-mapped dataset...")
data = np.memmap("data/train_data.bin", dtype=np.uint16, mode="r")

# 90/10 split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

train_tokens = len(train_data)
val_tokens = len(val_data)

# Calculate iterations needed: tokens = batch_size * block_size * iterations
# iterations = tokens / (batch_size * block_size)
# Calculate iterations dynamically to prevent overfitting
tokens_per_iter = BATCH_SIZE * BLOCK_SIZE
ITERATIONS = int((train_tokens * TARGET_EPOCHS) / tokens_per_iter)
WARMUP_ITERS = min(200, int(ITERATIONS * WARMUP_RATIO))

print(f"Train: {train_tokens:,} | Val: {val_tokens:,} | Epochs: {TARGET_EPOCHS}")
print(f"Iterations: {ITERATIONS:,} | Warmup: {WARMUP_ITERS}")
print(f"LR: {LEARNING_RATE} -> {MIN_LR}")

# -----------------------------------------------------------------------------
# 4. DATA LOADER
# -----------------------------------------------------------------------------
def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    seq_len = BLOCK_SIZE
    
    # Random positions
    ix = torch.randint(len(dataset) - seq_len, (BATCH_SIZE,))
    
    # Efficient batch extraction
    x = torch.stack([
        torch.from_numpy(dataset[i:i+seq_len].astype(np.int64)) 
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(dataset[i+1:i+seq_len+1].astype(np.int64)) 
        for i in ix
    ])
    
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

# -----------------------------------------------------------------------------
# 5. LEARNING RATE SCHEDULER
# -----------------------------------------------------------------------------
def get_lr(iter_num):
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS
    if iter_num > ITERATIONS:
        return MIN_LR
    decay_ratio = (iter_num - WARMUP_ITERS) / (ITERATIONS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# -----------------------------------------------------------------------------
# 6. MODEL
# -----------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"DAI Training ({CONFIG_NAME}): {EMBED_SIZE} embed, {NUM_LAYERS} layers, {NUM_HEADS} heads")
print(f"{'='*50}\n")

model = DAIModel(
    vocab_size=VOCAB_SIZE, 
    embed_size=EMBED_SIZE, 
    max_seq_length=BLOCK_SIZE, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS
)
model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.2f} M")

# -----------------------------------------------------------------------------
# 7. OPTIMIZER & SCALER
# -----------------------------------------------------------------------------
try:
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        fused=True
    )
    print("Using fused AdamW")
except:
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS
    )
    print("Using standard AdamW")

# AMP GradScaler for FP16 training
scaler = torch.amp.GradScaler('cuda')

# -----------------------------------------------------------------------------
# 8. TRAINING LOOP
# -----------------------------------------------------------------------------
print(f"\n{'Step':>6} | {'Loss':>10} | {'Val':>10} | {'LR':>12} | {'Speed':>8}")
print("-" * 60)

t0 = time.time()
running_loss = 0.0

model.train()
for iter in range(ITERATIONS):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    xb, yb = get_batch('train')
    
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)
    
    # Backward with scaler
    scaler.scale(loss).backward()
    
    # Gradient Clipping (Crucial step: we must unscale the gradients BEFORE clipping)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    
    running_loss += loss.item()
    
    if iter % EVAL_INTERVAL == 0 or iter == ITERATIONS - 1:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch('val')
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, val_loss = model(xb_val, yb_val)
        
        avg_loss = running_loss / EVAL_INTERVAL
        elapsed = time.time() - t0
        speed = (BATCH_SIZE * BLOCK_SIZE * EVAL_INTERVAL) / elapsed
        
        print(f"{iter:6d} | {avg_loss:10.4f} | {val_loss.item():10.4f} | {lr:12.8f} | {speed:6.0f}/s")
        
        running_loss = 0.0
        model.train()
        t0 = time.time()

# -----------------------------------------------------------------------------
# 9. SAVE
# -----------------------------------------------------------------------------
os.makedirs("weights", exist_ok=True)
save_path = f"weights/dai_agent_{CONFIG_NAME}.pt"

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': VOCAB_SIZE,
        'embed_size': EMBED_SIZE,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'block_size': BLOCK_SIZE,
    },
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

print(f"\n✓ Training complete! Saved to {save_path}")
print(f"Parameters: {num_params / 1e6:.2f} M")
