# Use SentencePiece directly for inference (better than HuggingFace conversion)
import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
import sys

# Import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import DAIModel

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Auto-detect GPU and set model size
def get_device_config():
    """Auto-detect best config based on available GPU"""
    if not torch.cuda.is_available():
        return {"layers": 6, "embed": 512, "heads": 8, "block": 256}
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    if 't4' in gpu_name or vram >= 15:
        # Colab T4, Kaggle, etc
        return {"layers": 12, "embed": 768, "heads": 12, "block": 512}
    elif 'p100' in gpu_name:
        return {"layers": 12, "embed": 768, "heads": 12, "block": 512}
    elif 'v100' in gpu_name:
        return {"layers": 16, "embed": 1024, "heads": 16, "block": 512}
    elif '4090' in gpu_name or vram >= 20:
        return {"layers": 24, "embed": 1536, "heads": 24, "block": 1024}
    else:
        # RTX 4060, 3060, etc
        return {"layers": 6, "embed": 512, "heads": 8, "block": 256}

# Get config
device_config = get_device_config()
print(f"Detected device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Using config: {device_config}")

# Model weights
WEIGHTS_DIR = "weights"
weight_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pt')] if os.path.exists(WEIGHTS_DIR) else []
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, weight_files[0]) if weight_files else None

# Load SentencePiece tokenizer
print("Loading SentencePiece tokenizer...")
SP_MODEL = "tokenizer/dai_sp.model"
if not os.path.exists(SP_MODEL):
    print(f"Warning: {SP_MODEL} not found, using HuggingFace tokenizer")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
    VOCAB_SIZE = tokenizer.get_vocab_size()
    use_sp = False
else:
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)
    VOCAB_SIZE = sp.get_piece_size()
    use_sp = True
    print(f"SentencePiece vocab: {VOCAB_SIZE}")

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
print("Loading DAI model...")

MODEL_CONFIG = {
    "embed_size": device_config["embed"],
    "num_heads": device_config["heads"],
    "num_layers": device_config["layers"],
    "block_size": device_config["block"],
}

model = DAIModel(
    vocab_size=VOCAB_SIZE,
    embed_size=MODEL_CONFIG["embed_size"],
    max_seq_length=MODEL_CONFIG["block_size"],
    num_layers=MODEL_CONFIG["num_layers"],
    num_heads=MODEL_CONFIG["num_heads"]
)

if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded: {WEIGHTS_PATH}")

model.to(DEVICE)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params / 1e6:.1f}M params")

# -----------------------------------------------------------------------------
# GENERATION FUNCTION
# -----------------------------------------------------------------------------
def encode(text):
    """Encode text to IDs"""
    if use_sp:
        return [sp.bos_id()] + sp.encode(text) + [sp.eos_id()]
    else:
        return tokenizer.encode(text).ids

def decode(ids):
    """Decode IDs to text"""
    if use_sp:
        return sp.decode(ids)
    else:
        return tokenizer.decode(ids)

def generate(prompt, max_tokens=100, temp=0.5, top_k=40, rep_pen=1.3):
    """Generate response"""
    ids = encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -MODEL_CONFIG["block_size"]:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temp
            
            # Repetition penalty
            if rep_pen != 1.0:
                for i in set(idx[0].tolist()):
                    logits[0][i] /= rep_pen
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if use_sp and idx_next.item() == sp.eos_id():
                break
    
    return decode(idx[0].tolist())

# -----------------------------------------------------------------------------
# CHAT
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("🧠 DAI (SentencePiece) is Online!")
print(f"Model: {num_params/1e6:.1f}M | GPU: {DEVICE}")
print("Type 'quit' to exit")
print("="*50 + "\n")

while True:
    user = input("You: ")
    if user.lower() in ['quit', 'exit']:
        break
    
    prompt = f"[USER] {user}\n[DAI]"
    print("DAI thinking...")
    
    resp = generate(prompt, max_tokens=100, temp=0.5, rep_pen=1.3)
    # Extract just the response
    resp = resp.split("[DAI]")[-1].split("[USER]")[0].strip()
    
    print(f"DAI: {resp}\n")
