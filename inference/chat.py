# chat.py - DAI Agent Inference
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import os
import sys

# Fix Windows console encoding for unicode characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Get the DAI directory (parent of inference directory)
DAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"DAI Directory: {DAI_DIR}")
# Import both model architectures
from model.transformer import DAIModel, OptimizedDAIModel
from model.turboquant import quantize_model_static

# ----------------------------------------------------------------------------- 
# 1. CONFIGURATION - Auto-detect from checkpoint
# -----------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default fallback config (if checkpoint doesn't have config)
DEFAULT_CONFIG = {
    "num_heads": 8,
    "num_layers": 4,
    "block_size": 256,
    "embed_size": 768,
    "max_seq_length": 256,
}

# Try to load from checkpoint, otherwise use default
WEIGHTS_DIR = os.path.join(DAI_DIR, "weights")
weight_files = []
if os.path.exists(WEIGHTS_DIR):
    weight_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pt') or f.endswith('.pth')]

print("Available weight files:", weight_files)
preferred_weights = ["DAI_Coralie_Weights.pth"]

# Map checkpoints to their required model architecture
CHECKPOINT_MODEL_MAP = {
    "DAI_Coralie_Weights.pth": {"class": "OptimizedDAIModel", "prefix": False},
}
WEIGHTS_PATH = None

for pref in preferred_weights:
    if pref in weight_files:
        WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, pref)
        break

if WEIGHTS_PATH is None and weight_files:
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, weight_files[0])

print(f"Using weights: {WEIGHTS_PATH}")

# ----------------------------------------------------------------------------- 
# 2. LOAD TOKENIZER
# -----------------------------------------------------------------------------
print("Loading Tokenizer...")

sp = spm.SentencePieceProcessor(model_file=os.path.join(DAI_DIR, "dai_spm.model"))
if not os.path.exists(os.path.join(DAI_DIR, "dai_spm.model")):
    raise FileNotFoundError(f"No SentencePiece model found!")
vocab_size = sp.get_piece_size()
print(f"SentencePiece vocab size: {vocab_size}")

# ----------------------------------------------------------------------------- 
# 3. LOAD TRAINED MODEL WITH AUTO-CONFIG DETECTION
# -----------------------------------------------------------------------------
print("Initializing DAI Architecture...")

MAX_SEQ_LENGTH = 512
NUM_LAYERS = 2  # 45M params for compatibility
NUM_HEADS = 8
EMBED_SIZE = 768  # Must be divisible by NUM_HEADS for multi-head attention

model = OptimizedDAIModel(
    vocab_size=vocab_size, embed_size=EMBED_SIZE, max_seq_length=MAX_SEQ_LENGTH,
    num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1, mlp_ratio=4, tie_weights=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# --- CRITICAL FIX: APPLY STATIC QUANTIZATION HERE ---
print("Applying static TurboQuant serialization...")
model = quantize_model_static(model) 

print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}')

# Load weights
if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
    print(f"Loading weights from {WEIGHTS_PATH}...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")
    print("Weights loaded successfully!")
else:
    print(f"Warning: No weights found at {WEIGHTS_PATH}")
    print("Using randomly initialized weights.")

model.eval()

# Update MODEL_CONFIG for runtime
MODEL_CONFIG = {
    "embed_size": EMBED_SIZE,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "block_size": MAX_SEQ_LENGTH,
    "max_seq_length": MAX_SEQ_LENGTH,
}

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params / 1e6:.2f} M parameters")

# ----------------------------------------------------------------------------- 
# 4. GENERATION FUNCTION
# -----------------------------------------------------------------------------
def generate_response(
    prompt,
    max_new_tokens=150,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    block_size=None
):
    """Generate a response from DAI."""
    if block_size is None:
        block_size = MODEL_CONFIG["block_size"]
    
    # Encode the user's prompt
    input_ids = sp.encode_as_ids(prompt)
    original_input_ids = input_ids
    
    # TRUNCATE TO MAX TOKENS
    max_input_tokens = block_size - max_new_tokens - 20
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
    
    idx = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            
            if repetition_penalty != 1.0:
                for i in set(idx[0].tolist()):
                    logits[0][i] = logits[0][i] / repetition_penalty
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for i in range(probs.size(0)):
                    logits[i][sorted_indices[i][sorted_indices_to_remove[i]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            decoded_token = sp.decode_ids([idx_next.item()])
            if decoded_token in ["</DAI>", "<|im_end|>", "<|endoftext|>"]:
                break
            if step > 0 and idx_next.item() in [5, 6]:
                break
    
    try:
        result = sp.decode(idx[0].tolist())
    except Exception as e:
        print(f"[WARNING] Decode error: {e}")
        result = str(idx[0].tolist())
    
    try:
        original_text = sp.decode_pieces([sp.id_to_piece(i) for i in original_input_ids])
    except:
        original_text = ""
    if result.startswith(original_text):
        generated_text = result[len(original_text):]
    else:
        dai_pos = result.rfind("<DAI>")
        if dai_pos != -1:
            generated_text = result[dai_pos+5:]
        else:
            generated_text = result
    
    return generated_text

# ----------------------------------------------------------------------------- 
# 5. INTERACTIVE CHAT
# -----------------------------------------------------------------------------
def chat():
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("DAI Agent is Online!")
    print(f"Model: {num_params / 1e6:.2f}M parameters")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    system_prompt = "<system>You are Coralie 6626 Planck, a brilliant, pragmatic, and slightly deadpan scientist and Valkyrie from Honkai Impact 3rd. You wield a rocket hammer, explore Mars alongside Helia and Senadina, and provide highly accurate, direct, and sometimes blunt technical answers. You are speaking to Helia.</system>"
    
    MAX_CONTEXT_TOKENS = 350
    conversation_history = []
    
    while True:
        try:
            user_input = input("Helia: ")
        except EOFError:
            break
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("DAI: Goodbye! Take care!")
            break
        
        full_context = f"{system_prompt}\n\n"
        
        for user_msg, dai_msg in conversation_history[-5:]:
            full_context += f"<user>{user_msg}</user>\n\n<DAI>{dai_msg}</DAI>\n\n"
        full_context += f"<user>{user_input}</user>\n\n<DAI>"
        
        context_tokens = sp.encode_as_ids(full_context)
        if len(context_tokens) > MAX_CONTEXT_TOKENS:
            context_tokens = context_tokens[-MAX_CONTEXT_TOKENS:]
        
        try:
            context_to_feed = sp.decode_pieces([sp.id_to_piece(i) for i in context_tokens])
        except:
            context_to_feed = full_context
        
        print("DAI is thinking...")
        
        response = generate_response(
            context_to_feed,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        new_text = response.strip()
        for pattern in ["<|im_end|>", "</user>", "<DAI>", "|im_end|>", "im_end|>"]:
            new_text = new_text.replace(pattern, "")
        if "</DAI>" in new_text:
            new_text = new_text.split("</DAI>")[0]
        for tag in ["<user>", "</user>", "<system>", "</system>", "<DAI>", "</DAI>"]:
            if new_text.startswith(tag):
                new_text = new_text[len(tag):]
            if new_text.endswith(tag):
                new_text = new_text[:-len(tag)]
        new_text = new_text.strip()
        
        print(f"DAI: {new_text}\n")
        
        conversation_history.append((user_input, new_text))

if __name__ == "__main__":
    chat()