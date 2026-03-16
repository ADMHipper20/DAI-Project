# chat.py - DAI Agent Inference
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import sys

# Import your custom architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import DAIModel

# -----------------------------------------------------------------------------
# 1. CONFIGURATION - Auto-detect from checkpoint
# -----------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default fallback config (if checkpoint doesn't have config)
DEFAULT_CONFIG = {
    "embed_size": 512,
    "num_heads": 8,  
    "num_layers": 6,
    "block_size": 256,
}

# Try to load from checkpoint, otherwise use default
# Check available weight files
WEIGHTS_DIR = "weights"
weight_files = []
if os.path.exists(WEIGHTS_DIR):
    weight_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pt')]

print("Available weight files:", weight_files)

# Priority: dai_agent_v1.pt > dai_agent_small.pt > dai_agent_medium_v1.pt
preferred_weights = ["dai_agent_v1.pt", "dai_agent_small.pt", "dai_agent_medium_v1.pt"]
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
tokenizer = Tokenizer.from_file("tokenizer/dai_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Vocab size: {VOCAB_SIZE}")

# -----------------------------------------------------------------------------
# 3. LOAD TRAINED MODEL WITH AUTO-CONFIG DETECTION
# -----------------------------------------------------------------------------
print("Waking up DAI...")

# Load checkpoint to get config
MODEL_CONFIG = DEFAULT_CONFIG.copy()

if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
    print(f"Loading checkpoint from {WEIGHTS_PATH}...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    # Try to extract config from checkpoint
    if 'config' in checkpoint:
        MODEL_CONFIG = checkpoint['config']
        print(f"Loaded config from checkpoint: {MODEL_CONFIG}")
    else:
        # Try to infer from state_dict keys
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        if isinstance(state_dict, dict):
            # Count layers from state dict
            layer_keys = [k for k in state_dict.keys() if 'blocks.' in k and '.attn' in k]
            if layer_keys:
                layers = set()
                for k in layer_keys:
                    parts = k.split('.')
                    for i, p in enumerate(parts):
                        if p == 'blocks' and i+1 < len(parts):
                            try:
                                layers.add(int(parts[i+1]))
                            except:
                                pass
                if layers:
                    MODEL_CONFIG["num_layers"] = max(layers) + 1
                    print(f"Inferred num_layers: {MODEL_CONFIG['num_layers']}")
            
            # Try to get embed_size from embedding or first layer
            if 'embeddings.token_embedding.weight' in state_dict:
                embed_size = state_dict['embeddings.token_embedding.weight'].shape[1]
                MODEL_CONFIG["embed_size"] = embed_size
                print(f"Inferred embed_size: {embed_size}")
                
print(f"Using model config: {MODEL_CONFIG}")

# Initialize model with detected/resolved config
model = DAIModel(
    vocab_size=VOCAB_SIZE, 
    embed_size=MODEL_CONFIG["embed_size"], 
    max_seq_length=MODEL_CONFIG["block_size"], 
    num_layers=MODEL_CONFIG["num_layers"], 
    num_heads=MODEL_CONFIG["num_heads"]
)

# Load weights
if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
    print(f"Loading weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Weights loaded successfully!")
else:
    print(f"Warning: No weights found at {WEIGHTS_PATH}")
    print("Using randomly initialized weights.")

model.to(DEVICE)
model.eval()

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params / 1e6:.2f} M parameters")

# -----------------------------------------------------------------------------
# 4. GENERATION FUNCTION
# -----------------------------------------------------------------------------
def generate_response(
    prompt, 
    max_new_tokens=150, 
    temperature=0.5,  # Lower for coherence
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.3,  # Stronger to prevent loops
    block_size=None
):
    """Generate a response from DAI."""
    if block_size is None:
        block_size = MODEL_CONFIG["block_size"]
    
    # Encode the user's prompt
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in set(idx[0].tolist()):
                    logits[0][i] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(probs.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for EOS token
            if tokenizer.decode([idx_next.item()]) == "[EOS]":
                break
    
    return tokenizer.decode(idx[0].tolist())

# -----------------------------------------------------------------------------
# 5. INTERACTIVE CHAT
# -----------------------------------------------------------------------------
def chat():
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("🧠 DAI Agent is Online!")
    print(f"Model: {num_params / 1e6:.2f}M parameters")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    conversation_history = ""
    
    # Load agent instructions if available
    agent_instructions_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "Agent_Instruction.md"
    )
    if os.path.exists(agent_instructions_path):
        with open(agent_instructions_path, "r") as f:
            system_prompt = f.read()
        conversation_history = f"[SYSTEM]{system_prompt}\n"
    
    while True:
        try:
            user_input = input("Helia: ")
        except EOFError:
            break
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("DAI: Goodbye! Take care! 👋")
            break
        
        # Add user message
        conversation_history += f"[USER] {user_input}\n[DAI] "
        
        # Keep context within limits
        max_context = min(800, MODEL_CONFIG["block_size"] * 3)
        context_to_feed = conversation_history[-max_context:]
        
        print("DAI is thinking...")
        
        # Generate response
        response = generate_response(
            context_to_feed,
            max_new_tokens=150,
            temperature=0.5,  # Lower = more focused/coherent
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3  # Stronger repetition penalty
        )
        
        # Extract only the new response
        new_text = response[len(context_to_feed):].strip()
        new_text = new_text.split("[USER]")[0].strip()
        new_text = new_text.split("[SYSTEM]")[0].strip()
        
        print(f"DAI: {new_text}\n")
        
        # Add response to history
        conversation_history += f"{new_text}\n"

if __name__ == "__main__":
    chat()
