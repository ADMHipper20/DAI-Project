import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential

# Import the modules we just built!
try:
    from attention import MultiHeadAttention
    from embedding import DAIEmbeddings
except ImportError:
    from .attention import MultiHeadAttention
    from .embedding import DAIEmbeddings

class FeedForward(nn.Module):
    """
    A simple linear network followed by a non-linearity.
    After the Attention mechanism figures out *how* words relate, 
    this network actually *computes* the new meaning.
    """
    def __init__(self, embed_size, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # In standard Transformers, the hidden layer is 4x the embed size
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(), # GELU is the modern standard activation function for LLMs (better than ReLU)
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    One complete layer of the Transformer.
    Combines Attention and FeedForward with Layer Normalization and Residual Connections.
    """
    def __init__(self, embed_size, num_heads, dropout_rate=0.1):
        super().__init__()
        # LayerNorm keeps our values stable so the RTX 4060 doesn't get exploding gradients
        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout_rate)
        
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffwd = FeedForward(embed_size, dropout_rate)

    def forward(self, x):
        # Residual connections: x = x + layer(norm(x))
        # This allows gradients to flow smoothly through deep networks
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DAIModel(nn.Module):
    """
    The complete DAI Agent Language Model!
    """
    def __init__(self, vocab_size, embed_size, max_seq_length, num_layers, num_heads, dropout_rate=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # 1. The Input Layer
        self.embeddings = DAIEmbeddings(vocab_size, embed_size, max_seq_length, dropout_rate)
        
        # 2. The Hidden Layers (Stacking multiple Transformer Blocks)
        # nn.ModuleDict or nn.ModuleList is required so PyTorch tracks the weights
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # 3. The Output Layer
        self.ln_f = nn.LayerNorm(embed_size) # Final normalization
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False) # Predicts the next token ID

    def forward(self, input_ids, targets=None):
        # 1. Convert IDs to rich vectors
        x = self.embeddings(input_ids)
        
        # 2. Pass through the Transformer blocks
        x = self.blocks(x)
        
        # 3. Final normalization and prediction
        x = self.ln_f(x)
        logits = self.lm_head(x) # Shape: (Batch, Sequence Length, Vocab Size)
        
        loss = None
        if targets is not None:
            # Reshape for cross_entropy calculation
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory"""
        # Use gradient checkpointing on the transformer blocks
        self._gradient_checkpointing = True
        # Replace forward with checkpointed version
        self._original_forward = self.forward
        self.forward = self._forward_with_checkpoint
    
    def _forward_with_checkpoint(self, input_ids, targets=None):
        """Forward pass with gradient checkpointing"""
        import torch.utils.checkpoint as cp
        
        x = self.embeddings(input_ids)
        
        # Checkpoint each block
        def block_forward(block, x):
            return block(x)
        
        # Apply checkpointing to each transformer block
        for block in self.blocks:
            x = cp.checkpoint(block_forward, block, x, use_reentrant=False)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

# --- QUICK TEST ---
if __name__ == "__main__":
    # Hyperparameters carefully tuned for an 8GB VRAM RTX 4060
    VOCAB_SIZE = 32000
    MAX_SEQ_LEN = 512
    EMBED_SIZE = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6 # 6 layers is a great start (GPT-1 had 12, but we want fast iterations first)
    
    # Initialize the full model
    model = DAIModel(
        vocab_size=VOCAB_SIZE, 
        embed_size=EMBED_SIZE, 
        max_seq_length=MAX_SEQ_LEN, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS
    )
    
    # Test with dummy data
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 128)) # Batch size 2, Sequence length 128
    dummy_targets = torch.randint(0, VOCAB_SIZE, (2, 128))
    
    logits, loss = model(dummy_input, dummy_targets)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Logits Shape (Predictions): {logits.shape}")
    print(f"Calculated Loss: {loss.item():.4f}")
    print("\nIf you see a ~35 Million parameter count and a valid loss number, DAI's brain is completely built! 🧠")