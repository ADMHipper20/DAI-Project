import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    'Causal' means the model can only look at past tokens, not future ones.
    """
    def __init__(self, embed_size, num_heads, dropout_rate=0.1):
        super().__init__()
        # Ensure the embedding size can be cleanly divided by the number of heads
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # A single linear layer to project Query, Key, and Value simultaneously (much faster)
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=False)
        
        # Output projection layer to mix the heads back together
        self.c_proj = nn.Linear(embed_size, embed_size, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, embed_size)
               This comes directly from our Embedding layer!
        """
        B, T, C = x.size() # Batch, Time (Sequence Length), Channels (Embed Size)
        
        # 1. Calculate Q, K, V for all heads in one batch
        # Apply the linear layer, then split the output into three equal chunks
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)
        
        # 2. Reshape to isolate the multiple heads
        # Shape becomes: (Batch, Num_Heads, Time, Head_Dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply Scaled Dot-Product Attention
        # is_causal=True automatically applies the triangular mask so tokens can't look ahead.
        # This heavily utilizes your RTX 4060's tensor cores.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0, 
            is_causal=True
        )
        
        # 4. Re-assemble the heads
        # Swap axes back to (Batch, Time, Num_Heads, Head_Dim) and flatten the last two
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. Final output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y

# --- QUICK TEST ---
if __name__ == "__main__":
    B, T, C = 4, 12, 512 # Batch: 4, Seq Len: 12, Embed Size: 512
    NUM_HEADS = 8
    
    # Simulate the output coming from your embedding.py
    simulated_embeddings = torch.randn(B, T, C)
    
    mha_layer = MultiHeadAttention(embed_size=C, num_heads=NUM_HEADS)
    output = mha_layer(simulated_embeddings)
    
    print(f"Input shape (from Embeddings): {simulated_embeddings.shape}")
    print(f"Output shape (from Attention): {output.shape}")
    print("\nIf shapes match, the Attention mechanism is cleanly processing the vectors!")