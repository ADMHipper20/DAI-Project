"""
DAI Optimized Transformer Model
============================== 
Efficient architecture for 100M-500M parameters.
Optimized for RTX 4060 (8GB VRAM) with FP16/INT8 support.

Key optimizations:
- SwiGLU activation (more efficient than GELU)
- RMSNorm (faster than LayerNorm)
- RoPE (Rotary Position Embedding) for better positional understanding
- Pre-norm architecture (better for deep networks)
- Weight tying between embeddings and lm_head
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

# =============================================================================
# RMSNorm - Faster than LayerNorm
# =============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - better than learned positional embeddings"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        # PRE-COMPUTE AND CACHE
        self._cos_cache = {}
        self._sin_cache = {}
        
    def forward(self, seq_len, device):
        # Cache by both seq_len AND device to avoid .to() transfers
        cache_key = (seq_len, str(device))
        if cache_key in self._cos_cache:
            return self._cos_cache[cache_key], self._sin_cache[cache_key]
        
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos_cached = emb.cos()[None, None, :, :]  # [1, 1, T, head_dim]
        sin_cached = emb.sin()[None, None, :, :]
        
        self._cos_cache[cache_key] = cos_cached
        self._sin_cache[cache_key] = sin_cached
        
        return cos_cached, sin_cached


def rotate_half(x):
    """Rotate half the hidden dims"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding - cos/sin shape: [1, 1, T, head_dim]"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# SwiGLU FeedForward - More efficient than GELU
# =============================================================================
class SwiGLU(nn.Module):
    """SwiGLU activation - used in LLaMA, PaLM, etc."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate = F.silu(self.w1(x))
        gate = torch.clamp(gate, min=-15, max=15)  # Prevent FP16 NaN
        return self.dropout(self.w2(gate * self.w3(x)))


# =============================================================================
# Optimized Multi-Head Attention
# =============================================================================
class OptimizedAttention(nn.Module):
    """
    Optimized Multi-Head Attention with RoPE.
    Uses Flash Attention when available via scaled_dot_product_attention.
    """
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(self.head_dim)
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Flash attention (uses GPU tensor cores)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0,
            is_causal=True
        )
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.o_proj(y))


# =============================================================================
# Transformer Block with Pre-Norm
# =============================================================================
class OptimizedTransformerBlock(nn.Module):
    """
    Optimized Transformer Block with Pre-Norm architecture.
    Pre-norm = Norm(x + Sublayer(x)) instead of Sublayer(Norm(x))
    Better for deep networks.
    """
    def __init__(self, embed_size, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_size * mlp_ratio
        
        # Pre-norm architecture
        self.norm1 = RMSNorm(embed_size)
        self.attn = OptimizedAttention(embed_size, num_heads, dropout)
        
        self.norm2 = RMSNorm(embed_size)
        self.ffn = SwiGLU(embed_size, hidden_dim, dropout)
    
    def forward(self, x):
        # Pre-norm attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm feedforward
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Embedding Layer with RoPE support
# =============================================================================
class OptimizedEmbedding(nn.Module):
    """Embedding layer with weight tying support"""
    def __init__(self, vocab_size, embed_size, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids):
        return self.token_embedding(input_ids)


# =============================================================================
# Complete Optimized DAI Model
# =============================================================================
class OptimizedDAIModel(nn.Module):
    """
    Optimized DAI Model for 100M-500M parameters.
    Uses modern techniques for better performance and efficiency.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        max_seq_length: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding
        self.embeddings = OptimizedEmbedding(vocab_size, embed_size, max_seq_length)
        
# Transformer blocks
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(embed_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        
        # Weight tying (reduces parameters, improves quality)
        if tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def resize_embeddings(self, new_vocab_size: int):
        """Resize embedding and lm_head to match tokenizer vocab size."""
        old_vocab = self.embeddings.token_embedding.num_embeddings
        old_embed = self.embeddings.token_embedding.weight.data.clone()
        
        self.embeddings.token_embedding = nn.Embedding(new_vocab_size, self.embed_size)
        nn.init.normal_(self.embeddings.token_embedding.weight, mean=0.0, std=0.02)
        
        # Copy old weights
        n = min(old_vocab, new_vocab_size)
        self.embeddings.token_embedding.weight.data[:n] = old_embed[:n]
        
        # Update lm_head (weight tied)
        if self.lm_head.weight.shape[0] != new_vocab_size:
            old_lm = self.lm_head.weight.data.clone()
            self.lm_head = nn.Linear(self.embed_size, new_vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            self.lm_head.weight.data[:n] = old_lm[:n]
            # Retie weights
            self.lm_head.weight = self.embeddings.token_embedding.weight
        
        return self

    def forward(self, input_ids, targets=None):
        x = self.embeddings(input_ids)
        
        # Through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0
    ):
        """Autoregressive text generation"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            input_ids = input_ids[:, -self.max_seq_length:]
            
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for token_id in input_ids[i]:
                        logits[i, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS
            if (next_token == 0).all():
                break
        
        return input_ids
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training"""
        import functools
        for layer in self.layers:
            original_forward = layer.forward
        
            @functools.wraps(original_forward)
            def checkpointed_forward(self, x):
                return torch.utils.checkpoint.checkpoint(original_forward, x, use_reentrant=False)
            
            # Bind method to layer
            layer.forward = lambda x, l=layer: torch.utils.checkpoint.checkpoint(l.forward, x, use_reentrant=False)


# =============================================================================
# Model Configs for Different Parameter Sizes
# =============================================================================
class DAIConfig:
    """Configuration presets for different model sizes"""
    
    # Small: ~100M parameters
    SMALL = {
        "vocab_size": 32000,
        "embed_size": 768,
        "max_seq_length": 512,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "dropout": 0.1,
    }
    
    # Medium: ~250M parameters  
    MEDIUM = {
        "vocab_size": 32000,
        "embed_size": 1024,
        "max_seq_length": 512,
        "num_layers": 20,
        "num_heads": 16,
        "mlp_ratio": 4,
        "dropout": 0.1,
    }
    
    # Large: ~500M parameters
    LARGE = {
        "vocab_size": 32000,
        "embed_size": 1280,
        "max_seq_length": 512,
        "num_layers": 24,
        "num_heads": 20,
        "mlp_ratio": 4,
        "dropout": 0.1,
    }
    
    @classmethod
    def get_config(cls, size: str = "small"):
        """Get config by size: 'small', 'medium', or 'large'"""
        if size.lower() == "small":
            return cls.SMALL
        elif size.lower() == "medium":
            return cls.MEDIUM
        elif size.lower() == "large":
            return cls.LARGE
        else:
            raise ValueError(f"Unknown size: {size}. Use 'small', 'medium', or 'large'")


# =============================================================================
# Legacy Compatibility - Keep original classes
# =============================================================================
class FeedForward(nn.Module):
    """Legacy feedforward for compatibility"""
    def __init__(self, embed_size, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Legacy transformer block for compatibility"""
    def __init__(self, embed_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        from .attention import MultiHeadAttention
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout_rate)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffwd = FeedForward(embed_size, dropout_rate)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DAIDecoderOnly(nn.Module):
    """Legacy decoder-only for compatibility"""
    def __init__(self, vocab_size, embed_size, max_seq_length, num_layers, num_heads, dropout_rate=0.1):
        super().__init__()
        from .embedding import DAIEmbeddings
        self.embeddings = DAIEmbeddings(vocab_size, embed_size, max_seq_length, dropout_rate)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight
    
    def forward(self, input_ids, targets=None):
        x = self.embeddings(input_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.reshape(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        return logits, loss


class DAIModel(nn.Module):
    """Legacy DAIModel for compatibility"""
    def __init__(
        self, vocab_size, embed_size, max_seq_length, num_layers, num_heads, 
        dropout_rate=0.1, architecture="decoder"
    ):
        super().__init__()
        
        if architecture == "encoder_decoder":
            # Use decoder-only for now
            self.decoder_only = DAIDecoderOnly(
                vocab_size, embed_size, max_seq_length, num_layers, num_heads, dropout_rate
            )
            self.is_encoder_decoder = False
        else:
            self.decoder_only = DAIDecoderOnly(
                vocab_size, embed_size, max_seq_length, num_layers, num_heads, dropout_rate
            )
            self.is_encoder_decoder = False
        
        self.max_seq_length = max_seq_length
        self.architecture = architecture
    
    def forward(self, input_ids, targets=None, encoder_hidden_states=None):
        return self.decoder_only(input_ids, targets)
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
        return self.decoder_only.generate(input_ids, max_new_tokens, temperature, top_k, top_p)
    
    def gradient_checkpointing_enable(self):
        pass  # Legacy compatibility


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("DAI Optimized Model Testing")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test different sizes
    for size_name in ["small", "medium", "large"]:
        config = DAIConfig.get_config(size_name)
        
        model = OptimizedDAIModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            max_seq_length=config["max_seq_length"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            mlp_ratio=config["mlp_ratio"],
            tie_weights=True
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        
        # Test forward pass
        dummy_input = torch.randint(0, config["vocab_size"], (2, 64)).to(device)
        dummy_targets = torch.randint(0, config["vocab_size"], (2, 64)).to(device)
        
        model = model.to(device)
        logits, loss = model(dummy_input, dummy_targets)
        
        print(f"\n{size_name.upper()} Model ({num_params/1e6:.1f}M params):")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test generation
        model.eval()
        with torch.no_grad():
            gen_ids = model.generate(dummy_input[:, :10], max_new_tokens=20)
        print(f"  Generated: {gen_ids.shape}")
    
    print("\n" + "="*60)
    print("All models working correctly!")
    print("="*60)