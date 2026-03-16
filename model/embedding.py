import torch
import torch.nn as nn

class DAIEmbeddings(nn.Module):
    """
    The foundational input layer for the DAI Agent.
    Converts raw token IDs into dense, position-aware mathematical vectors.
    """
    def __init__(self, vocab_size, embed_size, max_seq_length, dropout_rate=0.1):
        super().__init__()
        
        # 1. Token Embedding: Translates vocabulary IDs to vectors of size 'embed_size'
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        # 2. Positional Embedding: Translates sequence indices (0, 1, 2...) to vectors
        # Note: We use learned positional embeddings here (standard in GPT models)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=embed_size)
        
        # 3. Dropout: Randomly zeroes some elements to prevent the model from memorizing the data
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing token IDs.
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # Generate position IDs [0, 1, ..., seq_length - 1]
        # We ensure it sits on the exact same device (CUDA/RTX 4060) as the input data
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        
        # Expand positions to match the batch size: (1, sequence_length) -> (batch_size, sequence_length)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        
        # Extract the dense vectors
        # Shape becomes: (Batch, Sequence, Embed_Size)
        tok_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        
        # The magic: Combine 'what the word is' with 'where the word is'
        embeddings = tok_embeds + pos_embeds
        
        return self.dropout(embeddings)

# --- QUICK TEST (You can run this at the bottom of the script or in a notebook) ---
if __name__ == "__main__":
    # Mock parameters for our 8GB VRAM constraint
    VOCAB_SIZE = 32000  # Standard BPE vocab size
    EMBED_SIZE = 512    # Dimensionality of the vectors
    MAX_LEN = 1024      # Context window limit
    
    # Initialize the layer
    embed_layer = DAIEmbeddings(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, max_seq_length=MAX_LEN)
    
    # Create a dummy batch of token IDs (Batch Size: 4, Sequence Length: 12)
    dummy_input = torch.randint(0, VOCAB_SIZE, (4, 12))
    
    # Pass it through the layer
    output = embed_layer(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}  -> (Batch, Sequence)")
    print(f"Output shape: {output.shape} -> (Batch, Sequence, Embed_Size)")
    print("\nIf output is [4, 12, 512], the embedding layer is perfectly configured!")