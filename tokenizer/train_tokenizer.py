"""
DAI BPE Tokenizer Trainer
=========================
Trains a Byte-Pair Encoding tokenizer optimized for conversation data.
Uses BPE (not SentencePiece) as requested for shorter token outputs.

Usage:
    python tokenizer/train_tokenizer.py
"""

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_FILE = BASE_DIR / "data" / "training_corpus.txt"
OUTPUT_FILE = BASE_DIR / "tokenizer" / "dai_tokenizer.json"

# Tokenizer hyperparameters
VOCAB_SIZE = 80000  # Standard size for efficient inference
MIN_FREQUENCY = 3   # Minimum frequency for a token to be included
MAX_TOKEN_LENGTH = 80  # Maximum length of a single token

# Special tokens - order matters!
SPECIAL_TOKENS = [
    "[PAD]",  # Padding (id=0)
    "[UNK]",  # Unknown (id=1) 
    "[BOS]",  # Beginning of sequence (id=2)
    "[EOS]",  # End of sequence (id=3)
    "[SYSTEM]", # System prompt (id=4)
    "[USER]",   # User message (id=5)
    "[DAI]",    # AI response (id=6)
]

# =============================================================================
# TRAIN TOKENIZER
# =============================================================================
def train_tokenizer():
    """Train the BPE tokenizer"""
    
    # Check if corpus exists
    if not CORPUS_FILE.exists():
        print(f"ERROR: {CORPUS_FILE} not found!")
        print("\nPlease run prepare_data.py first to create the training corpus.")
        print("Command: python data/prepare_data.py")
        return False
    
    # Get file size
    file_size = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
    print(f"Training corpus: {CORPUS_FILE}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Target vocab size: {VOCAB_SIZE}")
    print()
    
    # Initialize BPE model with unknown token
    print("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Use whitespace pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer with special tokens
    print("Configuring trainer...")
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
    )
    
    # Train the tokenizer
    print("Training tokenizer (this may take a few minutes)...")
    tokenizer.train(files=[str(CORPUS_FILE)], trainer=trainer)
    
    # Add decoder for better output
    tokenizer.decoder = BPEDecoder(suffix="</w>")
    
    # Save the tokenizer
    print(f"Saving tokenizer to {OUTPUT_FILE}...")
    tokenizer.save(str(OUTPUT_FILE))
    
    # Verify
    print("\nVerifying tokenizer...")
    test_tokenizer = Tokenizer.from_file(str(OUTPUT_FILE))
    
    # Test encoding
    test_texts = [
        "[SYSTEM] You are DAI.",
        "[USER] Hello! How are you?",
        "[DAI] I'm doing great, thanks!"
    ]
    
    print("\nTest encodings:")
    for text in test_texts:
        ids = test_tokenizer.encode(text).ids
        decoded = test_tokenizer.decode(ids)
        print(f"  '{text[:30]}...' -> {len(ids)} tokens")
    
    # Print vocabulary stats
    vocab = test_tokenizer.get_vocab()
    print(f"\nTokenizer trained successfully!")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Output file: {OUTPUT_FILE}")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("DAI BPE Tokenizer Training")
    print("="*60)
    print()
    
    success = train_tokenizer()
    
    if success:
        print("\n" + "="*60)
        print("Tokenizer training complete!")
        print("="*60)
    else:
        print("\nTraining failed. Please check the errors above.")
