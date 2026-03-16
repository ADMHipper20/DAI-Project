# SentencePiece Tokenizer Trainer
# Better than BPE for handling unknown characters and multi-language

import sentencepiece as spm
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_FILE = BASE_DIR / "data" / "training_corpus.txt"
MODEL_PREFIX = str(BASE_DIR / "tokenizer" / "dai_sp")  # Output prefix (as string)

# SentencePiece parameters
VOCAB_SIZE = 32000      # Vocabulary size
CHAR_COVERAGE = 0.9995   # Character coverage
MODEL_TYPE = "bpe"       # BPE model

print("="*60)
print("SentencePiece Tokenizer Training")
print("="*60)

# Check corpus exists
if not CORPUS_FILE.exists():
    print(f"Error: {CORPUS_FILE} not found!")
    print("Run prepare_data.py first to create training corpus.")
    exit(1)

# -----------------------------------------------------------------------------
# TRAIN SENTENCEPIECE
# -----------------------------------------------------------------------------
# Build the training command as a list (required by sentencepiece)
train_args = [
    f"--input={CORPUS_FILE}",
    f"--model_prefix={MODEL_PREFIX}",
    f"--vocab_size={VOCAB_SIZE}",
    f"--character_coverage={CHAR_COVERAGE}",
    f"--model_type={MODEL_TYPE}",
    "--pad_id=0",
    "--unk_id=1",
    "--bos_id=2",
    "--eos_id=3",
    "--pad_piece=[PAD]",
    "--unk_piece=[UNK]",
    "--bos_piece=[BOS]",
    "--eos_piece=[EOS]",
    "--user_defined_symbols=[SYSTEM],[USER],[DAI]",
    "--input_sentence_size=1000000",
    "--shuffle_input_sentence=true",
]

print(f"\nTraining SentencePiece model...")
print(f"Corpus: {CORPUS_FILE}")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Model type: {MODEL_TYPE}")

# Train the model
spm.SentencePieceTrainer.train(train_args)

print(f"\n✓ Tokenizer trained successfully!")
print(f"Model saved to: {MODEL_PREFIX}.model")

# Verify the model
print("\nVerifying tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(MODEL_PREFIX + ".model")

# Test encoding/decoding
test_text = "Hello DAI! How are you today?"
encoded = sp.encode(test_text)
decoded = sp.decode(encoded)

print(f"Test encode: {test_text} -> {encoded}")
print(f"Test decode: {encoded} -> {decoded}")
print(f"Vocab size: {sp.get_piece_size()}")

print("\n" + "="*60)
print("Tokenizer training complete!")
print(f"Files created:")
print(f"  - {MODEL_PREFIX}.model")
print(f"  - {MODEL_PREFIX}.vocab")
print("="*60)
