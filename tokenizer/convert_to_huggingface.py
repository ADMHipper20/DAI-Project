# Convert SentencePiece to HuggingFace Tokenizer JSON format
# This allows using it with the HuggingFace tokenizers library

import json
import sentencepiece as spm
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SP_MODEL = BASE_DIR / "tokenizer" / "dai_sp.model"
OUTPUT_JSON = BASE_DIR / "tokenizer" / "dai_tokenizer.json"

print("Converting SentencePiece to HuggingFace format...")

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(str(SP_MODEL))

vocab_size = sp.get_piece_size()

# Build HuggingFace tokenizer JSON
tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 2, "content": "[BOS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 3, "content": "[EOS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 4, "content": "[SYSTEM]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 5, "content": "[USER]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
        {"id": 6, "content": "[DAI]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": True},
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True, "prepend_scheme": "always"},
    "post_processor": None,
    "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True, "prepend_scheme": "always"},
    "model": {
        "type": "SentencePieceBPE",
        "vocab": {},
        "merges": []
    }
}

# Add vocabulary
for i in range(vocab_size):
    piece = sp.id_to_piece(i)
    tokenizer_json["model"]["vocab"][piece] = i

# Add merges (for BPE)
# SentencePiece doesn't provide merges directly, so we create empty ones
# The tokenizer will work but might not be identical to SentencePiece

print(f"Vocabulary size: {vocab_size}")

# Save as JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f, indent=2)

print(f"✓ Saved to: {OUTPUT_JSON}")
print("\nNote: This is a basic conversion. For best results, use SentencePiece directly!")
