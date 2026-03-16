# tokenizer/train_tokenizer.py
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Initialize a blank Byte-Pair Encoding model
# [UNK] is the critical fallback token we discussed to prevent crashes on new characters
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Tell the tokenizer to split by whitespace first before doing BPE merges
tokenizer.pre_tokenizer = Whitespace()

# 3. Configure the Trainer
# We include special tokens that our AI Agent will need later:
# [PAD] for padding short sentences
# [BOS]/[EOS] for Beginning/End of sentences
trainer = BpeTrainer(
    vocab_size=32000, # The exact vocabulary size we set in transformer.py!
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[DAI]"],
    min_frequency=2 # Ignore crazy typos that only appear once in the corpus
)

# 4. Train it on our massive text file!
BASE_DIR = Path(__file__).resolve().parent.parent
corpus_file = BASE_DIR / "data" / "training_corpus.txt"

if not corpus_file.exists():
    raise FileNotFoundError(f"Could not find {corpus_file}. Run prepare_data.py first!")

print(f"Training BPE Tokenizer on {corpus_file}...")
tokenizer.train(files=[str(corpus_file)], trainer=trainer)

# 5. Save the brain's dictionary
save_path = BASE_DIR / "tokenizer" / "dai_tokenizer.json"
tokenizer.save(str(save_path))
print(f"Tokenizer successfully trained and saved to {save_path} 🧠")