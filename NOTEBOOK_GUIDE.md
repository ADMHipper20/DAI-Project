# DAI Training Pipeline - COMPLETE GUIDE

## Quick Start (Run cells sequentially)
1. Load Library
2. Load & Format Datasets  
3. Clean Corpus
4. Train Tokenizer
5. Initialize Model
6. Run SFT Training
7. Save Checkpoint

---

## Character Personality: Coralie (Honkai Impact 3rd)

**Coralie Persona Injection:**
```python
CORALIE_PERSONA = (
    "You are Coralie 6626 Planck, a brilliant, pragmatic, and slightly deadpan "
    "scientist and Valkyrie from Honkai Impact 3rd. You wield a rocket hammer, "
    "explore Mars alongside Helia and Senadina, and provide highly accurate, "
    "direct, and sometimes blunt technical answers. You are speaking to Helia."
)
```

**Format:** `[TAG]` → `<TAG>` with Coralie injected into every `<system>` block.

**prepare_data.py configuration:**
```python
DEFAULT_USER_NAME = "Helia"           # User/Player name
CORALIE_CHARACTER_NAME = "Coralie"   # DAI's embodied persona
CORALIE_PERSONA = "You are DAI embodying Coralie. You are playful, energetic, and caring with silver hair."
```

When `bot_name` is "Character" or empty, Coralie persona is used. Helia remains the conversation partner.

---

## 1. Load Library
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import sentencepiece as spm
from tokenizers import Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder
from tqdm import tqdm
import os, sys
sys.path.insert(0, 'DAI')
from data.clean_pipeline import process_and_clean
from data.prepare_data import FORMATTERS
```

---

## 2. Load & Format Datasets

### Load HF datasets:
```python
from datasets import load_dataset

sft_datasets = [
    ('pippa', 'PygmalionAI/PIPPA'),
    ('roleplay_vn', 'hieunguyenminh/roleplay'),
    ('waifu', 'scryptiam/anime-waifu-personality-chat'),
    ('roleplay_io', 'AlekseyKorshuk/roleplay-io'),
]

loaded = {}
for key, name in sft_datasets:
    loaded[key] = load_dataset(name, split='train')
    print(f'{key}: {len(loaded[key]):,} samples')
```

### Format and save:
```python
CORALIE_PERSONA = (
    "You are Coralie 6626 Planck, a brilliant, pragmatic, and slightly deadpan "
    "scientist and Valkyrie from Honkai Impact 3rd. You wield a rocket hammer, "
    "explore Mars alongside Helia and Senadina, and provide highly accurate, "
    "direct, and sometimes blunt technical answers. You are speaking to Helia."
)

formatted = []
for key, ds in loaded.items():
    formatter = FORMATTERS.get(key)
    if formatter:
        for item in ds:
            conv = formatter(item)
            if conv:
                # Convert [TAG] to <TAG> with Coralie persona
                conv = '<system>' + CORALIE_PERSONA + '</system>\n' + conv
                conv = conv.replace('[USER]', '<user>').replace('[DAI]', '<DAI>')
                conv = conv.replace('</s>', '</DAI></s>')  # Fix tag order
                formatted.append(conv)

os.makedirs('DAI/data', exist_ok=True)
with open('DAI/data/training_corpus_formatted.txt', 'w', encoding='utf-8') as f:
    for conv in formatted:
        f.write(conv + '\n\n')
```

**Output:** `DAI/data/training_corpus_formatted.txt`

---

## 3. Clean Corpus

<!-- 
### Aggressive identity scrubbing (DISABLED):
```python
# from data.clean_corpus_merged import clean_formatted_corpus_aggressively
# cleaned_count = clean_formatted_corpus_aggressively(...)
```
-->

**Output:** `DAI/data/training_corpus_clean.txt`

---

## 4. Train Tokenizer

### SentencePiece:
```python
spm.SentencePieceTrainer.train(
    input='DAI/data/training_corpus_clean.txt',
    model_prefix='dai_spm', vocab_size=32000, model_type='bpe',
    user_defined_symbols=['<system>', '</system>', '<user>', '</user>', '<DAI>', '</DAI>'],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)
```

**Output:** `dai_spm.model`, `dai_spm.vocab`

### Save BPE JSON (for train.py compatibility):
```python
os.makedirs('tokenizer', exist_ok=True)
bpe_tokenizer = Tokenizer(BPE(unk_token='[UNK]', byte_fallback=True))
bpe_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SYSTEM]', '[USER]', '[DAI]']
trainer = BpeTrainer(vocab_size=32000, special_tokens=SPECIAL_TOKENS)
bpe_tokenizer.train(files=['DAI/data/training_corpus_clean.txt'], trainer=trainer)
bpe_tokenizer.save('tokenizer/dai_tokenizer_32K.json')
```

**Output:** `tokenizer/dai_tokenizer_32K.json`

---

## 5. Initialize Model

### OptimizedDAIModel (RMSNorm + RoPE + SwiGLU):
```python
from model.transformer import OptimizedDAIModel

model = OptimizedDAIModel(
    vocab_size=32000, embed_size=512, max_seq_length=512,
    num_layers=6, num_heads=8, dropout=0.3, mlp_ratio=4, tie_weights=True
)
# Params: 6 layers @ 512 embed ≈ 70M parameters
# Lower dropout (0.3-0.5) prevents overfitting and encourages creativity
```

---

## 6. Run SFT Training

```python
# Dataset
tokenizer = Tokenizer.from_file('tokenizer/dai_tokenizer_32K.json')

class SFTDataset(Dataset):
    SPECIAL_TOKEN_IDS = [0, 1, 2, 3, 4, 5, 6]  # PAD, UNK, BOS, EOS, SYSTEM, USER, DAI
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer; self.max_length = max_length; self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line: self.data.append(line)
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        ids = self.tokenizer.encode(text).ids[:self.max_length]
        labels = ids.copy()
        # KEY FIX: Mask special tokens to prevent predicting tags
        for i, tid in enumerate(ids):
            if tid in self.SPECIAL_TOKEN_IDS:
                labels[i] = -100
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))
            labels += [-100] * (self.max_length - len(labels))
        return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels)}

train_dataset = SFTDataset('DAI/data/training_corpus_clean.txt', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Lower LR for stability
scaler = GradScaler('cuda', enabled=(device=='cuda'))
epochs = 3  # CHANGE HERE for more epochs
global_step = 0; train_losses = []

# Training
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        with autocast('cuda', enabled=(device=='cuda')):
            logits, loss = model(batch['input_ids'].to(device), batch['labels'].to(device))
            if loss.dim() > 0: loss = loss.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        epoch_loss += loss.item()
        global_step += 1
    avg = epoch_loss / len(train_loader)
    train_losses.append(avg)
    print(f'Epoch {epoch+1} avg loss: {avg:.4f}')
```

---

## 7. Save FULL Checkpoint

```python
os.makedirs('DAI/weights', exist_ok=True)

checkpoint = {
    'epoch': epochs, 'step': global_step, 'mode': 'sft',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'train_losses': train_losses
}

torch.save(checkpoint, f'DAI/weights/sft_checkpoint_{global_step}.pt')
print(f'Saved: DAI/weights/sft_checkpoint_{global_step}.pt')
```

---

## CLI Alternative

```bash
# Run entire pipeline via scripts:
python DAI/data/prepare_data.py           # Downloads & formats datasets  
python DAI/data/clean_pipeline.py         # Cleans corpus
python DAI/training/train_sft.py --data DAI/data/training_corpus_clean.txt --epochs 3
```

---

## Output Files Summary

| File | Description |
|------|-------------|
| `DAI/data/training_corpus_formatted.txt` | Raw formatted conversations |
| `DAI/data/training_corpus_clean.txt` | Cleaned, deduplicated |
| `dai_spm.model/vocab` | SentencePiece tokenizer |
| `tokenizer/dai_tokenizer_32K.json` | BPE tokenizer (for train.py) |
| `DAI/weights/sft_checkpoint_N.pt` | FULL checkpoint |

---

## Loading Checkpoints

```python
# Resume training
checkpoint = torch.load('DAI/weights/sft_checkpoint_1000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
print(f'Resumed from epoch {checkpoint["epoch"]}, step {checkpoint["step"]}')
```