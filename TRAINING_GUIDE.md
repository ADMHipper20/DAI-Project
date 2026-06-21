# DAI Training Guide

Complete step-by-step instructions for training DAI on your RTX 4060 (8GB VRAM).

---

## Quick Start Commands

### 1. SFT Training (Supervised Fine-tuning)

```bash
# Basic SFT training (text mode - recommended)
python DAI/training/train.py --mode sft --data DAI/data/training_corpus.txt

# SFT with custom settings (recommended for RTX 4060)
python DAI/training/train.py \
    --mode sft \
    --data DAI/data/training_corpus.txt \
    --batch-size 8 \
    --epochs 3 \
    --lr 1e-4 \
    --embed-size 512 \
    --num-layers 8 \
    --num-heads 8 \
    --max-length 512 \
    --fp16
```

### 2. DPO Training (Direct Preference Optimization)

```bash
# Basic DPO training
python DAI/training/train.py --mode dpo --data DAI/data/dpo_training_data.txt

# DPO with custom settings
python DAI/training/train.py \
    --mode dpo \
    --data DAI/data/dpo_training_data.txt \
    --batch-size 4 \
    --epochs 3 \
    --lr 1e-5 \
    --beta 0.1
```

---

## Full Command Reference

### Required Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `--mode` | Training mode: `sft` or `dpo` | `sft` |
| `--data` | Path to training data file | `DAI/data/training_corpus.txt` |

### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--architecture` | `decoder_only` | `decoder_only` (GPT-style) or `encoder_decoder` (T5-style) |
| `--embed-size` | 512 | Embedding dimension (512-1024 for RTX 4060) |
| `--num-layers` | 6 | Number of transformer layers (6-12) |
| `--num-heads` | 8 | Number of attention heads |
| `--max-length` | 512 | Maximum sequence length |
| `--tokenizer` | `tokenizer/dai_tokenizer_32K.json` | Path to tokenizer |

### Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 8 | Batch size (reduce to 4 if OOM) |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate (use 1e-5 for DPO) |
| `--beta` | 0.1 | DPO beta parameter |
| `--max-samples` | 100000 | Max samples per epoch |

### Data Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--use-binary` | `False` | Use binary data (text mode is more reliable) |

### Optimization Options
| Argument | Description |
|----------|-------------|
| `--fp16` | Use FP16 mixed precision (recommended) |
| `--int8` | Use INT8 quantization |
| `--no-gradient-checkpointing` | Disable gradient checkpointing |

### Other Options
| Argument | Description |
|----------|-------------|
| `--resume PATH` | Resume from checkpoint |
| `--gpus N` | Number of GPUs to use (0=auto) |

---

## Complete Training Examples

### Example 1: Small Model (137M params) - Fast Training
```bash
python DAI/training/train.py \
    --mode sft \
    --data DAI/data/training_corpus.txt \
    --batch-size 16 \
    --epochs 3 \
    --embed-size 384 \
    --num-layers 6 \
    --num-heads 6 \
    --max-length 512 \
    --fp16
```

### Example 2: Medium Model (368M params) - Balanced
```bash
python DAI/training/train.py \
    --mode sft \
    --data DAI/data/training_corpus.txt \
    --batch-size 8 \
    --epochs 3 \
    --embed-size 768 \
    --num-layers 8 \
    --num-heads 12 \
    --max-length 512 \
    --fp16
```

### Example 3: Large Model (670M params) - Best Quality
```bash
python DAI/training/train.py \
    --mode sft \
    --data DAI/data/training_corpus.txt \
    --batch-size 4 \
    --epochs 3 \
    --embed-size 1024 \
    --num-layers 10 \
    --num-heads 16 \
    --max-length 512 \
    --fp16 \
    --gradient-checkpointing
```

---

## Training Pipeline

### Step 1: Prepare Data
```bash
python DAI/data/prepare_data.py
```

### Step 2: Train/Update Tokenizer (if needed)
```bash
# Option A: Byte-Level BPE (recommended - handles all characters)
python DAI/tokenizer/train_tokenizer.py

# Option B: Tiktoken (OpenAI's GPT-4 tokenizer)
python DAI/tokenizer/train_tokenizer_tiktoken.py
```

### Step 3: Encode Data
```bash
python DAI/training/encode_data.py
```

### Step 4: Train Model
```bash
# SFT Training
python DAI/training/train.py --mode sft --data DAI/data/training_corpus.txt

# DPO Training
python DAI/training/train.py --mode dpo --data DAI/data/dpo_training_data.txt
```

---

## Model Size Reference

| Config | Params | Layers | Embed | Heads | RTX 4060 |
|--------|--------|--------|-------|-------|----------|
| SMALL  |  137M  |   6    |  384  |   6   |  ✅ Fast |
| MEDIUM |  368M  |   8    |  768  |  12   |  ✅ Good |
| LARGE  |  670M  |  10    |  1024 |  16   |  ⚠️ Slow |

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` to 4 or 2
- Enable gradient checkpointing (default is ON)
- Use `--fp16` (default is ON)
- Reduce `--max-length` to 256

### Slow Training
- Reduce `--num-layers` and `--embed-size`
- Use smaller batch size with gradient accumulation
- Check if other processes are using GPU

### DPO Errors ("cannot unpack" or "DataLoader worker")
- The training script now uses text mode by default (more reliable)
- Make sure your DPO data file exists: `DAI/data/dpo_training_data.txt`

### Checkpoints
Checkpoints are saved to: `DAI/weights/`
- Format: `{mode}_checkpoint_{step}.pt`
- Final model: `DAI/weights/dai_final.pt`

---

## Inference After Training

### Run Chat
```bash
cd DAI
python inference/chat.py
```

Or from project root:
```bash
python DAI/inference/chat.py
```

### Load Model Directly
```python
import torch
from DAI.model.transformer import DAIModel
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("DAI/tokenizer/dai_tokenizer_32K.json")

# Load checkpoint
checkpoint = torch.load("DAI/weights/dai_final.pt")

# Get config
config = checkpoint.get('config', {})
vocab_size = config.get('vocab_size', 32000)
embed_size = config.get('embed_size', 512)
num_layers = config.get('num_layers', 8)
num_heads = config.get('num_heads', 8)
max_seq_length = config.get('max_seq_length', 512)

# Create and load model
model = DAIModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    max_seq_length=max_seq_length,
    num_layers=num_layers,
    num_heads=num_heads
)
model.load_state_dict(checkpoint["model_state_dict"])
```
