# Files to Upload to Kaggle

## Folder Structure to Create on Kaggle:

```
DAI/
├── model/
│   ├── __init__.py
│   ├── transformer.py
│   ├── attention.py
│   └── embedding.py
├── tokenizer/
│   └── dai_tokenizer.json   # Your trained tokenizer
├── data/
│   └── training_corpus.txt   # Your training data
├── training/
│   └── kaggle_train.py       # THIS FILE
├── inference/
│   └── chat.py               # For testing
└── weights/                  # (empty - will save here)
```

## Quick Setup on Kaggle:

1. **Create new Notebook** on Kaggle
2. **Upload these files** (drag & drop or use "+ Add Data"):
   - `model/` folder (all 4 files)
   - `tokenizer/dai_tokenizer.json`
   - `data/training_corpus.txt`
   - `training/kaggle_train.py`

3. **Install dependencies** (in Kaggle cell):
```python
!pip install torch tokenizers tqdm
```

4. **Run training**:
```python
%cd DAI
%run training/kaggle_train.py
```

## Hardware Settings:

| GPU | layers | embed | batch_size |
|-----|--------|-------|------------|
| T4 (Kaggle) | 12 | 768 | 32 |
| P100 | 16 | 1024 | 48 |
| V100 | 20 | 1280 | 64 |

Edit `MODEL_CONFIG` in `kaggle_train.py` to match your GPU!

## After Training:

Download `weights/dai_kaggle.pt` and upload to your local:
```
DAI/weights/dai_kaggle.pt
```

Then run locally:
```bash
cd DAI/inference
python chat.py
```
