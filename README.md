# DAI 🧠

A custom AI Agent built entirely from scratch by **Erdős Helia** (Helia). 

This project explores the core mechanics of Generative AI and NLP by manually constructing and training a Transformer-based model without relying on external wrapper APIs (like OpenAI or OpenRouter).

## ⚙️ Hardware Requirements
Building and training an LLM locally requires specific resources. This project is optimized for the following baseline:
* **GPU:** NVIDIA RTX 4060 (8GB VRAM) - *Mixed precision training highly recommended.*
* **RAM:** 16GB minimum.
* **Storage:** 50GB+ free space (SSD strongly preferred for fast dataset loading and model checkpointing).

## 🛠️ Tech Stack & Dependencies
* **Environment:** Python 3.10
* **Core Libraries:**
  * `torch` (PyTorch with CUDA support) - Core neural network framework.
  * `numpy` - Matrix operations.
  * `tokenizers` / `sentencepiece` - Custom text-to-token pipelines.
  * `accelerate` & `bitsandbytes` - Crucial for optimizing memory usage and enabling 8-bit/16-bit training on 8GB VRAM.
  * `datasets` - For fetching and managing training corpora.
  * `tqdm` - Progress tracking.

## 🚀 Initialization
To start working on this project locally, set up your environment:

```bash
# Initialize project directory
mkdir DAI
cd DAI

# Set up a virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate
# OR (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
pip install numpy tokenizers accelerate bitsandbytes datasets tqdm

# Architecture Overview
Dataset
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding Layer
   ↓
Transformer Blocks
   ↓
Language Model Head
   ↓
Text Generation

# Project Structure
DAI/
│
├── data/
│   └── training_corpus.txt
│
├── tokenizer/
│   └── train_tokenizer.py
│
├── model/
│   ├── transformer.py
│   ├── attention.py
│   └── embedding.py
│
├── training/
│   └── train.py
│
├── inference/
│   └── generate.py
│
└── README.md

# Training Pipeline
1. Prepare dataset
2. Train tokenizer
3. Encode dataset
4. Initialize transformer model
5. Train with cross entropy loss
6. Save checkpoints
7. Run inference

# AI Agent Layer
Agent Components:

- Memory module
- Planning module
- Tool usage
- Response generator