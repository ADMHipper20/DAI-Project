"""
DAI Unified Training Script
===========================
Supports both SFT (Supervised Fine-tuning) and DPO (Direct Preference Optimization).

Optimizations for RTX 4060 (8GB VRAM):
- FP16 mixed precision training
- INT8 quantization option
- Gradient checkpointing
- Optimized batch sizes
- BPE tokenization (faster than SentencePiece)

Usage:
    # SFT Training
    python train.py --mode sft --data data/training_corpus.txt
    
    # DPO Training  
    python train.py --mode dpo --data data/dpo_training_data.txt
    
    # With custom settings
    python train.py --mode sft --fp16 --batch-size 8 --epochs 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import os
import sys
import json
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import DAIModel
from model.embedding import DAIEmbeddings


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 80000  # Must match tokenizer vocab size
    embed_size: int = 512
    max_seq_length: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    # Architecture options: "decoder_only" (GPT-style) or "encoder_decoder" (T5-style)
    architecture: str = "decoder_only"


@dataclass
class TrainConfig:
    """Training configuration optimized for RTX 4060"""
    # Batch settings (small for 8GB VRAM)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Learning rate
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Training duration
    epochs: int = 3
    save_interval: int = 500
    eval_interval: int = 100
    
    # Mixed precision / Quantization
    use_fp16: bool = True
    use_int8: bool = False
    use_gradient_checkpointing: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DPOConfig:
    """DPO-specific configuration"""
    beta: float = 0.1  # Temperature parameter (lower = more conservative)
    reference_model: bool = True
    label_smoothing: float = 0.0


# =============================================================================
# Dataset Classes
# =============================================================================

class SFTDataset(Dataset):
    """Dataset for Supervised Fine-tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, use_binary: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.use_binary = use_binary
        
        # Check if binary file exists
        binary_path = "data/train_data.bin"
        
        if use_binary and os.path.exists(binary_path):
            print(f"Loading binary data from {binary_path}...")
            # Load pre-encoded tokens from binary file (uint32 for vocab up to 80k)
            self.data = np.fromfile(binary_path, dtype=np.uint32).tolist()
            print(f"Loaded {len(self.data):,} tokens from binary")
            # Calculate number of possible sequences
            self.num_sequences = (len(self.data) - max_length) // 1  # Overlapping sequences
            print(f"Possible training sequences: {self.num_sequences:,}")
        else:
            # Fall back to text loading
            print(f"Loading SFT data from {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(line)
            print(f"Loaded {len(self.data)} SFT samples")
            self.num_sequences = len(self.data)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.use_binary and isinstance(self.data[0], int):
            # Binary mode: sample random sequence of tokens
            if len(self.data) < self.max_length:
                tokens = self.data[:self.max_length]
            else:
                start = idx % (len(self.data) - self.max_length)
                tokens = self.data[start:start + self.max_length]
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(tokens, dtype=torch.long)
            }
        else:
            # Text mode: tokenize on-the-fly
            text = self.data[idx % len(self.data)]
            
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids
            
            # Truncate if needed
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # Pad to max_length
            padding_length = self.max_length - len(ids)
            if padding_length > 0:
                ids = ids + [self.tokenizer.token_to_id("[PAD]")] * padding_length
            
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long)
            }


class DPODataset(Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading DPO data from {data_path}...")
        
        # Read entire file to handle multi-line format
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by the REJECTED marker (multi-line support)
        import re
        # Pattern: CHOSEN <!-- REJECTED: REJECTED_CONTENT -->
        pattern = r'(.*?)<!-- REJECTED:\s*(.*?)\s*-->'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for chosen, rejected in matches:
            chosen = chosen.strip()
            rejected = rejected.strip()
            
            if chosen and rejected:
                self.data.append((chosen, rejected))
        
        print(f"Loaded {len(self.data)} DPO samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        chosen, rejected = self.data[idx]
        
        # Extract prompt from chosen
        prompt_end = chosen.rfind("[DAI]")
        if prompt_end == -1:
            prompt = chosen
            chosen_response = ""
        else:
            prompt = chosen[:prompt_end].strip()
            chosen_response = chosen[prompt_end:].replace("[DAI]", "").strip()
        
        # Handle empty responses
        if not chosen_response:
            chosen_response = " "
        if not rejected:
            rejected = " "
        
        # Tokenize
        max_part = self.max_length // 2
        
        prompt_enc = self.tokenizer.encode(prompt).ids[:max_part]
        chosen_enc = self.tokenizer.encode(chosen_response).ids[:max_part]
        rejected_enc = self.tokenizer.encode(rejected).ids[:max_part]
        
        # Handle empty encoding (add at least one token)
        if len(prompt_enc) == 0:
            prompt_enc = [self.tokenizer.token_to_id("[PAD]")]
        if len(chosen_enc) == 0:
            chosen_enc = [self.tokenizer.token_to_id("[PAD]")]
        if len(rejected_enc) == 0:
            rejected_enc = [self.tokenizer.token_to_id("[PAD]")]
        
        # Pad
        pad_id = self.tokenizer.token_to_id("[PAD]")
        
        prompt_enc = prompt_enc + [pad_id] * (max_part - len(prompt_enc))
        chosen_enc = chosen_enc + [pad_id] * (max_part - len(chosen_enc))
        rejected_enc = rejected_enc + [pad_id] * (max_part - len(rejected_enc))
        
        return {
            "prompt_ids": torch.tensor(prompt_enc, dtype=torch.long),
            "chosen_ids": torch.tensor(chosen_enc, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_enc, dtype=torch.long),
        }


# =============================================================================
# Training Functions
# =============================================================================

def compute_sft_loss(model, batch, device):
    """Compute SFT loss (standard cross-entropy)"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Shift for causal LM
    logits, loss = model(input_ids[:, :-1], labels[:, 1:])
    
    # Ensure loss is scalar (required for backward)
    # Handle multi-GPU case where loss might have extra dimensions
    if loss.dim() > 0:
        loss = loss.mean()
    
    return loss


def compute_dpo_loss(model, batch, device, beta=0.1, reference_model=None):
    """Compute DPO loss"""
    prompt_ids = batch["prompt_ids"].to(device)
    chosen_ids = batch["chosen_ids"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    
    B = prompt_ids.shape[0]
    
    # Safeguard: skip if batch size is 0
    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Concatenate prompt + response
    chosen_full = torch.cat([prompt_ids, chosen_ids], dim=1)
    rejected_full = torch.cat([prompt_ids, rejected_ids], dim=1)
    
    # Get logits from policy model
    chosen_logits, _ = model(chosen_full[:, :-1])
    rejected_logits, _ = model(rejected_full[:, :-1])
    
    # Get logits from reference model
    if reference_model is not None:
        with torch.no_grad():
            ref_chosen_logits, _ = reference_model(chosen_full[:, :-1])
            ref_rejected_logits, _ = reference_model(rejected_full[:, :-1])
    else:
        ref_chosen_logits = chosen_logits.detach()
        ref_rejected_logits = rejected_logits.detach()
    
    # Compute log probabilities
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
    ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)
    
    # Gather for actual tokens
    chosen_tokens = chosen_full[:, 1:]
    rejected_tokens = rejected_full[:, 1:]
    
    chosen_log_prob = torch.gather(chosen_log_probs, -1, chosen_tokens.unsqueeze(-1)).squeeze(-1).mean(-1)
    rejected_log_prob = torch.gather(rejected_log_probs, -1, rejected_tokens.unsqueeze(-1)).squeeze(-1).mean(-1)
    ref_chosen_log_prob = torch.gather(ref_chosen_log_probs, -1, chosen_tokens.unsqueeze(-1)).squeeze(-1).mean(-1)
    ref_rejected_log_prob = torch.gather(ref_rejected_log_probs, -1, rejected_tokens.unsqueeze(-1)).squeeze(-1).mean(-1)
    
    # DPO loss
    policy_diff = (chosen_log_prob - rejected_log_prob)
    ref_diff = (ref_chosen_log_prob - ref_rejected_log_prob)
    
    loss = -F.logsigmoid(beta * (policy_diff - ref_diff)).mean()
    
    # Ensure loss is scalar (required for backward)
    # Handle multi-GPU case where loss might have extra dimensions
    if loss.dim() > 0:
        loss = loss.mean()
    
    return loss


def train_sft(
    model,
    train_loader,
    optimizer,
    scaler,
    config: TrainConfig,
    device
):
    """SFT training loop"""
    model.train()
    total_loss = 0
    global_step = 0
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"SFT Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mixed precision forward pass
            with autocast(device_type='cuda', enabled=config.use_fp16):
                loss = compute_sft_loss(model, batch, device)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            # Handle multi-GPU loss
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss_val = loss.mean().item()
            else:
                loss_val = loss.item()
            epoch_loss += loss_val * config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            # Save checkpoint
            if global_step % config.save_interval == 0 and global_step > 0:
                save_checkpoint(model, optimizer, scaler, epoch, global_step, "sft")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)


def train_dpo(
    model,
    train_loader,
    optimizer,
    scaler,
    config: TrainConfig,
    dpo_config: DPOConfig,
    device
):
    """DPO training loop"""
    model.train()
    
    # Create reference model (frozen copy)
    reference_model = None
    if dpo_config.reference_model:
        import copy
        # Get the underlying module (unwrap DataParallel if present)
        model_to_copy = model.module if isinstance(model, nn.DataParallel) else model
        reference_model = copy.deepcopy(model_to_copy)
        reference_model.requires_grad_(False)
        reference_model = reference_model.to(device)
        reference_model.eval()
        print("Reference model created (frozen)")
    
    total_loss = 0
    global_step = 0
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"DPO Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mixed precision forward pass
            with autocast(device_type='cuda', enabled=config.use_fp16):
                loss = compute_dpo_loss(
                    model, batch, device,
                    beta=dpo_config.beta,
                    reference_model=reference_model
                )
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            # Handle multi-GPU loss
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss_val = loss.mean().item()
            else:
                loss_val = loss.item()
            epoch_loss += loss_val * config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            # Save checkpoint
            if global_step % config.save_interval == 0 and global_step > 0:
                save_checkpoint(model, optimizer, scaler, epoch, global_step, "dpo")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average DPO Loss: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)


def save_checkpoint(model, optimizer, scaler, epoch, step, mode):
    """Save training checkpoint"""
    os.makedirs("DAI/weights", exist_ok=True)
    
    # Handle DataParallel state_dict (remove 'module.' prefix)
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "mode": mode,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    path = f"DAI/weights/{mode}_checkpoint_{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scaler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DAI Training Script")
    
    # Multi-GPU
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of GPUs to use (0=auto-detect)")
    
    # Mode
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "dpo"],
                        help="Training mode: sft (Supervised Fine-tuning) or dpo (Direct Preference Optimization)")
    
    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/dai_tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--use-binary", action="store_true", default=True,
                        help="Use pre-encoded binary data (train_data.bin)")
    
    # Model architecture
    parser.add_argument("--architecture", type=str, default="decoder_only",
                        choices=["decoder_only", "encoder_decoder"],
                        help="Model architecture")
    parser.add_argument("--embed-size", type=int, default=512,
                        help="Embedding size")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    # Optimization
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantization (bitsandbytes)")
    parser.add_argument("--int4", action="store_true",
                        help="Use INT4 quantization (bitsandbytes, requires load_in_4bit)")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    
    # DPO specific
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    
    print("="*60)
    print("DAI Training Configuration")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Architecture: {args.architecture}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embed size: {args.embed_size}")
    print(f"Layers: {args.num_layers}")
    print(f"Heads: {args.num_heads}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.fp16}")
    print(f"INT8: {args.int8}")
    if args.mode == "dpo":
        print(f"DPO Beta: {args.beta}")
    print("="*60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        max_seq_length=args.max_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        architecture=args.architecture
    )
    
    model = DAIModel(
        vocab_size=model_config.vocab_size,
        embed_size=model_config.embed_size,
        max_seq_length=model_config.max_seq_length,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout_rate=model_config.dropout,
        architecture=model_config.architecture
    )
    
    # Apply quantization for laptop/inference
    if args.int8 or args.int4:
        try:
            from bitsandbytes import nn
            import bitsandbytes as bnb
            
            if args.int4:
                # INT4 quantization - requires load_in_4bit
                print("⚡ Loading model with INT4 quantization...")
                # Convert model to 4-bit
                model = model.to(device)
                for param in model.parameters():
                    param.requires_grad = False
                
                # Use LLM.int4 for inference (not training)
                # For training with INT4, we need special handling
                print("INT4 mode: Note - Full INT4 training requires bitsandbytes. Using FP16 for now.")
                print("For pure INT4 inference on laptop, use separate inference script.")
            else:
                # INT8 quantization
                print("⚡ Loading model with INT8 quantization...")
                # Convert Linear layers to Int8Linear
                model = model.to(device)
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        module = nn.Int8Linear(
                            module.in_features,
                            module.out_features,
                            module.bias is not None
                        )
                print("INT8 quantization applied!")
        except ImportError:
            print("⚠️ bitsandbytes not installed. Install with: pip install bitsandbytes")
            print("Falling back to FP16...")
    else:
        # Enable gradient checkpointing for memory savings
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    
    model = model.to(device)
    
    # Multi-GPU setup
    num_gpus = args.gpus if args.gpus > 0 else torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n🚀 MULTI-GPU DETECTED: {num_gpus} GPUs!")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model)
        print("Using DataParallel for multi-GPU training")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create dataset
    if args.mode == "sft":
        train_dataset = SFTDataset(args.data, tokenizer, args.max_length, args.use_binary)
    else:
        train_dataset = DPODataset(args.data, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Optimizer
    if args.int8:
        try:
            from bitsandbytes.optim import PagedAdamW32bit
            optimizer = PagedAdamW32bit(model.parameters(), lr=args.lr)
            print("Using INT8-optimized PagedAdamW32bit")
        except ImportError:
            print("bitsandbytes not available, using AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if args.fp16 else None
    
    # Training config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_fp16=args.fp16,
        use_int8=args.int8,
        use_gradient_checkpointing=not args.no_gradient_checkpointing
    )
    
    dpo_config = DPOConfig(beta=args.beta)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        load_checkpoint(model, args.resume, optimizer, scaler)
    
    # Train
    print("\nStarting training...")
    if args.mode == "sft":
        train_sft(model, train_loader, optimizer, scaler, train_config, device)
    else:
        train_dpo(model, train_loader, optimizer, scaler, train_config, dpo_config, device)
    
    # Save final model
    print("\nSaving final model...")
    os.makedirs("DAI/weights", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "embed_size": args.embed_size,
            "max_seq_length": args.max_length,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "architecture": args.architecture
        }
    }, "DAI/weights/dai_final.pt")
    
    print("Training complete!")
    print(f"Model saved to DAI/weights/dai_final.pt")


if __name__ == "__main__":
    main()
