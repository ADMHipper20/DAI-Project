"""
DAI SFT Training Script
=======================
Supervised Fine-tuning for DAI with anti-overfitting measures.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import os
import sys
import json
from tqdm import tqdm
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import OptimizedDAIModel


@dataclass
class TrainConfig:
    batch_size: int = 16  # Reduced from 32 to prevent OOM
    epochs: int = 3
    learning_rate: float = 3e-5  # Reduced for stability with low vocab (27K)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    use_fp16: bool = True
    dropout: float = 0.15  # Increased from 0.1 to prevent overfitting


class SFTDataset(Dataset):
    SPECIAL_TOKEN_IDS = [0, 1, 2, 3, 4, 5, 6]  # PAD, UNK, BOS, EOS, SYSTEM, USER, DAI
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        ids = self.tokenizer.encode(text).ids[:self.max_length]
        labels = ids.copy()
        # KEY FIX: Mask special tokens to prevent model from predicting tags
        for i, tid in enumerate(ids):
            if tid in self.SPECIAL_TOKEN_IDS:
                labels[i] = -100
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))
            labels += [-100] * (self.max_length - len(labels))
        return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels)}


def save_checkpoint(model, optimizer, scaler, epoch, step, train_losses, path):
    """Save full checkpoint with all training state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else {},
        'scaler_state_dict': scaler.state_dict() if scaler else {},
        'train_losses': train_losses
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training_corpus_clean_v2.txt')
    parser.add_argument('--tokenizer', default='tokenizer/dai_tokenizer_32K.json')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.15)
    args = parser.parse_args()
    
    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    
    # Create model
    model = OptimizedDAIModel(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        max_seq_length=512,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        tie_weights=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params on {device}")
    
    # Dataset
    train_dataset = SFTDataset(args.data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training samples: {len(train_dataset)}")
    
    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler('cuda', enabled=args.use_fp16 and device == 'cuda')
    
    # Training
    model.train()
    train_losses = []
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            with autocast('cuda', enabled=args.use_fp16 and device == 'cuda'):
                logits, loss = model(
                    batch['input_ids'].to(device),
                    batch['labels'].to(device)
                )
                if loss.dim() > 0:
                    loss = loss.mean()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Reduce LR on plateau
        if avg_loss > best_loss:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.9
            print(f"Reduced LR to {optimizer.param_groups[0]['lr']:.2e}")
        else:
            best_loss = avg_loss
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scaler, epoch + 1, global_step, train_losses,
                       f"DAI/weights/sft_checkpoint_{global_step}.pt")
    
    print(f"Training complete. Final loss: {train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()