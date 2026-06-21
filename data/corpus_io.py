"""Corpus I/O utilities for loading conversation datasets."""
import json
from typing import List, Dict

def load_conversation_corpus(path: str) -> List[str]:
    """Load conversations from various corpus formats."""
    conversations = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try JSONL first
    lines = content.strip().split('\n\n')
    for block in lines:
        if '<system>' in block and '<user>' in block and '<DAI>' in block:
            conversations.append(block.strip())
    return conversations

def load_jsonl_corpus(path: str) -> List[Dict]:
    """Load JSONL format conversation data."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    pass
    return samples

def merge_corpora(corpus_paths: List[str]) -> List[str]:
    """Merge multiple corpus files into single conversation list."""
    all_convs = []
    for path in corpus_paths:
        convs = load_conversation_corpus(path)
        all_convs.extend(convs)
    return all_convs