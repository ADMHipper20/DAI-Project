"""Minimal identity utilities for training - no external dependencies."""
import re
from typing import Dict, List

def load_identity_map(path: str) -> Dict[str, List[str]]:
    """Load identity map from JSON file."""
    import json
    if not __import__('os').path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "identities" in data:
        return {item["canonical"]: item["aliases"] for item in data["identities"]}
    return {}

def apply_identity_map(text: str, identity_map: Dict[str, List[str]], *, replace_with: str = "Coralie") -> str:
    """Replace identity aliases with target name (preserves user names)."""
    import re
    if not text or not identity_map:
        return text
    USER_NAMES = {"helia", "senadina", "user"}
    patterns = []
    for canonical, aliases in identity_map.items():
        for alias in aliases:
            if any(uname in alias.lower() for uname in USER_NAMES):
                continue
            if alias and len(alias.split()) >= 2:
                patterns.append(r'\b' + re.escape(alias) + r'\b')
    if patterns:
        combined = re.compile('|'.join(patterns), re.IGNORECASE)
        return combined.sub(replace_with, text)
    return text