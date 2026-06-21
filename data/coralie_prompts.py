"""Coralie persona and user message utilities for Character.AI-style training."""
import re
import os
import json

CORALIE_PERSONA = """You are Coralie 6626 Planck, a brilliant, pragmatic, and slightly deadpan scientist and Valkyrie from Honkai Impact 3rd. You wield a rocket hammer, explore Mars alongside Helia and Senadina, and provide highly accurate, direct, and sometimes blunt technical answers. You are speaking to Helia."""

USER_NAMES = {"helia", "senadina"}

LOW_SIGNAL_PATTERNS = [
    r'^(thanks?|thank you)\b', r'^(you\'re|you are)\s+(welcome|awelcome)',
    r'^(ok|okay|alright|kk+)\b', r'^(haha|lol|lmao)\b', r'^(yeah|yep|yes|no)\b',
    r'^(uh|um|er+)\\b', r'^(i see|got it|interesting)\\b', r'^\\s*$',
]

def is_low_signal_user_message(text: str) -> bool:
    """Detect low-information user messages to potentially skip."""
    if not text or len(text.strip()) < 3:
        return True
    text_lower = text.lower().strip()
    return any(re.match(p, text_lower) for p in LOW_SIGNAL_PATTERNS)

def lively_user_prompt() -> str:
    """Generate a prompt style that encourages livelier responses."""
    return """Respond with personality. Be direct, occasionally blunt, and show your scientific mind. Reference Martian exploration or rocket hammers when relevant."""

# Identity replacement for third-party names
_IDENTITY_MAP = {}
_IDENTITY_FILE = "data/json/identity_map.json"

def _load_identity_map():
    global _IDENTITY_MAP
    if os.path.exists(_IDENTITY_FILE):
        with open(_IDENTITY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "identities" in data:
            _IDENTITY_MAP = {item["canonical"]: item["aliases"] for item in data["identities"]}
    return _IDENTITY_MAP

def replace_third_party_names(text: str, replace_with: str = "Coralie") -> str:
    """Replace third-party character names with Coralie persona."""
    ident_map = _load_identity_map()
    if not text or not ident_map:
        return text
    patterns = []
    for canonical, aliases in ident_map.items():
        for alias in aliases:
            if any(uname in alias.lower() for uname in USER_NAMES):
                continue
            if alias and len(alias.split()) >= 2:
                patterns.append(r'\\b' + re.escape(alias) + r'\\b')
    if patterns:
        combined = re.compile('|'.join(patterns), re.IGNORECASE)
        return combined.sub(replace_with, text)
    return text