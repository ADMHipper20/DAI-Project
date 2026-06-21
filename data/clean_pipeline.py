import os
import re
from typing import List, Dict

IDENTITY_FILE = "data/json/identity_map.json"
_IDENTITY_MAP = {}

def load_identity_map():
    global _IDENTITY_MAP
    if not _IDENTITY_MAP and os.path.exists(IDENTITY_FILE):
        import json
        with open(IDENTITY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "identities" in data:
            _IDENTITY_MAP = {item["canonical"]: item["aliases"] for item in data["identities"]}
    return _IDENTITY_MAP

def set_identity_map(path=IDENTITY_FILE):
    global IDENTITY_FILE
    IDENTITY_FILE = path

USER_NAMES = {"helia", "senadina", "user"}

def replace_third_party_names(text: str, replace_with: str = "Coralie") -> str:
    ident_map = load_identity_map()
    if not text or not ident_map:
        return text
    patterns = []
    for canonical, aliases in ident_map.items():
        for alias in aliases:
            if any(uname in alias.lower() for uname in USER_NAMES):
                continue
            if alias and len(alias.split()) >= 2:
                patterns.append(r'\b' + re.escape(alias) + r'\b')
    if patterns:
        combined = re.compile('|'.join(patterns), re.IGNORECASE)
        return combined.sub(replace_with, text)
    return text

CORALIE_PERSONA = "You are Coralie 6626 Planck, a brilliant, pragmatic, and slightly deadpan scientist and Valkyrie from Honkai Impact 3rd. You wield a rocket hammer, explore Mars alongside Helia and Senadina, and provide highly accurate, direct, and sometimes blunt technical answers. You are speaking to Helia."

def format_pippa(samples: List[Dict]) -> List[str]:
    convs = []
    for sample in samples:
        conversation = sample.get("conversation", [])
        if not conversation or len(conversation) < 2:
            continue
        turns = [(t["message"].strip(), t["is_human"]) for t in conversation if "is_human" in t and "message" in t]
        user_turns = [(m, i) for i, (m, h) in enumerate(turns) if h]
        if len(user_turns) == 0:
            continue
        for msg, idx in user_turns[:1]:
            if idx + 1 < len(turns):
                next_msg, _ = turns[idx + 1]
                next_msg = replace_third_party_names(next_msg)
                convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>{msg}</user>\n<DAI>{next_msg}</DAI>")
    return convs

def show_dataset_info(ds, name: str) -> str:
    info = f"\n{name}:\n"
    if hasattr(ds, '__len__'):
        info += f"  Size: {len(ds)}\n"
    if hasattr(ds, 'column_names'):
        info += f"  Columns: {ds.column_names}\n"
    if len(ds) > 0:
        sample = ds[0]
        info += f"  Sample keys: {list(sample.keys())}\n"
        for k, v in sample.items():
            if isinstance(v, str):
                preview = v[:100].replace('\n', ' ') + "..."
                info += f"  {k}: {preview}\n"
    return info

def format_roleplay_vn(ds) -> List[str]:
    convs = []
    for item in ds:
        text = item.get("text", "")
        if not text:
            continue
        text = re.sub(r'<\|system\|>.*?</s>', '', text, flags=re.DOTALL)
        text = re.sub(r'</s>', '', text)
        # Parse roleplay format: <|user|> and <|assistant|>
        user_content = ""
        parts = re.split(r'<\|(user|assistant)\|>', text)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Check previous part for tag
            prev_tag_idx = i - 1
            while prev_tag_idx >= 0 and not parts[prev_tag_idx]:
                prev_tag_idx -= 1
            if i > 0 and prev_tag_idx >= 0 and prev_tag_idx % 2 == 1:
                prev_tag = parts[prev_tag_idx]
                if prev_tag == "user":
                    user_content += f"<user>{part}</user>\n"
                elif prev_tag == "assistant":
                    part = replace_third_party_names(part)
                    user_content += f"<DAI>{part}</DAI>"
        if user_content:
            convs.append(f"<system>{CORALIE_PERSONA}</system>\n{user_content.strip()}")
    return convs

def format_waifu(ds) -> List[str]:
    convs = []
    for item in ds:
        dai_response = item.get("dialogue") or ""
        if dai_response:
            dai_response = replace_third_party_names(dai_response)
            convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>Hello!</user>\n<DAI>{dai_response.strip()}</DAI>")
    return convs

def format_no_robots(ds) -> List[str]:
    convs = []
    for item in ds:
        messages = item.get("messages", [])
        user_msg = ""
        dai_msg = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_msg = content
            elif role == "assistant" and user_msg:
                dai_msg = replace_third_party_names(content)
                convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>{user_msg}</user>\n<DAI>{dai_msg}</DAI>")
                user_msg = ""
                dai_msg = ""
    return convs

def format_openhermes(ds) -> List[str]:
    convs = []
    for item in ds:
        conversations = item.get("conversations", [])
        user_msg = ""
        dai_msg = ""
        for conv in conversations:
            from_human = conv.get("from", "") == "human"
            value = conv.get("value", "")
            if from_human:
                user_msg = value
            elif not from_human and user_msg:
                dai_msg = replace_third_party_names(value)
                convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>{user_msg}</user>\n<DAI>{dai_msg}</DAI>")
                user_msg = ""
                dai_msg = ""
    return convs

def format_physics(ds) -> List[str]:
    convs = []
    for item in ds:
        question = item.get("message_1", "") or item.get("question", "")
        answer = item.get("message_2", "") or item.get("answer", "")
        if question and answer:
            answer = replace_third_party_names(answer)
            convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>{question}</user>\n<DAI>{answer}</DAI>")
    return convs

def format_physics_json(folder_path: str) -> List[str]:
    convs = []
    import json
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            if fname.endswith('.json'):
                fpath = os.path.join(folder_path, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        item = json.load(f)
                    question = item.get("message_1", "") or item.get("question", "")
                    answer = item.get("message_2", "") or item.get("answer", "")
                    if question and answer:
                        answer = replace_third_party_names(answer)
                        convs.append(f"<system>{CORALIE_PERSONA}</system>\n<user>{question}</user>\n<DAI>{answer}</DAI>")
                except Exception:
                    pass
    return convs

def load_pippa_local() -> List[Dict]:
    import urllib.request
    import json
    url = 'https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa_deduped.jsonl'
    cache_path = 'data/pippa_deduped.jsonl'
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(cache_path):
        print(f'Downloading PIPPA dataset...')
        urllib.request.urlretrieve(url, cache_path)
    
    samples = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 300:
                break
            entry = json.loads(line)
            samples.append({
                'bot_name': entry.get('bot_name', 'Character'),
                'bot_greeting': entry.get('bot_greeting', ''),
                'bot_description': entry.get('bot_description', ''),
                'conversation': entry.get('conversation', [])
            })
    return samples