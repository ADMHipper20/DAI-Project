import os
import random
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. CONFIGURATION - DPO Datasets for AI Agentic Responses
# -----------------------------------------------------------------------------

DATASETS_CONFIG = {
    # === DPO DATASETS (Priority) ===
    # HuggingFaceH4/ultrafeedback_binarized - parquet format
    "ultrafeedback": {
        "name": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "config": None,  # Use default, parquet format
        "format": "dpo_ultrafeedback_parquet",
    },
    # Intel/orca_dpo_pairs
    "orca_dpo": {
        "name": "Intel/orca_dpo_pairs",
        "split": "train",
        "config": None,
        "format": "dpo_orca",
    },
    # mlabonne/chatml_dpo_pairs
    "chatml_dpo": {
        "name": "mlabonne/chatml_dpo_pairs",
        "split": "train",
        "config": None,
        "format": "dpo_chatml",
    },
    # === PRIORITY DATASETS ===
    # vicgalle/OpenHermesPreferences-roleplay
    "openhermes_roleplay": {
        "name": "vicgalle/OpenHermesPreferences-roleplay",
        "split": "train",
        "config": None,
        "format": "dpo_pair",
    },
    # NeuralNovel/Neural-DPO
    "neural_dpo": {
        "name": "NeuralNovel/Neural-DPO",
        "split": "train",
        "config": None,
        "format": "dpo_orca",
    },
    # jondurbin/py-dpo-v0.1
    "py_dpo": {
        "name": "jondurbin/py-dpo-v0.1",
        "split": "train",
        "config": None,
        "format": "dpo_pair",
    },
    # argilla/dpo-mix-7k - {dataset, chosen, rejected, chosen_rating, rejected_rating}
    "dpo_mix": {
        "name": "argilla/dpo-mix-7k",
        "split": "train",
        "config": None,
        "format": "dpo_argilla_mix",
    },
    # argilla/OpenHermes2.5-dpo-binarized-alpha
    "openhermes_dpo": {
        "name": "argilla/OpenHermes2.5-dpo-binarized-alpha",
        "split": "train",
        "config": None,
        "format": "dpo_openhermes",
    },
    # === LEGACY DATASETS (SFT) ===
    # PIPPA
    "pippa": {
        "name": "PygmalionAI/PIPPA", 
        "hub_file": "pippa_deduped.jsonl",
        "split": "train",
        "format": "pippa",
    },
    # Roleplay IO
    "roleplay_io": {
        "name": "AlekseyKorshuk/roleplay-io",
        "split": "train",
        "config": None,
        "format": "roleplay_io",
    },
    # Hieunguyenminh roleplay
    "roleplay_vn": {
        "name": "hieunguyenminh/roleplay",
        "split": "train",
        "config": None,
        "format": "roleplay_vn",
    },
    # Anime waifu personality
    "waifu": {
        "name": "scryptiam/anime-waifu-personality-chat",
        "split": "train",
        "config": None,
        "format": "waifu",
    },
}

OUTPUT_FILE = "data/training_corpus.txt"
DPO_OUTPUT_FILE = "data/dpo_training_data.txt"  # Separate file for DPO pairs
IDENTITY_FILE = "data/custom_identity.txt"
MAX_SAMPLES_PER_DATASET = 50000

AGENT_SYSTEM_PROMPT = """You are DAI, an advanced AI assistant created by Helia. You are helpful, intelligent, and conversational."""

os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# 2. FORMATTERS FOR DIFFERENT DATASET FORMATS
# -----------------------------------------------------------------------------

def format_pippa(item):
    """Format PygmalionAI/PIPPA dataset
    
    JSON fields:
    - bot_name: character name
    - bot_greeting: introductory line (first utterance)
    - bot_description: brief overview of character
    - bot_definitions: example conversations for persona steering
    - conversation: list of {message, is_human} dicts
    """
    try:
        bot_name = item.get("bot_name", "Character")
        bot_greeting = item.get("bot_greeting", "")
        bot_description = item.get("bot_description", "")
        bot_definitions = item.get("bot_definitions", "")
        conversation = item.get("conversation", [])
        
        if not conversation:
            return None
        
        # Build system prompt with character info
        formatted = "[SYSTEM]"
        
        # Add character description
        if bot_description:
            formatted += f" You are {bot_name}. {bot_description}"
        else:
            formatted += f" You are {bot_name}."
        
        # Add definitions as context
        if bot_definitions:
            formatted += f" Character context: {bot_definitions[:200]}..."
        
        formatted += f" {AGENT_SYSTEM_PROMPT}\n"
        
        # Add greeting as first DAI message if exists
        if bot_greeting:
            formatted += f"[USER] (start conversation)\n[DAI] {bot_greeting}\n"
        
        # Process conversation turns
        for turn in conversation:
            message = turn.get("message", "")
            is_human = turn.get("is_human", False)
            
            if not message:
                continue
                
            if is_human:
                formatted += f"[USER] {message}\n"
            else:
                formatted += f"[DAI] {message}\n"
        
        return formatted.strip()
        
    except Exception as e:
        return None


def format_roleplay_io(item):
    """Format AlekseyKorshuk/roleplay-io - {input_text, output_text}"""
    try:
        input_text = item.get("input_text", "")
        output_text = item.get("output_text", "")
        
        if not output_text:
            return None
            
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {input_text.strip() if input_text else 'Hello!'}\n"
        formatted += f"[DAI] {output_text.strip()}\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_roleplay_vn(item):
    """Format hieunguyenminh/roleplay - {name, description, text}"""
    try:
        name = item.get("name", "")
        description = item.get("description", "")
        text = item.get("text", "")
        
        if not text:
            return None
        
        context = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}"
        if name:
            context += f" You are {name}."
        if description:
            context += f" {description}"
        context += "\n"
        
        # Parse conversation
        lines = text.strip().split('\n')
        formatted = context
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Simple heuristic for dialogue
            if ":" in line:
                parts = line.split(":", 1)
                speaker = parts[0].strip().lower()
                content = parts[1].strip() if len(parts) > 1 else ""
                
                if speaker in ["user", "human", "you"]:
                    formatted += f"[USER] {content}\n"
                elif speaker in ["assistant", "ai", "dai", name.lower(), name.title()]:
                    formatted += f"[DAI] {content}\n"
        
        # Fallback: if no conversation parsed
        if "[DAI]" not in formatted:
            formatted += f"[USER] Tell me about yourself.\n[DAI] {text.strip()[:200]}"
        
        return formatted.strip()
    except Exception:
        return None


def format_waifu(item):
    """Format anime waifu personality chat"""
    try:
        trait = item.get("trait") or item.get("type") or "anime character"
        dai_response = item.get("dialogue") or item.get("text")
        
        if not dai_response:
            return None
            
        system_prompt = f"You are DAI. You have a {trait} personality."
        
        fake_user_prompts = [
            "Hey, say something to me.",
            "Talk to me for a second.",
            "How are you feeling?",
            "What's on your mind right now?",
            "Are you there?",
        ]
        user_input = random.choice(fake_user_prompts)

        return f"[SYSTEM] {system_prompt}\n[USER] {user_input}\n[DAI] {dai_response.strip()}\n"
    except Exception:
        return None


def format_dpo_pair(item):
    """Format DPO datasets with chosen/rejected pairs
    
    Common fields in DPO datasets:
    - prompt / instruction / input / question
    - chosen / completion / response / output (the preferred response)
    - rejected (the dispreferred response)
    
    We output BOTH: chosen first (for SFT), then rejected for DPO training
    """
    try:
        # Try common field names for prompt
        prompt = (
            item.get("prompt") or 
            item.get("instruction") or 
            item.get("input") or 
            item.get("question") or
            ""
        )
        
        # Try common field names for chosen/positive response
        chosen = (
            item.get("chosen") or 
            item.get("completion") or 
            item.get("response") or
            item.get("output") or
            item.get("accepted") or
            ""
        )
        
        # Try common field names for rejected/negative response
        rejected = (
            item.get("rejected") or 
            item.get("negative") or
            ""
        )
        
        if not chosen:
            return None
        
        # Format: system prompt + user prompt + chosen response
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {prompt.strip() if prompt else 'Hello!'}\n"
        formatted += f"[DAI] {chosen.strip()}\n"
        
        # If we have rejected response, add it too (for DPO training)
        if rejected:
            formatted += f"<!-- REJECTED: {rejected.strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_dpo_ultrafeedback(item):
    """Format HuggingFaceH4/ultrafeedback_binarized
    
    Fields: prompt, prompt_id, chosen, rejected
    """
    try:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if not chosen or not rejected:
            return None
        
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {prompt.strip()}\n"
        formatted += f"[DAI] {chosen.strip()}\n"
        formatted += f"<!-- REJECTED: {rejected.strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_dpo_orca(item):
    """Format Intel/orca_dpo_pairs and NeuralNovel/Neural-DPO
    
    Fields: system, question, chosen, rejected
    """
    try:
        system = item.get("system", "")
        question = item.get("question", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if not chosen or not rejected:
            return None
        
        # Build prompt with system context
        if system:
            prompt = f"{system}\n{question}"
        else:
            prompt = question
        
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {prompt.strip()}\n"
        formatted += f"[DAI] {chosen.strip()}\n"
        formatted += f"<!-- REJECTED: {rejected.strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_dpo_chatml(item):
    """Format mlabonne/chatml_dpo_pairs - Pre-formatted in ChatML
    
    Expected format: list of messages with 'role' and 'content'
    Or: {prompt, chosen, rejected} in ChatML structure
    """
    try:
        # Try to handle both formats
        if "messages" in item:
            # Multi-turn conversation format
            formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
            for msg in item["messages"]:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "user":
                    formatted += f"[USER] {content}\n"
                elif role in ["assistant", "ai"]:
                    formatted += f"[DAI] {content}\n"
            return formatted.strip()
        
        # Try DPO pair format with ChatML-like structure
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if not chosen:
            return None
        
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {prompt.strip() if prompt else 'Hello!'}\n"
        formatted += f"[DAI] {chosen.strip()}\n"
        
        if rejected:
            formatted += f"<!-- REJECTED: {rejected.strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_dpo_ultrafeedback_parquet(item):
    """
    Format HuggingFaceH4/ultrafeedback_binarized (parquet format)
    
    Fields: prompt, prompt_id, chosen (list of messages), rejected (list of messages)
    The chosen/rejected are lists with 'content' field
    """
    try:
        # Get prompt
        prompt = item.get("prompt", "")
        if isinstance(prompt, list):
            # Parse list of messages
            prompt_text = ""
            for msg in prompt:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    role = msg.get("role", "").lower()
                    if role == "user":
                        prompt_text += f"[USER] {content}\n"
                    elif role == "system":
                        prompt_text = f"[SYSTEM] {content}\n" + prompt_text
            prompt = prompt_text.strip()
        
        # Get chosen response
        chosen = item.get("chosen", [])
        if isinstance(chosen, list) and len(chosen) > 0:
            if isinstance(chosen[0], dict):
                chosen = chosen[0].get("content", "")
            else:
                chosen = str(chosen[0])
        else:
            chosen = ""
        
        # Get rejected response
        rejected = item.get("rejected", [])
        if isinstance(rejected, list) and len(rejected) > 0:
            if isinstance(rejected[0], dict):
                rejected = rejected[0].get("content", "")
            else:
                rejected = str(rejected[0])
        else:
            rejected = ""
        
        if not chosen or not rejected:
            return None
        
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {prompt.strip()}\n"
        formatted += f"[DAI] {chosen.strip()}\n"
        formatted += f"<!-- REJECTED: {rejected.strip()} -->\n"
        
        return formatted.strip()
    except Exception as e:
        return None


def format_dpo_argilla_mix(item):
    """
    Format argilla/dpo-mix-7k
    
    Fields: dataset, chosen, rejected, chosen_rating, rejected_rating
    """
    try:
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if not chosen or not rejected:
            return None
        
        # Try to get prompt from dataset field or reconstruct
        dataset = item.get("dataset", "")
        
        # Parse chosen - could be string or dict
        if isinstance(chosen, dict):
            chosen = chosen.get("content", str(chosen))
        
        if isinstance(rejected, dict):
            rejected = rejected.get("content", str(rejected))
        
        # Build prompt
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        if dataset:
            formatted += f"[USER] {dataset}\n"
        else:
            formatted += f"[USER] Hello!\n"
        
        formatted += f"[DAI] {str(chosen).strip()}\n"
        formatted += f"<!-- REJECTED: {str(rejected).strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


def format_dpo_openhermes(item):
    """
    Format argilla/OpenHermes2.5-dpo-binarized-alpha
    
    Key fields: category, conversations, input, chosen, rejected, 
    chosen_score, rejected_score, generations
    
    Note: Many fields may be null, focus on chosen/rejected
    """
    try:
        # Get chosen and rejected responses
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        if not chosen or not rejected:
            return None
        
        # Parse if dict
        if isinstance(chosen, dict):
            chosen = chosen.get("content", str(chosen))
        if isinstance(rejected, dict):
            rejected = rejected.get("content", str(rejected))
        
        # Try to get prompt/input
        prompt = item.get("input", "")
        if not prompt:
            # Try to get from conversations
            conversations = item.get("conversations", [])
            if isinstance(conversations, list):
                prompt = ""
                for msg in conversations:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role == "user":
                        prompt += f"[USER] {content}\n"
                    elif role == "system":
                        prompt = f"[SYSTEM] {content}\n"
        
        if not prompt:
            prompt = "Hello!"
        
        # Build formatted output
        formatted = f"[SYSTEM]{AGENT_SYSTEM_PROMPT}\n"
        formatted += f"[USER] {str(prompt).strip()}\n"
        formatted += f"[DAI] {str(chosen).strip()}\n"
        formatted += f"<!-- REJECTED: {str(rejected).strip()} -->\n"
        
        return formatted.strip()
    except Exception:
        return None


FORMATTERS = {
    "pippa": format_pippa,
    "roleplay_io": format_roleplay_io,
    "roleplay_vn": format_roleplay_vn,
    "waifu": format_waifu,
    "dpo_pair": format_dpo_pair,
    "dpo_chatml": format_dpo_chatml,
    "dpo_ultrafeedback": format_dpo_ultrafeedback,
    "dpo_ultrafeedback_parquet": format_dpo_ultrafeedback_parquet,
    "dpo_orca": format_dpo_orca,
    "dpo_argilla_mix": format_dpo_argilla_mix,
    "dpo_openhermes": format_dpo_openhermes,
}

# -----------------------------------------------------------------------------
# 3. LOAD AND PROCESS
# -----------------------------------------------------------------------------

def load_and_process_dataset(dataset_key, config):
    """Load and process a single dataset"""
    print(f"\n{'='*50}")
    print(f"Loading: {config['name']}")
    if "file" in config:
        print(f"File: {config['file']}")
    print(f"Format: {config.get('format')}")
    print(f"{'='*50}")
    
    try:
        # Load with specific file if specified
        if "hub_file" in config:
            print("Downloading raw file to bypass Windows URL error...")
            local_path = hf_hub_download(
                repo_id=config["name"],
                filename=config["hub_file"],
                repo_type="dataset",
                token='YOUR_TOKEN'
            )
            dataset = load_dataset("json", data_files=local_path, split=config["split"])
        else:
            # Try loading the dataset
            dataset = None
            explicit_config = config.get("config")
            
            if explicit_config:
                try:
                    dataset = load_dataset(
                        config["name"], 
                        name=explicit_config,
                        split=config["split"],
                        token='YOUR_TOKEN'
                    )
                    print(f"  Loaded with config: {explicit_config}")
                except Exception as e1:
                    print(f"  Failed with config '{explicit_config}': {str(e1)[:60]}")
            
            # Try default if explicit config failed or not specified
            if dataset is None:
                try:
                    dataset = load_dataset(
                        config["name"], 
                        split=config["split"],
                        token='YOUR_TOKEN'
                    )
                    print(f"  Loaded with default config")
                except Exception as e2:
                    print(f"  Failed with default: {str(e2)[:60]}")
                    return []
        
        # Get dataset length
        num_samples = len(dataset)
        print(f"Loaded {num_samples:,} samples")
        
        # Get formatter using the 'format' field from config
        format_type = config.get("format", dataset_key)
        formatter = FORMATTERS.get(format_type)
        if not formatter:
            print(f"Error: No formatter found for format '{format_type}' (dataset: {dataset_key})")
            return []
        
        # Limit samples if specified
        samples_to_process = dataset
        if MAX_SAMPLES_PER_DATASET and len(dataset) > MAX_SAMPLES_PER_DATASET:
            samples_to_process = dataset.select(range(MAX_SAMPLES_PER_DATASET))
        
        # Debug: Print first item keys
        if len(samples_to_process) > 0:
            first_item = samples_to_process[0]
            if hasattr(first_item, 'keys'):
                print(f"  First item keys: {list(first_item.keys())}")
            else:
                print(f"  First item type: {type(first_item)}")
        
        # Process samples
        processed = []
        failed = 0
        for item in tqdm(samples_to_process, desc=f"Processing {dataset_key}"):
            try:
                formatted = formatter(item)
                if formatted:
                    processed.append(formatted)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
        
        print(f"Successfully processed {len(processed)} from {dataset_key}")
        if failed > 0:
            print(f"  Failed/Skipped: {failed}")
        
        return processed
        
    except Exception as e:
        print(f"Error loading {config['name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def inject_identity(f, identity_file, repeats=50):
    """Inject custom identity data"""
    if os.path.exists(identity_file):
        print(f"\nInjecting custom identity from {identity_file}...")
        with open(identity_file, "r", encoding="utf-8") as id_file:
            identity_content = id_file.read()
            for _ in range(repeats):
                f.write(identity_content.strip() + "\n\n")


# -----------------------------------------------------------------------------
# 4. EXECUTE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("DAI Data Preparation Pipeline")
    print("Preparing training data with DPO support...")
    print("="*60)
    
    # Separate DPO datasets from regular ones
    dpo_keys = ["ultrafeedback", "orca_dpo", 
                "chatml_dpo", "openhermes_roleplay", "neural_dpo", "py_dpo"]
    
    all_conversations = []
    dpo_conversations = []
    
    for key, config in DATASETS_CONFIG.items():
        processed = load_and_process_dataset(key, config)
        
        if key in dpo_keys:
            dpo_conversations.extend(processed)
            print(f"  -> Added to DPO dataset: {len(processed)} samples")
        else:
            all_conversations.extend(processed)
    
    # Shuffle both datasets
    random.shuffle(all_conversations)
    random.shuffle(dpo_conversations)
    
    print(f"\nTotal SFT conversations: {len(all_conversations)}")
    print(f"Total DPO conversations: {len(dpo_conversations)}")
    
    # Write SFT data
    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for conv in tqdm(all_conversations, desc="Writing SFT"):
            f.write(conv + "\n\n")
        
        inject_identity(f, IDENTITY_FILE, repeats=500)
    
    # Write DPO data if available
    if dpo_conversations:
        print(f"\nWriting DPO data to {DPO_OUTPUT_FILE}...")
        with open(DPO_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for conv in tqdm(dpo_conversations, desc="Writing DPO"):
                f.write(conv + "\n\n")
    
    avg_chars = sum(len(c) for c in all_conversations) / max(len(all_conversations), 1)
    estimated_tokens = int((len(all_conversations) * avg_chars) / 1.3)
    
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"Total SFT conversations: {len(all_conversations)}")
    print(f"Total DPO conversations: {len(dpo_conversations)}")
    print(f"Estimated SFT tokens: ~{estimated_tokens:,}")
    print(f"SFT Output: {OUTPUT_FILE}")
    if dpo_conversations:
        print(f"DPO Output: {DPO_OUTPUT_FILE}")
    print("="*60)