"""
Day 3 - Script 1: Dataset Curation Pipeline
============================================
Loads the Alpaca dataset, deduplicates, filters by quality,
reformats to ChatML, creates train/val/test splits, and saves to disk.

Run with: python dataset_curation.py
Requirements: pip install datasets transformers
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import hashlib

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID       = "tatsu-lab/alpaca"
OUTPUT_DIR       = "./cleaned_alpaca"
MIN_RESPONSE_LEN = 10    # minimum characters in the output field
MAX_RESPONSE_LEN = 2000  # maximum characters in the output field
SEED             = 42

# ─────────────────────────────────────────────
# 1. Load raw dataset
# ─────────────────────────────────────────────
print(f"Loading dataset: {DATASET_ID}")
raw = load_dataset(DATASET_ID, split="train")
print(f"  Raw size: {len(raw):,} rows")

# ─────────────────────────────────────────────
# 2. Exact Deduplication
# ─────────────────────────────────────────────
def get_row_hash(example):
    """Create a unique hash from instruction + input + output."""
    content = f"{example['instruction']}|{example.get('input', '')}|{example['output']}"
    return {"_hash": hashlib.md5(content.encode()).hexdigest()}

print("\nApplying deduplication...")
hashed = raw.map(get_row_hash)

# Collect unique hashes and filter
seen_hashes = set()
def is_unique(example):
    h = example["_hash"]
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True

deduped = hashed.filter(is_unique)
deduped = deduped.remove_columns(["_hash"])
print(f"  After dedup: {len(deduped):,} rows  (removed {len(raw) - len(deduped):,} duplicates)")

# ─────────────────────────────────────────────
# 3. Quality Filtering
# ─────────────────────────────────────────────
def quality_filter(example):
    output = example.get("output", "").strip()
    instruction = example.get("instruction", "").strip()

    # Remove empty or trivially short responses
    if len(output) < MIN_RESPONSE_LEN:
        return False
    # Remove overly long responses (risk of truncation)
    if len(output) > MAX_RESPONSE_LEN:
        return False
    # Remove empty instructions
    if not instruction:
        return False
    # Remove obvious HTML artifacts
    if "<html" in output.lower() or "http://" in output[:20]:
        return False
    return True

print("\nApplying quality filters...")
filtered = deduped.filter(quality_filter)
print(f"  After filtering: {len(filtered):,} rows  (removed {len(deduped) - len(filtered):,} low-quality rows)")

# ─────────────────────────────────────────────
# 4. Reformat to ChatML using apply_chat_template
# ─────────────────────────────────────────────
print(f"\nLoading tokenizer for chat template: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def format_to_chatml(example):
    """
    Convert Alpaca-format rows into ChatML using the model's chat template.
    The response is NOT included in the prompt — it is a separate field used
    by SFTTrainer to compute loss only on the assistant's response.
    """
    system_msg = "You are a helpful AI assistant."
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content += f"\n\nContext:\n{example['input']}"

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]

    # Format the full conversation (including assistant response) as a string
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": full_text}

print("Reformatting to ChatML...")
formatted = filtered.map(
    format_to_chatml,
    remove_columns=filtered.column_names,
    desc="Formatting to ChatML",
)
print(f"  Formatted {len(formatted):,} rows")
print(f"\n  Sample output:\n{formatted[0]['text'][:500]}...")

# ─────────────────────────────────────────────
# 5. Train / Validation / Test Split
# ─────────────────────────────────────────────
print("\nCreating train/val/test splits...")

# 90% train, 10% temp
train_temp = formatted.train_test_split(test_size=0.10, seed=SEED)
# Split the 10% temp into 50/50 → 5% val, 5% test
val_test   = train_temp["test"].train_test_split(test_size=0.50, seed=SEED)

dataset_dict = DatasetDict({
    "train":      train_temp["train"],
    "validation": val_test["train"],
    "test":       val_test["test"],
})

for split, ds in dataset_dict.items():
    print(f"  {split:12s}: {len(ds):>6,} rows")

# ─────────────────────────────────────────────
# 6. Save to disk
# ─────────────────────────────────────────────
print(f"\nSaving cleaned dataset to: {OUTPUT_DIR}")
dataset_dict.save_to_disk(OUTPUT_DIR)
print("✅ Dataset saved successfully!")

# ─────────────────────────────────────────────
# 7. Quick token length distribution analysis
# ─────────────────────────────────────────────
def count_tokens(example):
    ids = tokenizer(example["text"], return_tensors="pt")["input_ids"]
    return {"token_count": ids.shape[1]}

print("\nComputing token length statistics on train split (sample 500 rows)...")
sample = dataset_dict["train"].select(range(min(500, len(dataset_dict["train"]))))
sample = sample.map(count_tokens)
counts = sample["token_count"]

print(f"  Min tokens : {min(counts)}")
print(f"  Max tokens : {max(counts)}")
print(f"  Avg tokens : {sum(counts) // len(counts)}")
print(f"  >512 tokens: {sum(1 for c in counts if c > 512)} / {len(counts)} rows")

print("\n✅ Day 3 Script 1 complete!")
