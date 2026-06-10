"""
Day 2 - Script 1: Exploring the Hugging Face Ecosystem
=======================================================
Demonstrates loading a tokenizer and dataset, and inspecting
the output structure without requiring a GPU.

Run with: python explore_hf_ecosystem.py
Requirements: pip install transformers datasets
"""

from transformers import AutoTokenizer
from datasets import load_dataset

# ─────────────────────────────────────────────
# 1. Load a lightweight tokenizer from the Hub
# ─────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading tokenizer for: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ─────────────────────────────────────────────
# 2. Tokenize raw vs. chat-formatted prompts
# ─────────────────────────────────────────────
raw_prompt = "What is fine-tuning in machine learning?"

chat_messages = [
    {"role": "system", "content": "You are a helpful AI assistant specializing in machine learning."},
    {"role": "user",   "content": "What is fine-tuning in machine learning?"},
]

# apply_chat_template formats messages into the model's expected prompt structure
formatted_prompt = tokenizer.apply_chat_template(
    chat_messages,
    tokenize=False,
    add_generation_prompt=True,
)

raw_tokens      = tokenizer(raw_prompt, return_tensors="pt")
formatted_tokens = tokenizer(formatted_prompt, return_tensors="pt")

print("\n─── Tokenization Comparison ───")
print(f"  Raw prompt token count      : {raw_tokens['input_ids'].shape[1]}")
print(f"  Formatted prompt token count: {formatted_tokens['input_ids'].shape[1]}")
print(f"\n  Formatted prompt preview:\n{formatted_prompt[:300]}...")

# ─────────────────────────────────────────────
# 3. Load a small dataset from the Hub
# ─────────────────────────────────────────────
print("\n─── Loading Dataset ───")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # first 100 rows only

print(f"  Dataset type  : {type(dataset)}")
print(f"  Number of rows: {len(dataset)}")
print(f"  Column names  : {dataset.column_names}")
print(f"\n  Sample row:\n  {dataset[0]}")

# ─────────────────────────────────────────────
# 4. Apply a formatting transformation with .map()
# ─────────────────────────────────────────────
def format_alpaca_prompt(example):
    """Convert Alpaca dataset rows into a unified instruction-response string."""
    if example.get("input", "").strip():
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}

formatted_dataset = dataset.map(format_alpaca_prompt, remove_columns=dataset.column_names)

print("\n─── After Formatting ───")
print(f"  Column names  : {formatted_dataset.column_names}")
print(f"\n  Sample formatted row:\n{formatted_dataset[0]['text']}")

print("\n✅ Day 2 Script 1 complete!")
