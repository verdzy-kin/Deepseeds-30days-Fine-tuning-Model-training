"""
Day 3 - Script 2: Format Converter Utilities
=============================================
Reusable functions to convert between common fine-tuning dataset formats:
  - Alpaca  → ChatML
  - ShareGPT → ChatML
  - Any ChatML → plain text (for inspection)

Run with: python format_converter.py
Requirements: pip install transformers
"""

from transformers import AutoTokenizer
from typing import Optional

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ─────────────────────────────────────────────
# Format 1: Alpaca → ChatML
# ─────────────────────────────────────────────
def alpaca_to_chatml(
    instruction: str,
    output: str,
    input_context: Optional[str] = None,
    system_message: str = "You are a helpful AI assistant.",
) -> str:
    """
    Convert a single Alpaca-format example to ChatML format.

    Args:
        instruction: The task instruction
        output: The expected model response
        input_context: Optional additional context
        system_message: System prompt

    Returns:
        Full ChatML-formatted string ready for tokenization
    """
    user_content = instruction
    if input_context and input_context.strip():
        user_content += f"\n\nContext:\n{input_context.strip()}"

    messages = [
        {"role": "system",    "content": system_message},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─────────────────────────────────────────────
# Format 2: ShareGPT → ChatML
# ─────────────────────────────────────────────
def sharegpt_to_chatml(
    conversations: list[dict],
    system_message: Optional[str] = None,
) -> str:
    """
    Convert a ShareGPT-format conversation list to ChatML.

    Args:
        conversations: List of {"from": "human"/"gpt", "value": str}
        system_message: Optional system prompt to prepend

    Returns:
        Full ChatML-formatted string
    """
    ROLE_MAP = {
        "human":  "user",
        "gpt":    "assistant",
        "system": "system",
    }

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    for turn in conversations:
        role  = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        value = turn.get("value", "").strip()
        if role and value:
            messages.append({"role": role, "content": value})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─────────────────────────────────────────────
# Format 3: ChatML messages → Inference prompt
#           (adds generation prompt, no response)
# ─────────────────────────────────────────────
def build_inference_prompt(
    user_message: str,
    system_message: str = "You are a helpful AI assistant.",
) -> str:
    """
    Build a prompt suitable for inference (no response included).
    Ends with the assistant generation token so the model knows to respond.

    Args:
        user_message: The user's query
        system_message: System prompt

    Returns:
        Inference-ready ChatML prompt string
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # <-- key difference for inference
    )


# ─────────────────────────────────────────────
# Demo / Test
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("1. Alpaca → ChatML")
    print("=" * 60)
    alpaca_result = alpaca_to_chatml(
        instruction="Summarize the following article in 3 bullet points.",
        input_context="Artificial intelligence is transforming industries...",
        output="• AI is rapidly changing multiple industries.\n• Key sectors include healthcare, finance, and logistics.\n• Adoption is accelerating due to falling compute costs.",
    )
    print(alpaca_result)

    print("\n" + "=" * 60)
    print("2. ShareGPT → ChatML")
    print("=" * 60)
    sharegpt_convo = [
        {"from": "human", "value": "What is the difference between LoRA and QLoRA?"},
        {"from": "gpt",   "value": "LoRA adds low-rank trainable adapters to the model's weight matrices while keeping the base model frozen. QLoRA builds on LoRA by first quantizing the base model to 4-bit precision, drastically reducing its VRAM footprint, then training the LoRA adapters in full precision on top of the quantized model."},
        {"from": "human", "value": "Which should I use on a 12GB GPU?"},
        {"from": "gpt",   "value": "QLoRA is the better choice on a 12GB GPU. It allows you to fine-tune 7B parameter models comfortably within that VRAM budget, whereas standard LoRA on an unquantized 7B model would require ~28GB of VRAM."},
    ]
    sharegpt_result = sharegpt_to_chatml(
        sharegpt_convo,
        system_message="You are an expert in LLM fine-tuning techniques.",
    )
    print(sharegpt_result)

    print("\n" + "=" * 60)
    print("3. Inference Prompt (no response)")
    print("=" * 60)
    inference_prompt = build_inference_prompt(
        user_message="Explain gradient checkpointing in one paragraph.",
    )
    print(inference_prompt)
    print("\n✅ Day 3 Script 2 complete!")
