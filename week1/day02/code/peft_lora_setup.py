"""
Day 2 - Script 2: Setting Up a LoRA Model with PEFT
=====================================================
Demonstrates wrapping a pre-trained model with LoRA adapters
and comparing trainable vs. total parameter counts.

Run with: python peft_lora_setup.py
Requirements: pip install transformers peft torch accelerate
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ─────────────────────────────────────────────
# 1. Load base model
# ─────────────────────────────────────────────
print(f"Loading base model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",     # use bf16/fp16 if supported, else fp32
    device_map="auto",      # auto-select GPU/CPU
)

# ─────────────────────────────────────────────
# 2. Inspect model layers to find target modules
# ─────────────────────────────────────────────
print("\n─── Model Layer Names (sample) ───")
for name, _ in list(model.named_modules())[:20]:
    if name:
        print(f"  {name}")

# ─────────────────────────────────────────────
# 3. Define LoRA Configuration
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # rank of the low-rank decomposition matrices
    lora_alpha=32,      # scaling factor (alpha/r = effective learning rate scale)
    lora_dropout=0.05,  # dropout applied to LoRA layers to prevent overfitting
    bias="none",        # do not train bias terms
    target_modules=[    # Qwen2.5 attention projection layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print(f"\n─── LoRA Config ───")
print(f"  Rank (r)       : {lora_config.r}")
print(f"  Alpha          : {lora_config.lora_alpha}")
print(f"  Target Modules : {lora_config.target_modules}")

# ─────────────────────────────────────────────
# 4. Wrap model with PEFT LoRA adapters
# ─────────────────────────────────────────────
peft_model = get_peft_model(model, lora_config)

# ─────────────────────────────────────────────
# 5. Compare parameter counts
# ─────────────────────────────────────────────
def count_parameters(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(peft_model)
frozen_params = total_params - trainable_params

print("\n─── Parameter Counts ───")
print(f"  Total parameters    : {total_params:>15,}")
print(f"  Trainable (LoRA)    : {trainable_params:>15,}  ({100 * trainable_params / total_params:.4f}%)")
print(f"  Frozen (base model) : {frozen_params:>15,}  ({100 * frozen_params / total_params:.4f}%)")

# peft helper method — equivalent to above
print("\n─── PEFT Built-in Summary ───")
peft_model.print_trainable_parameters()

print("\n✅ Day 2 Script 2 complete!")
