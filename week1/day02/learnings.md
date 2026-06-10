# Day 2: The Hugging Face Ecosystem

## 📖 Concepts Learned

### Core Libraries Overview
The Hugging Face ecosystem is the industry-standard toolkit for working with large language models. The five key libraries are:

| Library | Purpose | Key Use Case |
|---|---|---|
| `transformers` | Load, run, and fine-tune pre-trained models | Model loading, inference, training |
| `datasets` | Download, process, and cache datasets | Dataset loading, tokenization, streaming |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA, QLoRA, etc.) | Adapter-based training |
| `trl` | Transformer Reinforcement Learning | SFT, DPO, RLHF training loops |
| `accelerate` | Distributed & mixed-precision training | Multi-GPU, TPU, CPU offloading |

---

### `transformers`
- **`AutoTokenizer`**: Automatically loads the correct tokenizer for a model from the Hub.
- **`AutoModelForCausalLM`**: Loads the correct model architecture for causal language modeling (text generation).
- **`pipeline`**: High-level API for quick inference without manual tokenization.
- Models are identified by their Hub repo ID, e.g., `"unsloth/Qwen2.5-7B-Instruct"`.

### `datasets`
- **`load_dataset("name")`**: Downloads and caches datasets from the Hugging Face Hub.
- Supports **streaming** (`streaming=True`) for massive datasets that don't fit in RAM.
- The `.map()` function applies transformations (tokenization, formatting) efficiently across all dataset splits.
- Datasets are stored as Apache Arrow files for fast columnar access.

### `peft` (Parameter-Efficient Fine-Tuning)
- **LoRA (Low-Rank Adaptation)**: Injects small trainable rank-decomposition matrices into attention layers, freezing the rest of the model weights. Reduces trainable parameters by ~99%.
- **`LoraConfig`**: Defines the rank (`r`), scaling factor (`lora_alpha`), dropout, and target modules.
- **`get_peft_model(model, config)`**: Wraps a base model with LoRA adapters.

### `trl` (Transformer Reinforcement Learning)
- **`SFTTrainer`**: A convenience wrapper around Hugging Face `Trainer` for Supervised Fine-Tuning. Handles dataset formatting, packing, and loss masking automatically.
- **`DPOTrainer`**: For Direct Preference Optimization — training on preference pairs (chosen vs. rejected responses).
- Works seamlessly with PEFT adapters.

### `accelerate`
- Abstracts away the complexity of training across different hardware setups (single GPU, multi-GPU, TPU, CPU).
- **Mixed precision** (`fp16`, `bf16`): Reduces memory usage during training with minimal quality loss.
- **Gradient accumulation**: Simulates larger batch sizes on memory-constrained hardware.
- Configured via `accelerate config` CLI command or `TrainingArguments`.

---

## 🛠️ Implementation Details

### Scripts Created
- **`explore_hf_ecosystem.py`**: Demonstrates loading a tokenizer, model, and dataset. Shows the token count difference between a raw vs. formatted prompt.
- **`peft_lora_setup.py`**: Sets up a base model with a `LoraConfig`, prints trainable vs. total parameter counts.

### Key Findings
- Loading `Qwen2.5-0.5B` (small model for exploration) and inspecting tokenizer outputs.
- A LoRA config with `r=16` on `Qwen2.5-0.5B` reduces trainable parameters from ~494M to ~2.4M (~0.5% of total).
- `SFTTrainer` requires data formatted with a `"text"` column or a `formatting_func` callback.

---

## ❓ What Confused Me / Challenges
- The difference between `AutoModelForCausalLM` and `AutoModelForSeq2SeqLM` — causal models (GPT-style) predict the next token, while seq2seq models (T5-style) encode then decode.
- Understanding `target_modules`: you need to know the architecture's attention layer names (e.g., `"q_proj"`, `"v_proj"` for LLaMA/Qwen). The `peft` library has helper utilities to list them.
- `accelerate` vs. `torch.distributed`: `accelerate` is a higher-level wrapper that makes distributed training accessible without deep PyTorch distributed knowledge.

---

## 🔗 Useful Links / Resources
- [Hugging Face `transformers` Docs](https://huggingface.co/docs/transformers)
- [Hugging Face `datasets` Docs](https://huggingface.co/docs/datasets)
- [PEFT Library Docs](https://huggingface.co/docs/peft)
- [TRL Library Docs](https://huggingface.co/docs/trl)
- [Accelerate Docs](https://huggingface.co/docs/accelerate)
- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
