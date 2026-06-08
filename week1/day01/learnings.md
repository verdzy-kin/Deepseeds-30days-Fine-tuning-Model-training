# Day 1: Why Fine-Tune? Understanding the "When and Why"

## 📖 Concepts Learned
- **When Fine-Tuning Wins Over Prompting**:
  - Consistent output formatting (e.g., always respond in JSON, always follow a strict schema)
  - Specialized vocabulary / domain-specific terminology (medical, legal, financial)
  - Task-specific behavior that needs to be deeply embedded into the model's weights
  - Lower inference cost at scale (no need for lengthy system prompts on every call)
- **When NOT to Fine-Tune**:
  - Better prompt engineering already achieves the goal
  - Dataset is too small (< a few hundred high-quality examples)
  - The task requires broad general reasoning, not specialization
- **Fine-Tuning vs. RAG**:
  - RAG (Retrieval-Augmented Generation) = inject knowledge at inference time
  - Fine-tuning = bake knowledge/behavior into weights at training time
  - They are complementary, not mutually exclusive

## 🛠️ Implementation Details
- *No code today — this is a conceptual foundations day*
- Reviewed case studies: consistent format use cases, domain-specific vocabulary, specialized behaviors
- Reviewed the Hugging Face model hub for suitable base models (Qwen2.5-7B, Llama-3.2-3B)

## ❓ What Confused Me / Challenges
- Had issues getting the openAI API key, however i was able to use Gemini instead.
- The basic concepts of fine tuning are a bit confusing.

## 🔗 Useful Links / Resources
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Library](https://huggingface.co/docs/peft)
