# Deepseeds-30days-Fine-tuning-Model-training

Welcome to the **Deepseeds 30 Days of Fine-Tuning & Model Training** repository! This is a structured 30-day journey dedicated to mastering the art and science of customizing pre-trained language models, curating high-quality datasets, and optimizing model performance using parameter-efficient techniques like LoRA and QLoRA.

---

## 🚀 Overview

Fine-tuning transforms general-purpose language models into domain-specific experts. Over the next 30 days, this repository will document daily learnings, code scripts, experiments, and projects focused on the end-to-end model training lifecycle—from dataset curation to distributed training and production deployment.

---

## 📅 30-Day Curriculum & Roadmap

### Week 1: Foundations of Fine-Tuning & Model Training
- **Day 1**: Why Fine-Tune? (Consistent format, vocabulary, cost vs. prompting)
- **Day 2**: Hugging Face Ecosystem (`transformers`, `datasets`, `peft`, `trl`, `accelerate`)
- **Day 3**: Dataset Curation & Formatting (Instruction format, deduplication, quality filtering)
- **Day 4**: LoRA: Low-Rank Adaptation (Math of LoRA, PEFT)
- **Day 5**: QLoRA: Fine-Tuning on Consumer Hardware (4-bit quantization, bitsandbytes, PEFT)
- **Day 6**: Unsloth: Fast Fine-Tuning (CUDA kernels, optimization)
- **Day 7**: Weekly Project: Sunday Presentation (LoRA visual diagram, QLoRA model response demo)

### Week 2: Production Training Configs & Alignment
- **Day 8**: Axolotl: Production Training Configs (YAML schema, configuration)
- **Day 9**: Data Synthesis with LLMs (Self-instruct, Constitutional AI, Evol-Instruct)
- **Day 10**: Instruction Tuning vs. Continued Pre-Training (Domain adaptation vs. chat alignment)
- **Day 11**: DPO: Direct Preference Optimization (Preference training, trl DPO)
- **Day 12**: Evaluation-Driven Fine-Tuning (Metrics: ROUGE/BLEU, LLM-as-a-judge)
- **Day 13**: Merging & Quantizing Your Fine-Tuned Model (peft merge, llama.cpp GGUF conversion)
- **Day 14**: Weekly Project: Sunday Presentation (DPO model demo, HF model/data card walkthrough)

### Week 3: Deepening & Optimization
- **Day 15**: Multi-Task Fine-Tuning (Catastrophic forgetting, LoRA merging, TIES/DARE)
- **Day 16**: Training Instability & Debugging (Loss spikes, mode collapse, wandb logging)
- **Day 17**: GRPO: Training Reasoning Models (Group Relative Policy Optimization, DeepSeek R1 reasoning)
- **Day 18**: Open Source Contribution Day (Unsloth, Axolotl, TRL)
- **Day 19**: Serving Your Fine-Tuned Model (Ollama, vLLM, HF Inference Endpoints, FastAPI wrapper)
- **Day 20**: Domain Adaptation at Scale (Case studies: CodeLlama, BioMedLM, Lawyer-Llama)
- **Day 21**: Weekly Project: Sunday Presentation (GRPO reasoning model demo, GRPO vs. RLHF analysis)

### Week 4: Mastery & Capstone
- **Days 22–27**: Capstone Project (Curated dataset + data card, QLoRA + DPO training, GGUF merge, deployed endpoint)
- **Day 28**: Documentation & README (Architecture diagram, setup docs, API docs)
- **Day 29**: Polish & Deploy (Railway, Render, AWS Lambda + demo video)
- **Day 30**: Final Presentation (Full showcase to community)

---

## 📊 Daily Progress Log

| Day | Topic | Status | Key Output / Learnings Link |
| :--- | :--- | :---: | :--- |
| **01** | Why Fine-Tune? | ⬜ | [learnings.md](file:///d:/30daysGenAI/Model%20training/week1/day01/learnings.md) |
| **02** | Hugging Face Ecosystem | ⬜ | - |
| **03** | Dataset Curation & Formatting | ⬜ | - |
| **04** | LoRA: Low-Rank Adaptation | ⬜ | - |
| **05** | QLoRA: Fine-Tuning on Consumer Hardware | ⬜ | - |
| **06** | Unsloth: Fast Fine-Tuning | ⬜ | - |
| **07** | Weekly Project: LoRA & QLoRA Setup Demo | ⬜ | - |
| **08** | Axolotl: Production Configs | ⬜ | - |
| **09** | Data Synthesis with LLMs | ⬜ | - |
| **10** | Instruction Tuning vs. Continued Pre-Training | ⬜ | - |
| **11** | DPO: Direct Preference Optimization | ⬜ | - |
| **12** | Evaluation-Driven Fine-Tuning | ⬜ | - |
| **13** | Merging & Quantizing Fine-Tuned Models | ⬜ | - |
| **14** | Weekly Project: DPO & HF Walkthrough | ⬜ | - |
| **15** | Multi-Task Fine-Tuning | ⬜ | - |
| **16** | Training Instability & Debugging | ⬜ | - |
| **17** | GRPO: Training Reasoning Models | ⬜ | - |
| **18** | Open Source Contribution Day | ⬜ | - |
| **19** | Serving Your Fine-Tuned Model | ⬜ | - |
| **20** | Domain Adaptation at Scale | ⬜ | - |
| **21** | Weekly Project: GRPO Reasoning Demo | ⬜ | - |
| **22-27**| Capstone Project Development | ⬜ | - |
| **28** | Documentation & README | ⬜ | - |
| **29** | Polish & Deploy | ⬜ | - |
| **30** | Final Presentation | ⬜ | - |

---

## 🛠️ Getting Started

### Prerequisites
- [Python](https://www.python.org/) (3.10 or higher)
- [PyTorch](https://pytorch.org/) (compatible with your CUDA version)
- [Git](https://git-scm.com/) installed
- Hugging Face account and API Token

---

## 📚 Resources & Documentation
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Axolotl Repository](https://github.com/OpenAccess-AI-Collective/axolotl)

---

## 🤝 Contributing
Contributions are welcome! If you have any suggestions, please feel free to open an issue or submit a pull request.

---

*Happy Fine-Tuning! 🚀🧠*