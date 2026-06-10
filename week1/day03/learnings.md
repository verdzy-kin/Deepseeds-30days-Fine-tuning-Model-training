# Day 3: Dataset Curation & Formatting

## 📖 Concepts Learned

### Why Dataset Quality > Quantity
> *"Garbage in, garbage out"* is even more true for fine-tuning than for training from scratch. A model fine-tuned on 1,000 high-quality examples will outperform one fine-tuned on 100,000 noisy ones.

The three pillars of dataset quality:
1. **Relevance** — examples must be representative of the target task/domain
2. **Correctness** — responses must be factually/logically accurate
3. **Diversity** — avoid repetitive patterns that cause the model to overfit to surface-level features

---

### Instruction Formatting
Fine-tuning datasets for instruction-following must follow a consistent prompt template. The most common formats:

#### Alpaca Format (3-field)
```
### Instruction:
{instruction}

### Input:
{input}   ← optional context

### Response:
{output}
```

#### ChatML Format (multi-turn)
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

#### ShareGPT Format (conversational list)
```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ]
}
```

**Key rule**: Always apply the tokenizer's `apply_chat_template()` to ensure the model sees the same format it was pre-trained with. Do not invent custom formats for base models with an existing chat template.

---

### Deduplication
Duplicate rows in a fine-tuning dataset cause:
- Overfitting to repeated examples
- Inflated training loss improvement that doesn't generalize
- Wasted compute

**Deduplication strategies**:
| Method | How | When to Use |
|---|---|---|
| Exact match | Hash each row; drop duplicates | Fast, cheap, catches 100% of exact copies |
| Near-duplicate (MinHash/LSH) | Locality-sensitive hashing on n-grams | Catches paraphrased duplicates |
| Semantic dedup (embedding similarity) | Embed rows, cluster, pick representative | Most powerful; expensive at scale |

For datasets < 100k rows, exact deduplication with pandas `.drop_duplicates()` is usually sufficient. For large corpora, use `datasketch` (MinHash) or `faiss` (embedding similarity).

---

### Quality Filtering
Steps to filter out low-quality examples:

1. **Length filtering**: Remove examples with response length < 10 tokens or > 2048 tokens (too short = trivial; too long = truncation risk).
2. **Perplexity filtering**: Use a small reference model to score each example; remove outliers with very high perplexity (incoherent text).
3. **Keyword/pattern filtering**: Remove examples with boilerplate text, HTML artifacts, `<UNK>` tokens, or PII.
4. **Instruction-response alignment**: Verify the response actually answers the instruction (can use LLM-as-judge for this at small scale).
5. **Language filtering**: Use `langdetect` or `fastText` to keep only target-language examples.

---

### Dataset Splits
Always create proper train/validation/test splits **before** tokenizing:
- **Train**: Used during gradient updates
- **Validation**: Used to monitor loss during training (early stopping)
- **Test**: Held out entirely — only used for final evaluation

Standard split ratio: **90% / 5% / 5%** for small datasets. For large datasets (>100k), a fixed val/test set of 1k–5k examples is sufficient.

---

## 🛠️ Implementation Details

### Scripts Created
- **`dataset_curation.py`**: Loads the Alpaca dataset, applies exact deduplication, performs length filtering, reformats to ChatML format, and saves the cleaned dataset to disk.
- **`format_converter.py`**: Utility functions to convert between Alpaca, ChatML, and ShareGPT formats.

### Key Findings
- The `tatsu-lab/alpaca` dataset (52k rows) contains ~1,200 near-duplicate or empty instruction rows — a significant ~2.3% noise rate.
- Length filtering (dropping responses < 5 words) removed an additional ~300 rows.
- After cleaning: **~50,500 high-quality instruction-response pairs** ready for tokenization.
- Using `datasets.Dataset.train_test_split()` creates reproducible splits with a fixed `seed`.

---

## ❓ What Confused Me / Challenges
- Understanding **when to tokenize** — you should tokenize *after* all filtering and formatting to avoid operating on token IDs instead of text.
- The difference between **packing** and **padding**: Packing concatenates multiple short examples into one sequence (more efficient), while padding pads each example to the same length (simpler but wastes compute). `SFTTrainer` supports packing natively.
- **Loss masking on input tokens**: During training, we only want to compute loss on the *response* tokens, not the instruction tokens. `SFTTrainer` handles this automatically via `dataset_text_field`.

---

## 🔗 Useful Links / Resources
- [Hugging Face `datasets` Docs — Processing Data](https://huggingface.co/docs/datasets/process)
- [Alpaca Dataset on HF Hub](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [OpenHermes 2.5 Dataset (high-quality example)](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [DataTrove: Large-Scale Data Processing](https://github.com/huggingface/datatrove)
- [MinHash Deduplication with `datasketch`](https://github.com/ekzhu/datasketch)
- [The Curse of Duplicates in Fine-Tuning](https://arxiv.org/abs/2205.10487)
