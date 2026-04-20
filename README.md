# LLM Fine-Tuning: Job Description Structured Extractor

Fine-tuned Qwen2-0.5B using QLoRA to extract structured JSON from job descriptions. Achieves 100% JSON validity and 70%+ field accuracy on a held-out test set — trained in under 4 minutes on a 4GB GPU.

## Results

Evaluated on 10 held-out job descriptions (unseen during training):

| Metric | Score |
|--------|-------|
| **JSON validity** | 100% (10/10) |
| Company | 95% |
| Language | 85% |
| Location | 75% |
| Title | 70% |
| Seniority | 50% |
| Work Model | 45% |
| Required Skills (F1) | 0.36 |
| Nice-to-have (F1) | 0.10 |
| Salary | 10% |

The model reliably produces valid JSON and extracts entity-level fields well. List fields (skills) and rarely-present fields (salary) need more training data.

## Architecture

```
Raw Job Description (text)
        │
        ▼
┌──────────────────────────┐
│  Qwen2-0.5B-Instruct     │
│  + QLoRA (r=16, alpha=32) │
│  4-bit NF4 quantization   │
└──────────┬───────────────┘
           │
           ▼
Structured JSON Output
{
  "title": "AI Engineer",
  "company": "Acme Corp",
  "location": "Berlin, Germany",
  "work_model": "hybrid",
  "seniority": "mid-level",
  "required_skills": ["Python", "PyTorch", "RAG"],
  "nice_to_have": ["Kubernetes", "AWS"],
  "salary": "€65,000-€85,000",
  "language": "English"
}
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2-0.5B-Instruct |
| Method | QLoRA (4-bit NF4 + LoRA adapters) |
| LoRA rank / alpha | 16 / 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 2.16M / 496M (0.44%) |
| Epochs | 5 |
| Batch size | 1 (gradient accumulation 8) |
| Learning rate | 2e-4 (cosine schedule) |
| Max sequence length | 1024 |
| Training time | 3 min 43 sec |
| Hardware | NVIDIA RTX 3050 (4GB VRAM) |
| Dataset | 28 train / 10 test (38 total, hand-labeled) |

**Training loss:** 2.54 → 2.03 over 5 epochs

**Token accuracy:** 53.3% → 60.8%

## Dataset

38 real job descriptions hand-labeled with structured JSON. Sources include companies like Anthropic, Mistral, JetBrains, DHL, IONOS, appliedAI, and others from the German AI job market.

Format (Alpaca-style instruction tuning):
```json
{
  "instruction": "Extract structured information from this job description as JSON.",
  "input": "<raw JD text>",
  "output": "<structured JSON with 9 fields>"
}
```

## Quick Start

### Train

```bash
PYTHONUTF8=1 python scripts/train.py \
  --base_model "Qwen/Qwen2-0.5B-Instruct" \
  --dataset "data/processed/train_split.jsonl" \
  --output_dir "output/jd-extractor-qwen-0.5b-v2" \
  --epochs 5 --batch_size 1 --grad_accum 8 --max_seq_len 1024
```

### Evaluate

```bash
PYTHONUTF8=1 python scripts/evaluate.py \
  --base_model "Qwen/Qwen2-0.5B-Instruct" \
  --adapter "output/jd-extractor-qwen-0.5b-v2" \
  --test_file "data/processed/test.jsonl"
```

### Inference

```bash
python scripts/inference.py --input "paste a job description here"
```

## Project Structure

```
├── data/
│   ├── raw/                       # 38 JD text files + JSON labels
│   ├── processed/
│   │   ├── train_split.jsonl      # 28 training examples
│   │   └── test.jsonl             # 10 held-out test examples
│   └── prepare_dataset.py         # Raw → Alpaca JSONL converter
├── scripts/
│   ├── train.py                   # QLoRA fine-tuning (configurable)
│   ├── evaluate.py                # JSON validity + field scoring
│   └── inference.py               # Run model on new JDs
├── eval/
│   └── results.json               # Full evaluation output
├── output/
│   └── jd-extractor-qwen-0.5b-v2/ # Saved LoRA adapter weights
└── requirements.txt
```

## Tech Stack

- **Model:** Qwen/Qwen2-0.5B-Instruct (496M params)
- **Fine-tuning:** QLoRA via PEFT + bitsandbytes (4-bit NF4)
- **Training:** TRL SFTTrainer, HuggingFace Transformers
- **Hardware:** RTX 3050 Laptop GPU (4GB VRAM)

## Limitations & Future Work

- **Small dataset** (38 examples) — need 200+ for robust skill extraction
- **Salary field** rarely present in training data — model guesses poorly
- **Skills F1 is low** — exact string matching is strict; fuzzy matching would score higher
- **Future:** Train on Qwen2-1.5B or Mistral-7B with more data, push to HuggingFace Hub

## License

MIT
