# LLM Fine-Tuning: Job Description Structured Extractor

Fine-tuned LLMs (Gemma-2-2B for dev, Mistral-7B for production) using QLoRA to extract structured information from job descriptions. Converts raw JD text into structured JSON with title, company, skills, seniority, location, and salary.

## Why This Project

Most JD parsing relies on regex or keyword matching — brittle and domain-specific. A fine-tuned LLM handles variations in formatting, language, and structure that rule-based systems miss.

## Architecture

```
Raw Job Description (text)
        │
        ▼
┌──────────────────────┐
│  Gemma-2-2B (dev)    │
│  Mistral-7B (prod)   │
│  + LoRA (r=16, a=32) │
└──────────┬───────────┘
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

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Gemma-2-2B (dev) / Mistral-7B-v0.3 (production) |
| Fine-tuning | QLoRA (4-bit quantization + LoRA adapters) |
| Framework | PyTorch, HuggingFace Transformers, PEFT, TRL |
| Quantization | bitsandbytes (NF4) |
| Training | Google Colab (T4 GPU) / Kaggle (2x T4) |
| Tracking | Weights & Biases |
| Dataset | Custom JD dataset (instruction-tuning format) |

## Project Structure

```
llm-fine-tuning/
├── README.md
├── requirements.txt
├── data/
│   ├── prepare_dataset.py      # Convert raw JDs to training format
│   ├── raw/                    # Raw job descriptions (text files)
│   └── processed/              # Alpaca-format JSONL for training
├── scripts/
│   ├── train.py                # QLoRA fine-tuning script
│   ├── evaluate.py             # Compare base vs fine-tuned
│   └── inference.py            # Run the fine-tuned model
├── notebooks/
│   └── fine_tune_colab.ipynb   # Colab-ready notebook (run this)
├── eval/
│   └── results.json            # Evaluation metrics
└── output/
    └── jd-extractor-lora/      # Saved LoRA adapter weights
```

## Quick Start

### 1. Prepare dataset
```bash
python data/prepare_dataset.py
```

### 2. Train (Colab recommended)
Open `notebooks/fine_tune_colab.ipynb` in Google Colab and run all cells.

Or locally with GPU:
```bash
python scripts/train.py
```

### 3. Evaluate
```bash
python scripts/evaluate.py --adapter output/jd-extractor-lora
```

### 4. Inference
```bash
python scripts/inference.py --input "paste a job description here"
```

## Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| JSON validity | TBD | TBD | TBD |
| Field accuracy | TBD | TBD | TBD |
| Exact match (skills) | TBD | TBD | TBD |

## Training Details

- **Method:** QLoRA (4-bit NF4 quantization + LoRA rank 16)
- **LoRA config:** r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
- **Training:** 3 epochs, lr=2e-4, batch_size=4, gradient_accumulation=4
- **Hardware:** Google Colab T4 (16GB VRAM)
- **Training time:** ~TBD minutes
- **Trainable params:** ~0.1% of total

## Dataset

Custom dataset of job descriptions paired with structured JSON labels. Sources:
- Manually labeled JDs from job search pipeline
- Augmented with variations in formatting and language

Format (Alpaca-style):
```json
{
  "instruction": "Extract structured information from this job description as JSON.",
  "input": "<raw JD text>",
  "output": "<structured JSON>"
}
```
