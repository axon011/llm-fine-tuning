"""
QLoRA Fine-Tuning on Kaggle T4 GPU
===================================
This script runs as a Kaggle notebook. It:
1. Loads the LinkedIn Jobs & Skills dataset (already on Kaggle)
2. Filters for AI/ML/engineering roles
3. Converts to Alpaca instruction format
4. Combines with our hand-labeled training data
5. Fine-tunes Qwen2-0.5B with QLoRA
6. Evaluates on held-out test set
7. Saves adapter to Kaggle output

Usage: Upload as Kaggle notebook with GPU T4x2 accelerator enabled.
Dataset: Add "asaniczka/1-3m-linkedin-jobs-and-skills-2024" as input.
"""

# %% [markdown]
# # QLoRA Fine-Tuning: Job Description Structured Extractor
# Train Qwen2-0.5B to extract structured JSON from job descriptions.

# %% Install dependencies
# !pip install -q transformers peft bitsandbytes trl datasets accelerate

# %% Imports
import json
import random
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# %% Config
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "/kaggle/working/jd-extractor-qwen-0.5b-v3"
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
MAX_SEQ_LEN = 1024
SEED = 42

random.seed(SEED)

# %% [markdown]
# ## Step 1: Load LinkedIn Dataset & Create Training Data

# %% Load LinkedIn data
LINKEDIN_PATH = "/kaggle/input/1-3m-linkedin-jobs-and-skills-2024"

# Load job postings with structured fields
postings = pd.read_csv(f"{LINKEDIN_PATH}/linkedin_job_postings.csv", low_memory=False)
print(f"Total LinkedIn postings: {len(postings)}")

# Load skills mapping
job_skills = pd.read_csv(f"{LINKEDIN_PATH}/job_skills.csv", low_memory=False)
print(f"Total skill entries: {len(job_skills)}")

# %% Filter for AI/ML/Engineering roles
AI_KEYWORDS = [
    "machine learning", "artificial intelligence", "ai engineer",
    "data scientist", "ml engineer", "nlp", "deep learning",
    "llm", "computer vision", "mlops", "data engineer",
    "python developer", "backend engineer", "software engineer",
]

# Filter by title containing AI/ML keywords
mask = postings["title"].str.lower().fillna("").apply(
    lambda t: any(kw in t for kw in AI_KEYWORDS)
)
ai_postings = postings[mask].copy()
print(f"AI/ML filtered postings: {len(ai_postings)}")

# Further filter: must have description
ai_postings = ai_postings[ai_postings["description"].str.len() > 200].copy()
print(f"With description > 200 chars: {len(ai_postings)}")

# %% Merge skills
skills_grouped = job_skills.groupby("job_link")["skill"].apply(list).reset_index()
skills_grouped.columns = ["job_link", "skills_list"]

ai_postings = ai_postings.merge(skills_grouped, on="job_link", how="left")
ai_postings["skills_list"] = ai_postings["skills_list"].apply(
    lambda x: x if isinstance(x, list) else []
)

# Keep only rows with at least 2 skills
ai_postings = ai_postings[ai_postings["skills_list"].apply(len) >= 2].copy()
print(f"With 2+ skills: {len(ai_postings)}")

# %% Sample and convert to Alpaca format
SAMPLE_SIZE = 300
if len(ai_postings) > SAMPLE_SIZE:
    ai_sample = ai_postings.sample(n=SAMPLE_SIZE, random_state=SEED)
else:
    ai_sample = ai_postings

INSTRUCTION = (
    "Extract structured information from the following job description. "
    "Return valid JSON with these fields: title, company, location, "
    "work_model, seniority, required_skills (list), nice_to_have (list), "
    "salary, language."
)


def infer_work_model(text):
    text_lower = text.lower() if isinstance(text, str) else ""
    if "remote" in text_lower:
        return "remote"
    if "hybrid" in text_lower:
        return "hybrid"
    return "on-site"


def infer_seniority(title):
    title_lower = title.lower() if isinstance(title, str) else ""
    if any(w in title_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
        return "senior"
    if any(w in title_lower for w in ["junior", "jr.", "entry", "intern", "graduate"]):
        return "entry-level"
    return "mid-level"


def row_to_alpaca(row):
    description = str(row.get("description", ""))[:3000]  # Truncate long descriptions

    output_json = {
        "title": str(row.get("title", "")),
        "company": str(row.get("company_name", "")),
        "location": str(row.get("location", "")),
        "work_model": infer_work_model(description),
        "seniority": infer_seniority(row.get("title", "")),
        "required_skills": row.get("skills_list", [])[:10],
        "nice_to_have": [],
        "salary": str(row.get("salary", "")) if pd.notna(row.get("salary")) else "",
        "language": "English",
    }

    return {
        "instruction": INSTRUCTION,
        "input": description,
        "output": json.dumps(output_json, indent=2),
    }


linkedin_examples = [row_to_alpaca(row) for _, row in ai_sample.iterrows()]
print(f"LinkedIn examples created: {len(linkedin_examples)}")

# %% Load our hand-labeled data
HAND_LABELED_DATA = """PASTE_TRAIN_SPLIT_HERE"""

# If running with uploaded data file:
hand_labeled = []
hand_labeled_path = "/kaggle/input/jd-hand-labeled/train_split.jsonl"
if os.path.exists(hand_labeled_path):
    with open(hand_labeled_path, encoding="utf-8") as f:
        hand_labeled = [json.loads(line) for line in f]
    print(f"Hand-labeled examples loaded: {len(hand_labeled)}")
else:
    print("No hand-labeled data found, using LinkedIn data only")

# %% Combine datasets
all_train = hand_labeled + linkedin_examples
random.shuffle(all_train)

# Split: 90% train, 10% eval
split_idx = int(len(all_train) * 0.9)
train_data = all_train[:split_idx]
eval_data = all_train[split_idx:]

print(f"\nFinal dataset: {len(train_data)} train, {len(eval_data)} eval")

# %% Save processed data
Path("/kaggle/working").mkdir(exist_ok=True)
with open("/kaggle/working/train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")
with open("/kaggle/working/eval.jsonl", "w") as f:
    for item in eval_data:
        f.write(json.dumps(item) + "\n")

# %% [markdown]
# ## Step 2: Fine-Tune with QLoRA

# %% Format prompt
def format_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


# %% Load model with 4-bit quantization
print(f"\n[1/5] Loading base model: {BASE_MODEL}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

# %% Apply LoRA
print(f"[2/5] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# %% Load dataset
print(f"[3/5] Loading dataset ({len(train_data)} examples)")

train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(
    lambda x: {"text": format_prompt(x)},
    remove_columns=train_dataset.column_names,
)

# %% Train
print(f"[4/5] Training ({EPOCHS} epochs, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM})")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=10,
    lr_scheduler_type="cosine",
    report_to="none",
    optim="adamw_torch",
    max_length=MAX_SEQ_LEN,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()

# %% Save
print(f"[5/5] Saving adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config = {
    "base_model": BASE_MODEL,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "max_seq_len": MAX_SEQ_LEN,
    "trainable_params": trainable,
    "total_params": total,
    "train_size": len(train_data),
    "eval_size": len(eval_data),
    "linkedin_examples": len(linkedin_examples),
    "hand_labeled_examples": len(hand_labeled),
}
with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
    json.dump(config, f, indent=2)

# %% [markdown]
# ## Step 3: Evaluate on held-out data

# %% Evaluate
print("\n" + "=" * 50)
print("EVALUATION")
print("=" * 50)


def generate(model, tokenizer, jd_text):
    prompt = (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{jd_text.strip()[:2000]}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response


def parse_json_safe(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for char in ["{", "["]:
            idx = text.find(char)
            if idx >= 0:
                try:
                    return json.loads(text[idx:])
                except json.JSONDecodeError:
                    pass
    return None


json_valid = 0
json_invalid = 0
field_scores = []

STRING_FIELDS = ["title", "company", "location", "work_model", "seniority", "salary", "language"]
LIST_FIELDS = ["required_skills", "nice_to_have"]

for i, example in enumerate(eval_data):
    raw_output = generate(model, tokenizer, example["input"])
    parsed = parse_json_safe(raw_output)
    expected = json.loads(example["output"])

    if parsed:
        json_valid += 1
        scores = {}
        for field in STRING_FIELDS:
            pred = str(parsed.get(field, "")).lower().strip()
            exp = str(expected.get(field, "")).lower().strip()
            if pred == exp:
                scores[field] = 1.0
            elif exp in pred or pred in exp:
                scores[field] = 0.5
            else:
                scores[field] = 0.0
        for field in LIST_FIELDS:
            pred_set = {s.lower().strip() for s in (parsed.get(field) or [])}
            exp_set = {s.lower().strip() for s in (expected.get(field) or [])}
            if pred_set and exp_set:
                matches = pred_set & exp_set
                p = len(matches) / len(pred_set)
                r = len(matches) / len(exp_set)
                scores[field] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            else:
                scores[field] = 0.0
        field_scores.append(scores)
    else:
        json_invalid += 1

    print(f"  [{i+1}/{len(eval_data)}] JSON valid: {json_valid}/{json_valid + json_invalid}")

# %% Print results
print(f"\n{'=' * 50}")
print(f"RESULTS (on {len(eval_data)} eval examples)")
print(f"{'=' * 50}")
print(f"JSON validity: {json_valid}/{json_valid + json_invalid} ({100*json_valid/(json_valid+json_invalid):.0f}%)")

if field_scores:
    print(f"\nField accuracy (avg):")
    for field in STRING_FIELDS:
        avg = sum(s[field] for s in field_scores) / len(field_scores)
        print(f"  {field}: {avg:.2f}")
    for field in LIST_FIELDS:
        avg = sum(s[field] for s in field_scores) / len(field_scores)
        print(f"  {field} (F1): {avg:.2f}")

# Save results
results = {
    "json_valid": json_valid,
    "json_invalid": json_invalid,
    "total": len(eval_data),
    "field_scores_avg": {
        field: sum(s[field] for s in field_scores) / len(field_scores)
        for field in STRING_FIELDS + LIST_FIELDS
    } if field_scores else {},
}
with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAdapter saved to: {OUTPUT_DIR}")
print("Download from Kaggle Output tab!")
