"""Build a clean Kaggle notebook from cell definitions."""
import json

cells = []

def code(source):
    cells.append({"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None})

def md(source):
    cells.append({"cell_type": "markdown", "source": source, "metadata": {}})

# ── Cell 0: Install ──
code("!pip install -q transformers peft trl datasets accelerate")

# ── Cell 1: Title ──
md("# QLoRA Fine-Tuning: Job Description Structured Extractor\nTrain Qwen2-0.5B to extract structured JSON from job descriptions using LinkedIn dataset.")

# ── Cell 2: Imports + Config ──
code("""import json
import random
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

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

INSTRUCTION = (
    "Extract structured information from the following job description. "
    "Return valid JSON with these fields: title, company, location, "
    "work_model, seniority, required_skills (list), nice_to_have (list), "
    "salary, language."
)

AI_KEYWORDS = [
    "machine learning", "artificial intelligence", "ai engineer",
    "ml engineer", "data scientist", "nlp", "deep learning",
    "llm", "computer vision", "mlops", "data engineer",
    "python developer", "backend engineer", "software engineer",
    "ai developer", "research engineer", "applied scientist",
    "prompt engineer", "langchain", "rag",
]

print("Imports done")""")

# ── Cell 3: Load Dataset ──
md("## Step 1: Load LinkedIn Dataset")

code("""import zipfile

LINKEDIN_PATH = None

if os.path.exists("/kaggle/input"):
    for d in os.listdir("/kaggle/input"):
        full = os.path.join("/kaggle/input", d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "linkedin_job_postings.csv")):
            LINKEDIN_PATH = full
            break

if LINKEDIN_PATH is None:
    print("Dataset not mounted. Downloading...")
    DATA_DIR = "/kaggle/working/data"
    os.makedirs(DATA_DIR, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("asaniczka/1-3m-linkedin-jobs-and-skills-2024", path=DATA_DIR, unzip=True)
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".zip") and zipfile.is_zipfile(fpath):
            print("Unzipping " + fname)
            with zipfile.ZipFile(fpath, "r") as z:
                z.extractall(DATA_DIR)
            os.remove(fpath)
    LINKEDIN_PATH = DATA_DIR

print("Path: " + LINKEDIN_PATH)
print("Files: " + str(os.listdir(LINKEDIN_PATH)))

postings = pd.read_csv(os.path.join(LINKEDIN_PATH, "linkedin_job_postings.csv"), low_memory=False)
print("Postings: " + str(len(postings)))

job_skills_df = pd.read_csv(os.path.join(LINKEDIN_PATH, "job_skills.csv"), low_memory=False)
print("Skills rows: " + str(len(job_skills_df)))

job_summary_df = pd.read_csv(os.path.join(LINKEDIN_PATH, "job_summary.csv"), low_memory=False)
print("Summaries: " + str(len(job_summary_df)))""")

# ── Cell 5: Filter + Merge + Convert ──
md("## Step 2: Filter AI/ML Jobs and Convert to Training Format")

code("""# Filter by job title
mask = postings["job_title"].str.lower().fillna("").apply(
    lambda t: any(kw in t for kw in AI_KEYWORDS)
)
ai_postings = postings[mask].copy()
print("AI/ML jobs: " + str(len(ai_postings)))

# Merge skills (comma-separated string -> list)
job_skills_df["skills_list"] = job_skills_df["job_skills"].fillna("").apply(
    lambda x: [s.strip() for s in x.split(",") if s.strip()]
)
ai_postings = ai_postings.merge(job_skills_df[["job_link", "skills_list"]], on="job_link", how="left")
ai_postings["skills_list"] = ai_postings["skills_list"].apply(lambda x: x if isinstance(x, list) else [])

# Merge descriptions
ai_postings = ai_postings.merge(job_summary_df[["job_link", "job_summary"]], on="job_link", how="left")
ai_postings = ai_postings[ai_postings["job_summary"].str.len() > 200].copy()
ai_postings = ai_postings[ai_postings["skills_list"].apply(len) >= 2].copy()
print("With desc + skills: " + str(len(ai_postings)))

# Sample 300
SAMPLE_SIZE = 300
if len(ai_postings) > SAMPLE_SIZE:
    ai_sample = ai_postings.sample(n=SAMPLE_SIZE, random_state=SEED)
else:
    ai_sample = ai_postings


def infer_work_model(job_type):
    if not isinstance(job_type, str):
        return "on-site"
    jt = job_type.lower()
    if "remote" in jt:
        return "remote"
    if "hybrid" in jt:
        return "hybrid"
    return "on-site"


def infer_seniority(job_level):
    if not isinstance(job_level, str):
        return "mid-level"
    jl = job_level.lower()
    if "senior" in jl or "director" in jl or "lead" in jl:
        return "senior"
    if "entry" in jl or "intern" in jl or "associate" in jl:
        return "entry-level"
    return "mid-level"


def row_to_alpaca(row):
    desc = str(row.get("job_summary", ""))[:3000]
    output_json = {
        "title": str(row.get("job_title", "")),
        "company": str(row.get("company", "")),
        "location": str(row.get("job_location", "")),
        "work_model": infer_work_model(row.get("job_type", "")),
        "seniority": infer_seniority(row.get("job_level", "")),
        "required_skills": row.get("skills_list", [])[:10],
        "nice_to_have": [],
        "salary": "",
        "language": "English",
    }
    return {
        "instruction": INSTRUCTION,
        "input": desc,
        "output": json.dumps(output_json, indent=2),
    }


linkedin_examples = [row_to_alpaca(row) for _, row in ai_sample.iterrows()]
print("LinkedIn examples: " + str(len(linkedin_examples)))

# Split 90/10
all_data = linkedin_examples
random.shuffle(all_data)
split_idx = int(len(all_data) * 0.9)
train_data = all_data[:split_idx]
eval_data = all_data[split_idx:]
print("Train: " + str(len(train_data)) + ", Eval: " + str(len(eval_data)))""")

# ── Cell 7: Train ──
md("## Step 3: Fine-Tune with LoRA")

code("""def format_prompt(example):
    return (
        "### Instruction:\\n" + example["instruction"] + "\\n\\n"
        + "### Input:\\n" + example["input"] + "\\n\\n"
        + "### Response:\\n" + example["output"]
    )

print("[1/5] Loading model: " + BASE_MODEL)
if torch.cuda.is_available():
    print("GPU: " + torch.cuda.get_device_name(0))

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("[2/5] Applying LoRA (r=" + str(LORA_R) + ", alpha=" + str(LORA_ALPHA) + ")")
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print("   Trainable: " + str(trainable) + " / " + str(total))

print("[3/5] Preparing dataset (" + str(len(train_data)) + " examples)")
train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(
    lambda x: {"text": format_prompt(x)}, remove_columns=train_dataset.column_names,
)

print("[4/5] Training (" + str(EPOCHS) + " epochs)")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, fp16=False, logging_steps=10, save_strategy="epoch",
    warmup_steps=10, lr_scheduler_type="cosine", report_to="none",
    optim="adamw_torch", max_length=MAX_SEQ_LEN,
)
trainer = SFTTrainer(model=model, train_dataset=train_dataset, args=training_args, processing_class=tokenizer)
trainer.train()

print("[5/5] Saving adapter")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config_dict = {
    "base_model": BASE_MODEL, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
    "lr": LR, "max_seq_len": MAX_SEQ_LEN, "trainable_params": trainable,
    "total_params": total, "train_size": len(train_data), "eval_size": len(eval_data),
}
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)
print("Done training!")""")

# ── Cell 9: Evaluate ──
md("## Step 4: Evaluate")

code("""print("=" * 50)
print("EVALUATION on " + str(len(eval_data)) + " examples")
print("=" * 50)

STRING_FIELDS = ["title", "company", "location", "work_model", "seniority", "salary", "language"]
LIST_FIELDS = ["required_skills", "nice_to_have"]


def generate(mdl, tok, jd_text):
    prompt = "### Instruction:\\n" + INSTRUCTION + "\\n\\n### Input:\\n" + jd_text.strip()[:2000] + "\\n\\n### Response:\\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(mdl.device)
    with torch.no_grad():
        outputs = mdl.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tok.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response


def parse_json_safe(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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

    print("  [" + str(i + 1) + "/" + str(len(eval_data)) + "] JSON valid: " + str(json_valid) + "/" + str(json_valid + json_invalid))

print()
print("=" * 50)
total_eval = json_valid + json_invalid
print("JSON validity: " + str(json_valid) + "/" + str(total_eval) + " (" + str(round(100 * json_valid / total_eval)) + "%)")
if field_scores:
    n = len(field_scores)
    print("Field accuracy (avg):")
    for field in STRING_FIELDS:
        avg = sum(s[field] for s in field_scores) / n
        print("  " + field + ": " + str(round(avg, 2)))
    for field in LIST_FIELDS:
        avg = sum(s[field] for s in field_scores) / n
        print("  " + field + " (F1): " + str(round(avg, 2)))

results = {
    "json_valid": json_valid, "json_invalid": json_invalid, "total": len(eval_data),
    "field_scores_avg": {
        f: round(sum(s[f] for s in field_scores) / len(field_scores), 3) for f in STRING_FIELDS + LIST_FIELDS
    } if field_scores else {},
}
with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Adapter + results saved to: " + OUTPUT_DIR)""")

# ── Build notebook ──
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

# Verify all cells compile
errors = 0
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and not cell["source"].startswith("!"):
        try:
            compile(cell["source"], "cell_" + str(i), "exec")
        except SyntaxError as e:
            print("SYNTAX ERROR cell " + str(i) + ": " + str(e))
            errors += 1

if errors == 0:
    with open("notebooks/kaggle_train.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    with open("notebooks/kaggle-push/kaggle_train.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook built: " + str(len(cells)) + " cells, 0 errors")
else:
    print("FAILED: " + str(errors) + " syntax errors")
