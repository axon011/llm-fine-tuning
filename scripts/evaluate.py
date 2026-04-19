"""
Evaluate fine-tuned model vs base model on held-out JDs.

Compares:
1. JSON validity (can the output be parsed?)
2. Field accuracy (do extracted fields match labels?)
3. Skills extraction (precision/recall on skill lists)

Usage:
    python scripts/evaluate.py --adapter output/jd-extractor-lora
    python scripts/evaluate.py --adapter output/jd-extractor-lora --test_file data/processed/test.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = "Extract structured information from the following job description. Return valid JSON with these fields: title, company, location, work_model, seniority, required_skills (list), nice_to_have (list), salary, language."

STRING_FIELDS = ["title", "company", "location", "work_model", "seniority", "salary", "language"]
LIST_FIELDS = ["required_skills", "nice_to_have"]


def load_model(base_model: str, adapter_path: str = None):
    """Load model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer


def generate(model, tokenizer, jd_text: str) -> str:
    """Generate extraction for a single JD."""
    prompt = (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{jd_text.strip()}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def parse_json_safe(text: str) -> dict | None:
    """Try to parse JSON from model output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass

    return None


def score_string_field(predicted: str, expected: str) -> float:
    """Score a string field (exact or partial match)."""
    if not predicted or not expected:
        return 0.0
    if predicted.lower().strip() == expected.lower().strip():
        return 1.0
    if expected.lower() in predicted.lower() or predicted.lower() in expected.lower():
        return 0.5
    return 0.0


def score_list_field(predicted: list, expected: list) -> dict:
    """Score a list field (precision, recall, F1)."""
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_lower = {s.lower().strip() for s in predicted}
    exp_lower = {s.lower().strip() for s in expected}

    if not pred_lower or not exp_lower:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matches = pred_lower & exp_lower
    precision = len(matches) / len(pred_lower)
    recall = len(matches) / len(exp_lower)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_single(predicted: dict, expected: dict) -> dict:
    """Evaluate a single prediction against expected output."""
    scores = {}

    for field in STRING_FIELDS:
        pred_val = predicted.get(field, "")
        exp_val = expected.get(field, "")
        scores[field] = score_string_field(str(pred_val), str(exp_val))

    for field in LIST_FIELDS:
        pred_val = predicted.get(field, [])
        exp_val = expected.get(field, [])
        scores[field] = score_list_field(pred_val, exp_val)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-2-2b")
    parser.add_argument("--adapter", default="output/jd-extractor-lora")
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    # Load test data
    test_data = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"Loaded {len(test_data)} test examples")

    # Evaluate fine-tuned model
    print(f"\nLoading fine-tuned model ({args.adapter})...")
    ft_model, ft_tokenizer = load_model(args.base_model, args.adapter)

    results = {
        "json_valid": 0,
        "json_invalid": 0,
        "field_scores": [],
        "examples": [],
    }

    for i, example in enumerate(test_data):
        jd_text = example["input"]
        expected = json.loads(example["output"])

        raw_output = generate(ft_model, ft_tokenizer, jd_text)
        parsed = parse_json_safe(raw_output)

        if parsed:
            results["json_valid"] += 1
            scores = evaluate_single(parsed, expected)
            results["field_scores"].append(scores)
        else:
            results["json_invalid"] += 1
            scores = None

        results["examples"].append({
            "input_preview": jd_text[:100],
            "expected": expected,
            "raw_output": raw_output[:500],
            "parsed": parsed,
            "scores": scores,
        })

        total = results["json_valid"] + results["json_invalid"]
        print(f"  [{i+1}/{len(test_data)}] JSON valid: {results['json_valid']}/{total}")

    # Aggregate metrics
    n = len(results["field_scores"])
    if n > 0:
        agg = {}
        for field in STRING_FIELDS:
            agg[field] = sum(s[field] for s in results["field_scores"]) / n
        for field in LIST_FIELDS:
            agg[field] = {
                "precision": sum(s[field]["precision"] for s in results["field_scores"]) / n,
                "recall": sum(s[field]["recall"] for s in results["field_scores"]) / n,
                "f1": sum(s[field]["f1"] for s in results["field_scores"]) / n,
            }
        results["aggregate"] = agg

    # Save
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"JSON validity: {results['json_valid']}/{results['json_valid'] + results['json_invalid']}")
    if "aggregate" in results:
        print(f"\nField accuracy (avg):")
        for field in STRING_FIELDS:
            print(f"  {field}: {results['aggregate'][field]:.2f}")
        for field in LIST_FIELDS:
            f1 = results['aggregate'][field]['f1']
            print(f"  {field}: F1={f1:.2f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    import os
    main()
