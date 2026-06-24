"""
Evaluate the fine-tuned model (and optionally the base model) on held-out JDs.

Measures:
1. JSON validity (can the output be parsed?)
2. Field accuracy (do extracted fields match labels?)
3. Skills extraction (precision/recall on skill lists)

By default it also evaluates the un-tuned base model and prints a side-by-side
comparison so the effect of fine-tuning is visible. Pass --no-compare-base to skip.

Usage:
    python scripts/evaluate.py --adapter output/jd-extractor-qwen-0.5b-v2
    python scripts/evaluate.py --adapter output/jd-extractor-qwen-0.5b-v2 --no-compare-base
"""

import argparse
import json
import os
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


def run_eval(model, tokenizer, test_data: list, label: str) -> dict:
    """Run the full eval loop for one model and return a results dict."""
    results = {
        "json_valid": 0,
        "json_invalid": 0,
        "field_scores": [],
        "examples": [],
    }

    for i, example in enumerate(test_data):
        jd_text = example["input"]
        expected = json.loads(example["output"])

        raw_output = generate(model, tokenizer, jd_text)
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
        print(f"  [{label}] [{i+1}/{len(test_data)}] JSON valid: {results['json_valid']}/{total}")

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

    return results


def _metric_value(results: dict, field: str) -> float:
    """Pull a single comparable number for a field (F1 for list fields)."""
    agg = results.get("aggregate")
    if not agg:
        return 0.0
    if field in LIST_FIELDS:
        return agg[field]["f1"]
    return agg[field]


def print_comparison(base: dict | None, ft: dict):
    """Print a base-vs-fine-tuned comparison table (or a single summary)."""
    n_test = ft["json_valid"] + ft["json_invalid"]
    print(f"\n{'='*60}")

    if base is None:
        print(f"JSON validity: {ft['json_valid']}/{n_test}")
        if "aggregate" in ft:
            print("\nField accuracy (fine-tuned):")
            for field in STRING_FIELDS:
                print(f"  {field:16s}: {_metric_value(ft, field):.2f}")
            for field in LIST_FIELDS:
                print(f"  {field:16s}: F1={_metric_value(ft, field):.2f}")
        return

    print(f"{'Metric':18s}{'Base':>8s}{'Fine-tuned':>12s}{'Delta':>8s}")
    print(f"{'-'*46}")
    print(f"{'JSON valid':18s}{base['json_valid']:>6d}/{n_test:<2d}"
          f"{ft['json_valid']:>9d}/{n_test:<2d}"
          f"{ft['json_valid'] - base['json_valid']:>+8d}")
    for field in STRING_FIELDS + LIST_FIELDS:
        b = _metric_value(base, field)
        f = _metric_value(ft, field)
        suffix = " (F1)" if field in LIST_FIELDS else ""
        print(f"{field + suffix:18s}{b:>8.2f}{f:>12.2f}{f - b:>+8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--adapter", default="output/jd-extractor-qwen-0.5b-v2")
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--compare-base", dest="compare_base", action="store_true", default=True,
                        help="Also evaluate the un-tuned base model (default).")
    parser.add_argument("--no-compare-base", dest="compare_base", action="store_false",
                        help="Skip the base-model baseline.")
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
    ft_results = run_eval(ft_model, ft_tokenizer, test_data, label="fine-tuned")

    base_results = None
    if args.compare_base:
        # Free the fine-tuned model before loading the base model (4GB GPU).
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nLoading base model ({args.base_model}) for baseline...")
        base_model, base_tokenizer = load_model(args.base_model, adapter_path=None)
        base_results = run_eval(base_model, base_tokenizer, test_data, label="base")

    # Save (fine-tuned results, plus base baseline if computed)
    out = dict(ft_results)
    if base_results is not None:
        out["base_baseline"] = {
            "json_valid": base_results["json_valid"],
            "json_invalid": base_results["json_invalid"],
            "aggregate": base_results.get("aggregate"),
        }
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print_comparison(base_results, ft_results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
