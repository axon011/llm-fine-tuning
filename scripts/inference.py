"""
Run inference with the fine-tuned JD extractor model.

Usage:
    python scripts/inference.py --input "paste JD text here"
    python scripts/inference.py --file data/raw/sunhat.txt
    python scripts/inference.py --interactive
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = "Extract structured information from the following job description. Return valid JSON with these fields: title, company, location, work_model, seniority, required_skills (list), nice_to_have (list), salary, language."


def load_model(base_model: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    print(f"Loading {base_model} + adapter from {adapter_path}...")

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

    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer


def extract(model, tokenizer, jd_text: str, max_new_tokens: int = 512) -> str:
    """Run inference on a single JD."""
    prompt = (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{jd_text.strip()}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="google/gemma-2-2b")
    p.add_argument("--adapter", default="output/jd-extractor-lora")
    p.add_argument("--input", type=str, help="JD text to extract from")
    p.add_argument("--file", type=str, help="Path to JD text file")
    p.add_argument("--interactive", action="store_true", help="Interactive mode")
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.adapter)

    if args.file:
        jd_text = open(args.file, encoding="utf-8").read()
        result = extract(model, tokenizer, jd_text)
        print(result)

    elif args.input:
        result = extract(model, tokenizer, args.input)
        print(result)

    elif args.interactive:
        print("Interactive mode. Paste JD text, then press Enter twice to extract.")
        print("Type 'quit' to exit.\n")
        while True:
            lines = []
            print("--- Paste JD (double Enter to submit) ---")
            while True:
                line = input()
                if line == "":
                    if lines and lines[-1] == "":
                        break
                    lines.append(line)
                elif line.lower() == "quit":
                    return
                else:
                    lines.append(line)

            jd_text = "\n".join(lines).strip()
            if jd_text:
                result = extract(model, tokenizer, jd_text)
                print("\n--- Extracted ---")
                print(result)
                print()

    else:
        print("Provide --input, --file, or --interactive")


if __name__ == "__main__":
    main()
