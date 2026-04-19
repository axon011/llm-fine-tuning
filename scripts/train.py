"""
QLoRA fine-tuning script for JD structured extraction.

Fine-tunes Mistral-7B (4-bit quantized) with LoRA adapters
on a custom job description dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --base_model mistralai/Mistral-7B-v0.3 --epochs 3

For Colab, use notebooks/fine_tune_colab.ipynb instead.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="google/gemma-2-2b")
    p.add_argument("--dataset", default="data/processed/train.jsonl")
    p.add_argument("--output_dir", default="output/jd-extractor-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--wandb_project", default="jd-extractor")
    return p.parse_args()


def format_prompt(example: dict) -> str:
    """Format a single example into the chat template."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def main():
    args = parse_args()

    print(f"[1/5] Loading base model: {args.base_model}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    print(f"[2/5] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"[3/5] Loading dataset: {args.dataset}")

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   {len(dataset)} training examples")

    # Format prompts
    dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset.column_names,
    )

    print(f"[4/5] Starting training ({args.epochs} epochs)")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=False,
        bf16=False,
        logging_steps=5,
        save_strategy="epoch",
        warmup_steps=5,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="adamw_torch",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"[5/5] Saving adapter to {args.output_dir}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config for reproducibility
    config = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "trainable_params": trainable,
        "total_params": total,
        "dataset_size": len(dataset),
    }
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
