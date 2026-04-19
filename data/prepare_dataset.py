"""
Prepare training dataset from raw job descriptions.

Converts raw JD text files + their structured labels into
Alpaca-format JSONL for instruction fine-tuning.

Usage:
    python data/prepare_dataset.py
"""

import json
import os
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = Path(__file__).parent / "processed" / "train.jsonl"

SYSTEM_PROMPT = "Extract structured information from the following job description. Return valid JSON with these fields: title, company, location, work_model, seniority, required_skills (list), nice_to_have (list), salary, language."


def create_training_example(jd_text: str, structured: dict) -> dict:
    """Create a single Alpaca-format training example."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input": jd_text.strip(),
        "output": json.dumps(structured, indent=2)
    }


def load_jd_pair(jd_path: Path) -> dict | None:
    """Load a JD text file and its corresponding label JSON."""
    label_path = jd_path.with_suffix(".json")
    if not label_path.exists():
        print(f"  Skipping {jd_path.name} — no matching .json label")
        return None

    jd_text = jd_path.read_text(encoding="utf-8")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    return create_training_example(jd_text, labels)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    jd_files = sorted(RAW_DIR.glob("*.txt"))
    if not jd_files:
        print(f"No .txt files found in {RAW_DIR}/")
        print("Add JD text files (.txt) and their labels (.json) to data/raw/")
        print()
        print("Example:")
        print("  data/raw/sunhat.txt     — raw JD text")
        print("  data/raw/sunhat.json    — structured label JSON")
        create_example_files()
        return

    examples = []
    for jd_path in jd_files:
        example = load_jd_pair(jd_path)
        if example:
            examples.append(example)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Created {len(examples)} training examples -> {OUTPUT_FILE}")


def create_example_files():
    """Create example files to show the expected format."""
    example_jd = RAW_DIR / "_example.txt"
    example_label = RAW_DIR / "_example.json"

    if not example_jd.exists():
        example_jd.write_text(
            "AI Engineer (m/w/d) - Startup\n"
            "Logicc\n"
            "Hamburg, Germany (Flexible: Remote, Hybrid, or on-site)\n"
            "Full-time, Permanent\n\n"
            "Required Skills:\n"
            "- Practical experience building software products with LLM integration\n"
            "- Proficiency in Python and typical AI/ML ecosystem tools\n"
            "- Understanding of modern AI product components: RAG, Tool Use, Evals\n\n"
            "Nice-to-Have:\n"
            "- C# or similar language experience\n"
            "- React, Next.js, and Tailwind expertise\n\n"
            "Salary: 55,000 - 70,000 EUR\n",
            encoding="utf-8"
        )

    if not example_label.exists():
        example_label.write_text(
            json.dumps({
                "title": "AI Engineer",
                "company": "Logicc",
                "location": "Hamburg, Germany",
                "work_model": "remote/hybrid/onsite",
                "seniority": "mid-level",
                "required_skills": ["Python", "LLM integration", "RAG", "AI/ML ecosystem"],
                "nice_to_have": ["C#", "React", "Next.js", "Tailwind"],
                "salary": "55,000-70,000 EUR",
                "language": "German"
            }, indent=2),
            encoding="utf-8"
        )

    print(f"Created example files in {RAW_DIR}/")
    print("  _example.txt  — sample JD")
    print("  _example.json — sample label")
    print()
    print("Create your own .txt + .json pairs, then rerun this script.")


if __name__ == "__main__":
    main()
