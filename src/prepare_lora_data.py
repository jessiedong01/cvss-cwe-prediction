"""
Format CVE records as chat-style JSONL for MLX-LM LoRA fine-tuning.

Output files (in data/):
  train.jsonl  — 80% of 2022-2023 CVEs
  valid.jsonl  — 20% of 2022-2023 CVEs

Each line:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
  python src/prepare_lora_data.py
  python src/prepare_lora_data.py --max-train 20000   # cap for quick experiments
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_splits import make_splits

DATA_DIR = Path(__file__).parent.parent / "data"

USER_TEMPLATES = {
    "default": (
        "You are a cybersecurity expert. Given the CVE description below, predict the NVD enrichment fields.\n"
        "Return ONLY a JSON object with no extra text.\n\n"
        "CVE Description:\n{description}"
    ),
    "minimal": (
        "Analyze this CVE and return JSON with these exact fields: "
        "cvss_score (0.0-10.0), severity (CRITICAL/HIGH/MEDIUM/LOW), cwe (CWE-NNN), "
        "attack_vector, attack_complexity, privileges_required, user_interaction, "
        "scope, confidentiality, integrity, availability.\n\nCVE: {description}"
    ),
    "structured": (
        "You are a senior cybersecurity analyst. Think step by step before producing your final JSON.\n\n"
        "CVE Description:\n{description}\n\n"
        "Step 1 — Identify the vulnerability type and CWE category.\n"
        "Step 2 — Assess the attack vector, complexity, required privileges, and user interaction.\n"
        "Step 3 — Estimate the impact on confidentiality, integrity, and availability, "
        "and whether scope changes.\n"
        "Step 4 — Derive the CVSS v3 base score and overall severity tier.\n\n"
        "Output ONLY JSON on its own line after your reasoning (no text after the closing brace)."
    ),
}

# Default alias used by evaluate_lora.py
USER_TEMPLATE = USER_TEMPLATES["default"]

ASSISTANT_TEMPLATE = """{{"cvss_score": {cvss_score}, "severity": "{severity}", "cwe": "{cwe}", "attack_vector": "{attack_vector}", "attack_complexity": "{attack_complexity}", "privileges_required": "{privileges_required}", "user_interaction": "{user_interaction}", "scope": "{scope}", "confidentiality": "{confidentiality}", "integrity": "{integrity}", "availability": "{availability}", "cvss_vector": "{cvss_vector}"}}"""


def row_to_example(row, prompt_style: str = "default") -> dict:
    user_content = USER_TEMPLATES[prompt_style].format(description=row["description"])
    assistant_content = ASSISTANT_TEMPLATE.format(
        cvss_score=row["cvss_score"],
        severity=row["severity"],
        cwe=row["cwe"],
        attack_vector=row["attack_vector"],
        attack_complexity=row["attack_complexity"],
        privileges_required=row["privileges_required"],
        user_interaction=row["user_interaction"],
        scope=row["scope"],
        confidentiality=row["confidentiality"],
        integrity=row["integrity"],
        availability=row["availability"],
        cvss_vector=row["cvss_vector"],
    )
    return {"messages": [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]}


def write_jsonl(rows, path: Path, prompt_style: str = "default") -> None:
    with open(path, "w") as f:
        for _, row in rows.iterrows():
            f.write(json.dumps(row_to_example(row, prompt_style=prompt_style)) + "\n")
    print(f"  Wrote {len(rows):,} examples -> {path.name}")


def prepare(max_train: int = 0, prompt_style: str = "default") -> None:
    splits = make_splits()
    train, val = splits["train"], splits["val"]

    if max_train > 0:
        train = train.head(max_train)

    print(f"\nPreparing LoRA training data (prompt_style={prompt_style}) ...")
    write_jsonl(train, DATA_DIR / "train.jsonl", prompt_style=prompt_style)
    write_jsonl(val,   DATA_DIR / "valid.jsonl", prompt_style=prompt_style)
    print("Done. Train with:")
    print("  python -m mlx_lm.lora --config scripts/lora_config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=0,
                        help="cap on training examples (0 = all)")
    parser.add_argument("--prompt-style", default="default",
                        choices=list(USER_TEMPLATES.keys()),
                        help="prompt format variant to use for LoRA training data")
    args = parser.parse_args()
    prepare(args.max_train, prompt_style=args.prompt_style)
