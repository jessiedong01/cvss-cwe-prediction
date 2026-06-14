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

USER_TEMPLATE = """You are a cybersecurity expert. Given the CVE description below, predict the NVD enrichment fields.
Return ONLY a JSON object with no extra text.

CVE Description:
{description}"""

ASSISTANT_TEMPLATE = """{{"cvss_score": {cvss_score}, "severity": "{severity}", "cwe": "{cwe}", "attack_vector": "{attack_vector}", "attack_complexity": "{attack_complexity}", "privileges_required": "{privileges_required}", "user_interaction": "{user_interaction}", "scope": "{scope}", "confidentiality": "{confidentiality}", "integrity": "{integrity}", "availability": "{availability}", "cvss_vector": "{cvss_vector}"}}"""


def row_to_example(row) -> dict:
    user_content = USER_TEMPLATE.format(description=row["description"])
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


def write_jsonl(rows, path: Path) -> None:
    with open(path, "w") as f:
        for _, row in rows.iterrows():
            f.write(json.dumps(row_to_example(row)) + "\n")
    print(f"  Wrote {len(rows):,} examples -> {path.name}")


def prepare(max_train: int = 0) -> None:
    splits = make_splits()
    train, val = splits["train"], splits["val"]

    if max_train > 0:
        train = train.head(max_train)

    print(f"\nPreparing LoRA training data ...")
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val,   DATA_DIR / "valid.jsonl")
    print("Done. Train with:")
    print("  python -m mlx_lm.lora --config scripts/lora_config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=0,
                        help="cap on training examples (0 = all)")
    args = parser.parse_args()
    prepare(args.max_train)
