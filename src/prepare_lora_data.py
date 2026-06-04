"""
Format CVE records as chat-style JSONL for MLX-LM LoRA fine-tuning.

The target lists the eight CVSS components first, then CWE, score and
severity. The model therefore commits to the components before it writes a
score, and we can check whether its score agrees with its own components.

Output (in data/lora/):
  train.jsonl  all 2022-2023 training CVEs (optionally rebalanced)
  valid.jsonl  a fixed sample of validation CVEs, for the training loss curve

Usage:
  python src/prepare_lora_data.py
  python src/prepare_lora_data.py --oversample-low 5 --out data/lora_bal
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import COMPONENTS
from prepare_splits import make_splits

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_DESC_CHARS = 2000  # ~500 tokens; affects <1% of CVEs
MAX_TRAIN_TOKENS = 768  # drop longer training examples (1.4%); keeps peak GPU memory bounded

USER_TEMPLATE = (
    "You are a cybersecurity expert. Given the CVE description below, predict the NVD enrichment fields.\n"
    "Return ONLY a JSON object with no extra text.\n\n"
    "CVE Description:\n{description}"
)


def clip(desc: str) -> str:
    return desc if len(desc) <= MAX_DESC_CHARS else desc[:MAX_DESC_CHARS] + " ..."


def target(row) -> str:
    out = {f: row[f] for f in COMPONENTS}
    out["cwe"] = row["cwe"]
    out["cvss_score"] = float(row["cvss_score"])
    out["severity"] = row["severity"]
    return json.dumps(out)


def row_to_example(row) -> dict:
    return {"messages": [
        {"role": "user", "content": USER_TEMPLATE.format(description=clip(row["description"]))},
        {"role": "assistant", "content": target(row)},
    ]}


def n_tokens(examples: list[dict]) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    texts = [tok.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def write_jsonl(df: pd.DataFrame, path: Path, max_tokens: int = 0) -> None:
    examples = [row_to_example(row) for _, row in df.iterrows()]
    if max_tokens:
        lengths = n_tokens(examples)
        kept = [e for e, n in zip(examples, lengths) if n <= max_tokens]
        print(f"  dropped {len(examples) - len(kept):,} examples over {max_tokens} tokens")
        examples = kept
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"  wrote {len(examples):,} examples -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lora"))
    ap.add_argument("--oversample-low", type=int, default=1, help="repeat LOW examples this many times")
    ap.add_argument("--n-valid", type=int, default=400)
    args = ap.parse_args()

    splits = make_splits()
    train, val = splits["train"], splits["val"]
    if args.oversample_low > 1:
        low = train[train["severity"] == "LOW"]
        train = pd.concat([train] + [low] * (args.oversample_low - 1))
    train = train.sample(frac=1.0, random_state=42)  # shuffle once, fixed seed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"severity mix: {train['severity'].value_counts(normalize=True).round(3).to_dict()}")
    write_jsonl(train, out / "train.jsonl", MAX_TRAIN_TOKENS)
    write_jsonl(val.sample(args.n_valid, random_state=0), out / "valid.jsonl", MAX_TRAIN_TOKENS)


if __name__ == "__main__":
    main()
