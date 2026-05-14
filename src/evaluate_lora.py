"""
Evaluate a trained LoRA adapter (Phi-3.5 or Qwen) on held-out 2024 CVEs.

Requires:  pip install mlx-lm   (Apple Silicon only)
           Trained adapter at:  adapters/phi35/   or   adapters/qwen/

Usage:
  python src/evaluate_lora.py --adapter adapters/phi35 --model microsoft/Phi-3.5-mini-instruct --n 200
  python src/evaluate_lora.py --adapter adapters/qwen  --model Qwen/Qwen2.5-0.5B-Instruct   --n 200
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prepare_splits import make_splits
from prepare_lora_data import USER_TEMPLATE
from evaluate import (
    parse_json_response, summarize, print_summary, save_results,
    CVSS_VECTOR_FIELDS,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(model_name: str, adapter_path: str):
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise ImportError("mlx-lm not installed. Run: pip install mlx-lm")

    print(f"Loading {model_name} with adapter {adapter_path} ...")
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    return model, tokenizer, generate


def query_model(desc: str, model, tokenizer, generate_fn, max_tokens: int = 512) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_TEMPLATE.format(description=desc)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    response = generate_fn(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return response


def _make_result(row, pred: dict, raw: str, model_tag: str) -> dict:
    return {
        "model":            model_tag,
        "cve_id":           row["cve_id"],
        "parsed":           bool(pred),
        "raw_response":     raw,
        "true_severity":    row["severity"],
        "pred_severity":    str(pred.get("severity", "")).upper(),
        "true_cwe":         row["cwe"],
        "pred_cwe":         str(pred.get("cwe", "")),
        "true_cvss_score":  float(row["cvss_score"]),
        "pred_cvss_score":  float(pred.get("cvss_score", 5.0)),
        "true_cvss_vector": row["cvss_vector"],
        "pred_cvss_vector": str(pred.get("cvss_vector", "")),
        "true_components":  {f: row[f] for f in CVSS_VECTOR_FIELDS},
        "pred_components": {
            "attack_vector":       str(pred.get("attack_vector", "")).upper(),
            "attack_complexity":   str(pred.get("attack_complexity", "")).upper(),
            "privileges_required": str(pred.get("privileges_required", "")).upper(),
            "user_interaction":    str(pred.get("user_interaction", "")).upper(),
            "scope":               str(pred.get("scope", "")).upper(),
            "confidentiality":     str(pred.get("confidentiality", "")).upper(),
            "integrity":           str(pred.get("integrity", "")).upper(),
            "availability":        str(pred.get("availability", "")).upper(),
        },
    }


def run(model_name: str, adapter_path: str, n_samples: int, sample_seed: int = 42) -> None:
    splits = make_splits()
    test = splits["test"]
    if n_samples > 0:
        test = test.sample(min(n_samples, len(test)), random_state=sample_seed).reset_index(drop=True)

    model, tokenizer, generate_fn = load_model(model_name, adapter_path)
    model_tag = Path(adapter_path).name

    print(f"Evaluating {model_tag} on {len(test)} examples ...")
    results = []

    for _, row in tqdm(test.iterrows(), total=len(test)):
        raw = ""
        pred = {}
        try:
            raw = query_model(row["description"], model, tokenizer, generate_fn)
            pred = parse_json_response(raw)
        except Exception as e:
            print(f"\n[warn] {row['cve_id']}: {e}")

        results.append(_make_result(row, pred, raw, model_tag))

    out_path = RESULTS_DIR / f"results_lora_{model_tag}.jsonl"
    save_results(results, out_path)

    summary = summarize(model_tag, results)
    print_summary(summary)

    slug = model_tag.replace("/", "_")
    with open(RESULTS_DIR / f"summary_lora_{slug}.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "per_component_acc"}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="microsoft/Phi-3.5-mini-instruct",
                        help="base model on HuggingFace Hub")
    parser.add_argument("--adapter", default="adapters/phi35",
                        help="path to trained LoRA adapter directory")
    parser.add_argument("--n",       type=int, default=200,
                        help="number of test samples (0 = all)")
    args = parser.parse_args()
    run(args.model, args.adapter, args.n)
