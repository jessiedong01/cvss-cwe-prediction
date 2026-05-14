"""
Shared evaluation utilities used by all model scripts.
"""
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = Path(__file__).parent.parent / "results"

CVSS_VECTOR_FIELDS = [
    "attack_vector", "attack_complexity", "privileges_required",
    "user_interaction", "scope", "confidentiality", "integrity", "availability",
]

SEVERITY_CLASSES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def cvss_mae(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def exact_match(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([t.upper() == p.upper() for t, p in zip(y_true, y_pred)]))


def per_component_accuracy(true_rows: list[dict], pred_rows: list[dict]) -> dict[str, float]:
    results = {}
    for field in CVSS_VECTOR_FIELDS:
        t = [r.get(field, "").upper() for r in true_rows]
        p = [r.get(field, "").upper() for r in pred_rows]
        results[field] = exact_match(t, p)
    return results


def top_k_accuracy(y_true: list[str], y_pred_topk: list[list[str]], k: int = 3) -> float:
    hits = sum(t in [p.upper() for p in preds[:k]] for t, preds in zip(y_true, y_pred_topk))
    return hits / len(y_true)


def parse_json_response(text: str) -> dict:
    """Extract first JSON object from a model response string."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Extract first { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def summarize(model_name: str, results: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of per-example result dicts."""
    valid = [r for r in results if r.get("parsed")]
    n_total = len(results)
    n_valid = len(valid)

    summary = {
        "model": model_name,
        "n_total": n_total,
        "n_parsed": n_valid,
        "parse_rate": n_valid / n_total if n_total else 0,
    }

    if not valid:
        return summary

    summary["cvss_mae"] = cvss_mae(
        [r["true_cvss_score"] for r in valid],
        [r["pred_cvss_score"] for r in valid],
    )
    summary["severity_acc"] = exact_match(
        [r["true_severity"] for r in valid],
        [r["pred_severity"] for r in valid],
    )
    summary["cwe_acc"] = exact_match(
        [r["true_cwe"] for r in valid],
        [r["pred_cwe"] for r in valid],
    )
    summary["cvss_vector_acc"] = exact_match(
        [r["true_cvss_vector"] for r in valid],
        [r["pred_cvss_vector"] for r in valid],
    )
    summary["per_component_acc"] = per_component_accuracy(
        [r["true_components"] for r in valid],
        [r["pred_components"] for r in valid],
    )

    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['model']} ===")
    print(f"  Parse rate       : {summary.get('parse_rate', 0):.1%} ({summary.get('n_parsed')}/{summary.get('n_total')})")
    print(f"  CVSS MAE         : {summary.get('cvss_mae', float('nan')):.3f}")
    print(f"  Severity acc     : {summary.get('severity_acc', float('nan')):.3f}")
    print(f"  CWE acc (top-1)  : {summary.get('cwe_acc', float('nan')):.3f}")
    print(f"  CVSS vector acc  : {summary.get('cvss_vector_acc', float('nan')):.3f}")
    if "per_component_acc" in summary:
        print("  Per-component accuracy:")
        for k, v in summary["per_component_acc"].items():
            print(f"    {k:<25s}: {v:.3f}")


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results -> {path}")
