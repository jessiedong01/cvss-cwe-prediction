"""
Load all result JSONL files from results/ and produce a comparison table.

Usage:
  python src/compare_results.py
  python src/compare_results.py --latex     # also write results/comparison.tex
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import summarize, print_summary, CVSS_VECTOR_FIELDS, SEVERITY_CLASSES

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results_file(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def build_table() -> pd.DataFrame:
    result_files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
    if not result_files:
        print("No results files found. Run the baseline scripts first.")
        return pd.DataFrame()

    rows = []
    for path in result_files:
        results = load_results_file(path)
        if not results:
            continue
        model_name = results[0].get("model", path.stem.replace("results_", ""))
        summary = summarize(model_name, results)

        comp_acc = summary.get("per_component_acc", {})
        row = {
            "Model":              summary["model"],
            "Parse rate":         f"{summary.get('parse_rate', 0):.1%}",
            "CVSS MAE":           f"{summary.get('cvss_mae', float('nan')):.3f}",
            "Severity acc":       f"{summary.get('severity_acc', float('nan')):.3f}",
            "CWE acc":            f"{summary.get('cwe_acc', float('nan')):.3f}",
            "Vector acc":         f"{summary.get('cvss_vector_acc', float('nan')):.3f}",
            "Vector partial":     f"{summary.get('cvss_vector_partial_match', float('nan')):.3f}",
            "AV acc":             f"{comp_acc.get('attack_vector', float('nan')):.3f}",
            "AC acc":             f"{comp_acc.get('attack_complexity', float('nan')):.3f}",
            "PR acc":             f"{comp_acc.get('privileges_required', float('nan')):.3f}",
            "UI acc":             f"{comp_acc.get('user_interaction', float('nan')):.3f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def breakdown_by_severity(results: list[dict]) -> None:
    """Print per-severity-tier accuracy and CVSS MAE."""
    from collections import defaultdict
    valid = [r for r in results if r.get("parsed")]
    if not valid:
        return

    tier_records: dict[str, list] = defaultdict(list)
    for r in valid:
        tier_records[r["true_severity"]].append(r)

    print(f"\n{'Severity tier':<12} {'Count':>6}  {'Sev acc':>8}  {'CVSS MAE':>9}  {'CWE acc':>8}")
    print("-" * 52)
    for tier in SEVERITY_CLASSES:
        recs = tier_records.get(tier, [])
        if not recs:
            continue
        sev_acc = sum(r["true_severity"] == r["pred_severity"] for r in recs) / len(recs)
        mae = float(np.mean([abs(r["true_cvss_score"] - r["pred_cvss_score"]) for r in recs]))
        cwe_acc = sum(r["true_cwe"] == r["pred_cwe"] for r in recs) / len(recs)
        print(f"  {tier:<10} {len(recs):>6}  {sev_acc:>8.1%}  {mae:>9.3f}  {cwe_acc:>8.1%}")


def breakdown_by_cwe(results: list[dict], top_n: int = 20) -> None:
    """Print per-CWE-category accuracy for the most-common categories."""
    from collections import Counter, defaultdict
    valid = [r for r in results if r.get("parsed")]
    if not valid:
        return

    top_cwes = [cwe for cwe, _ in Counter(r["true_cwe"] for r in valid).most_common(top_n)]
    cwe_records: dict[str, list] = defaultdict(list)
    for r in valid:
        if r["true_cwe"] in top_cwes:
            cwe_records[r["true_cwe"]].append(r)

    print(f"\n{'CWE':<12} {'Count':>6}  {'Sev acc':>8}  {'CWE acc':>8}  {'CVSS MAE':>9}")
    print("-" * 52)
    for cwe in top_cwes:
        recs = cwe_records[cwe]
        sev_acc = sum(r["true_severity"] == r["pred_severity"] for r in recs) / len(recs)
        cwe_acc = sum(r["true_cwe"] == r["pred_cwe"] for r in recs) / len(recs)
        mae = float(np.mean([abs(r["true_cvss_score"] - r["pred_cvss_score"]) for r in recs]))
        print(f"  {cwe:<10} {len(recs):>6}  {sev_acc:>8.1%}  {cwe_acc:>8.1%}  {mae:>9.3f}")


def failure_analysis(path: Path, n: int = 20) -> None:
    """Print failure modes, per-tier breakdown, and per-CWE breakdown for a results file."""
    from collections import Counter
    results = load_results_file(path)
    model = results[0].get("model", path.stem)
    print(f"\n=== Failure analysis: {model} ===")

    sev_errors = [(r["true_severity"], r["pred_severity"])
                  for r in results if r.get("parsed") and r["true_severity"] != r["pred_severity"]]
    cwe_errors  = [(r["true_cwe"], r["pred_cwe"])
                   for r in results if r.get("parsed") and r["true_cwe"] != r["pred_cwe"]]

    print(f"\nSeverity confusion (true -> pred), top {n}:")
    for (t, p), cnt in Counter(sev_errors).most_common(n):
        print(f"  {t} -> {p}: {cnt}")

    print(f"\nMost mis-predicted CWEs (true), top {n}:")
    for cwe, cnt in Counter([t for t, p in cwe_errors]).most_common(n):
        print(f"  {cwe}: {cnt} errors")

    print("\n--- Accuracy breakdown by severity tier ---")
    breakdown_by_severity(results)

    print("\n--- Accuracy breakdown by CWE category (top 20) ---")
    breakdown_by_cwe(results, top_n=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--failures", type=str, default="",
                        help="path to a specific results JSONL for failure analysis")
    parser.add_argument("--breakdown", action="store_true",
                        help="print per-tier and per-CWE breakdown for all results files")
    args = parser.parse_args()

    df = build_table()
    if df.empty:
        sys.exit(0)

    try:
        from tabulate import tabulate
        print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df.to_string(index=False))

    if args.latex:
        tex = df.to_latex(index=False, caption="CVE triage model comparison", label="tab:results")
        tex_path = RESULTS_DIR / "comparison.tex"
        tex_path.write_text(tex)
        print(f"\nLaTeX table -> {tex_path}")

    csv_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV -> {csv_path}")

    if args.failures:
        failure_analysis(Path(args.failures))
    elif args.breakdown:
        for f in sorted(RESULTS_DIR.glob("results_*.jsonl")):
            failure_analysis(f)
    else:
        # Auto-run failure analysis on first available results file
        files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
        if files:
            failure_analysis(files[0])
