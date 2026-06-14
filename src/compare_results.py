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

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import summarize, print_summary, CVSS_VECTOR_FIELDS

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
            "Model":            summary["model"],
            "Parse rate":       f"{summary.get('parse_rate', 0):.1%}",
            "CVSS MAE":         f"{summary.get('cvss_mae', float('nan')):.3f}",
            "Severity acc":     f"{summary.get('severity_acc', float('nan')):.3f}",
            "CWE acc":          f"{summary.get('cwe_acc', float('nan')):.3f}",
            "Vector acc":       f"{summary.get('cvss_vector_acc', float('nan')):.3f}",
            "AV acc":           f"{comp_acc.get('attack_vector', float('nan')):.3f}",
            "AC acc":           f"{comp_acc.get('attack_complexity', float('nan')):.3f}",
            "PR acc":           f"{comp_acc.get('privileges_required', float('nan')):.3f}",
            "UI acc":           f"{comp_acc.get('user_interaction', float('nan')):.3f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def failure_analysis(path: Path, n: int = 20) -> None:
    """Print the most common failure modes for a given results file."""
    results = load_results_file(path)
    model = results[0].get("model", path.stem)
    print(f"\n=== Failure analysis: {model} ===")

    sev_errors = [(r["true_severity"], r["pred_severity"])
                  for r in results if r.get("parsed") and r["true_severity"] != r["pred_severity"]]
    cwe_errors  = [(r["true_cwe"], r["pred_cwe"])
                   for r in results if r.get("parsed") and r["true_cwe"] != r["pred_cwe"]]

    print(f"\nSeverity confusion (true -> pred), top {n}:")
    from collections import Counter
    for (t, p), cnt in Counter(sev_errors).most_common(n):
        print(f"  {t} -> {p}: {cnt}")

    print(f"\nMost mis-predicted CWEs (true), top {n}:")
    for cwe, cnt in Counter([t for t, p in cwe_errors]).most_common(n):
        print(f"  {cwe}: {cnt} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--failures", type=str, default="",
                        help="path to a specific results JSONL for failure analysis")
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
    else:
        # Auto-run failure analysis on first available results file
        files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
        if files:
            failure_analysis(files[0])
