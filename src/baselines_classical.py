"""
Classical ML baselines: TF-IDF + Logistic Regression and Random Forest.

Trains three tasks jointly:
  1. Severity classification (4-class)
  2. CWE classification (multi-class, ~150 categories)
  3. CVSS score regression (Ridge)

Also trains per-component CVSS vector classifiers.

Usage:
  python src/baselines_classical.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from prepare_splits import make_splits
from evaluate import (
    cvss_mae, exact_match, per_component_accuracy,
    summarize, print_summary, save_results, CVSS_VECTOR_FIELDS,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── TF-IDF config ────────────────────────────────────────────────────────────
TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    max_features=50_000,
    sublinear_tf=True,
    min_df=2,
)


def build_tfidf(train_texts: list[str]):
    vec = TfidfVectorizer(**TFIDF_KWARGS)
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def row_to_components(row) -> dict:
    return {f: row[f] for f in CVSS_VECTOR_FIELDS}


def run(model_type: str = "lr") -> None:
    assert model_type in ("lr", "rf"), "model_type must be 'lr' or 'rf'"
    label = "LogisticRegression" if model_type == "lr" else "RandomForest"
    print(f"\n{'='*60}")
    print(f"  Classical baseline: TF-IDF + {label}")
    print(f"{'='*60}")

    splits = make_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    # ── TF-IDF features ──────────────────────────────────────────────────────
    print("Building TF-IDF features ...")
    vec, X_train = build_tfidf(train["description"].tolist())
    X_val  = vec.transform(val["description"].tolist())
    X_test = vec.transform(test["description"].tolist())

    # ── Severity ─────────────────────────────────────────────────────────────
    print("Training severity classifier ...")
    if model_type == "lr":
        sev_clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    else:
        sev_clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    sev_clf.fit(X_train, train["severity"])
    sev_pred = sev_clf.predict(X_test)
    print(classification_report(test["severity"], sev_pred, zero_division=0))

    # ── CWE ──────────────────────────────────────────────────────────────────
    print("Training CWE classifier ...")
    top_cwes = train["cwe"].value_counts().head(100).index.tolist()
    train_cwe = train["cwe"].where(train["cwe"].isin(top_cwes), other="OTHER")
    test_cwe  = test["cwe"].where(test["cwe"].isin(top_cwes), other="OTHER")

    if model_type == "lr":
        cwe_clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    else:
        cwe_clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    cwe_clf.fit(X_train, train_cwe)
    cwe_pred = cwe_clf.predict(X_test)
    cwe_acc = exact_match(test_cwe.tolist(), cwe_pred.tolist())
    print(f"CWE top-1 accuracy (top-100 classes): {cwe_acc:.3f}")

    # ── CVSS score ───────────────────────────────────────────────────────────
    print("Training CVSS score regressor ...")
    score_reg = Ridge(alpha=1.0)
    score_reg.fit(X_train, train["cvss_score"])
    score_pred = score_reg.predict(X_test)
    score_pred_clipped = np.clip(score_pred, 0, 10)
    mae = cvss_mae(test["cvss_score"].tolist(), score_pred_clipped.tolist())
    print(f"CVSS score MAE: {mae:.3f}")

    # ── Per-component CVSS vector classifiers ────────────────────────────────
    print("Training per-component CVSS vector classifiers ...")
    comp_preds = {f: [] for f in CVSS_VECTOR_FIELDS}
    for field in CVSS_VECTOR_FIELDS:
        if model_type == "lr":
            clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X_train, train[field])
        comp_preds[field] = clf.predict(X_test).tolist()

    comp_true = [row_to_components(row) for _, row in test.iterrows()]
    comp_pred = [{f: comp_preds[f][i] for f in CVSS_VECTOR_FIELDS} for i in range(len(test))]
    comp_acc = per_component_accuracy(comp_true, comp_pred)
    print("Per-component accuracy:")
    for k, v in comp_acc.items():
        print(f"  {k:<25s}: {v:.3f}")

    # ── Assemble per-example results ─────────────────────────────────────────
    results = []
    for i, (_, row) in enumerate(test.iterrows()):
        pred_vec_parts = {f: comp_preds[f][i] for f in CVSS_VECTOR_FIELDS}
        results.append({
            "model": f"tfidf_{model_type}",
            "cve_id": row["cve_id"],
            "parsed": True,
            "true_severity":    row["severity"],
            "pred_severity":    sev_pred[i],
            "true_cwe":         row["cwe"],
            "pred_cwe":         cwe_pred[i],
            "true_cvss_score":  row["cvss_score"],
            "pred_cvss_score":  float(score_pred_clipped[i]),
            "true_cvss_vector": row["cvss_vector"],
            "pred_cvss_vector": "",   # classical models don't reconstruct full string
            "true_components":  row_to_components(row),
            "pred_components":  pred_vec_parts,
        })

    out_path = RESULTS_DIR / f"results_tfidf_{model_type}.jsonl"
    save_results(results, out_path)

    summary = summarize(f"TF-IDF + {label}", results)
    print_summary(summary)

    # Save summary
    with open(RESULTS_DIR / f"summary_tfidf_{model_type}.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "per_component_acc"}, f, indent=2)


if __name__ == "__main__":
    run("lr")
    run("rf")
