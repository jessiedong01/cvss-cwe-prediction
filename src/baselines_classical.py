"""
Classical baselines over TF-IDF features.

Models (hyperparameters chosen on the 2022-2023 validation split):
  lr       logistic regression
  lr_bal   logistic regression with class-balanced weights
  rf       random forest, 300 trees (not tuned; reported as-is)

Each model produces two systems:
  *_direct   severity from a severity classifier, score from ridge regression
  *_formula  predict the 8 CVSS components, then compute score and severity
             with the CVSS v3.1 formula (src/cvss.py)

Outputs:
  results/preds_<system>.jsonl.gz   per-example predictions on the test set
  results/summary_classical.json    metrics with 95% bootstrap CIs
  results/tuning_classical.json     validation scores for every setting tried

Usage:
  python src/baselines_classical.py
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).parent))
from cvss import base_score, severity
from metrics import COMPONENTS, summarize
from prepare_splits import make_splits

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TFIDF_KWARGS = dict(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2)
C_GRID = [0.3, 1.0, 3.0, 10.0]
ALPHA_GRID = [0.3, 1.0, 3.0, 10.0]
TOP_CWES = 100


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr(C: float, balanced: bool = False) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None)


def rf(n_trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=42)


def accuracy(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def tune_C(X_tr, y_tr, X_va, y_va, balanced: bool, name: str, tuning: dict) -> float:
    scores = {}
    for C in C_GRID:
        scores[C] = accuracy(y_va, lr(C, balanced).fit(X_tr, y_tr).predict(X_va))
        log(f"  {name} C={C}: val acc {scores[C]:.4f}")
    tuning[name] = scores
    return max(scores, key=scores.get)


def write_preds(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    splits = make_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    log(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    vec = TfidfVectorizer(**TFIDF_KWARGS)
    X_tr = vec.fit_transform(train["description"])
    X_va = vec.transform(val["description"])
    X_te = vec.transform(test["description"])

    top = set(train["cwe"].value_counts().head(TOP_CWES).index)
    cwe_tr = train["cwe"].where(train["cwe"].isin(top), "OTHER")
    cwe_va = val["cwe"].where(val["cwe"].isin(top), "OTHER")
    coverage = float(test["cwe"].isin(top).mean())
    log(f"top-{TOP_CWES} CWEs cover {coverage:.1%} of test CVEs")

    tuning: dict = {}

    # ── CVSS score by ridge regression (shared by every *_direct system) ──
    maes = {}
    for a in ALPHA_GRID:
        p = np.clip(Ridge(alpha=a).fit(X_tr, train["cvss_score"]).predict(X_va), 0, 10)
        maes[a] = float(np.mean(np.abs(p - val["cvss_score"])))
        log(f"  ridge alpha={a}: val MAE {maes[a]:.4f}")
    tuning["ridge_alpha"] = maes
    alpha = min(maes, key=maes.get)
    ridge_pred = np.clip(Ridge(alpha=alpha).fit(X_tr, train["cvss_score"]).predict(X_te), 0, 10)

    # ── Fit the three models ──
    preds: dict[str, dict] = {}
    chosen: dict = {"ridge_alpha": alpha, "top_cwes": TOP_CWES, "cwe_coverage_test": coverage}

    for name, balanced in (("lr", False), ("lr_bal", True)):
        log(f"tuning {name}")
        C_sev = tune_C(X_tr, train["severity"], X_va, val["severity"], balanced, f"{name}_severity", tuning)
        C_cwe = tune_C(X_tr, cwe_tr, X_va, cwe_va, False, f"{name}_cwe", tuning) if name == "lr" else chosen["lr"]["C_cwe"]
        # one C for all components, chosen by mean validation accuracy
        comp_scores = {}
        for C in C_GRID:
            comp_scores[C] = float(np.mean([
                accuracy(val[f], lr(C, balanced).fit(X_tr, train[f]).predict(X_va)) for f in COMPONENTS
            ]))
            log(f"  {name}_components C={C}: mean val acc {comp_scores[C]:.4f}")
        tuning[f"{name}_components"] = comp_scores
        C_comp = max(comp_scores, key=comp_scores.get)
        chosen[name] = {"C_severity": C_sev, "C_cwe": C_cwe, "C_components": C_comp, "balanced": balanced}

        log(f"fitting {name} on train: {chosen[name]}")
        preds[name] = {
            "severity": lr(C_sev, balanced).fit(X_tr, train["severity"]).predict(X_te),
            "cwe": lr(C_cwe).fit(X_tr, cwe_tr).predict(X_te),
            "components": {f: lr(C_comp, balanced).fit(X_tr, train[f]).predict(X_te) for f in COMPONENTS},
        }

    log("fitting rf (300 trees; components 100 trees)")
    chosen["rf"] = {"trees": 300, "trees_components": 100}
    preds["rf"] = {
        "severity": rf(300).fit(X_tr, train["severity"]).predict(X_te),
        "cwe": rf(300).fit(X_tr, cwe_tr).predict(X_te),
        "components": {f: rf(100).fit(X_tr, train[f]).predict(X_te) for f in COMPONENTS},
    }

    # ── Assemble systems ──
    summary = {"settings": chosen, "systems": {}}
    rows = test.to_dict("records")
    for name, p in preds.items():
        for mode in ("direct", "formula"):
            system = f"{name}_{mode}"
            records = []
            for i, row in enumerate(rows):
                comps = {f: p["components"][f][i] for f in COMPONENTS}
                if mode == "formula":
                    score = base_score(comps)
                    sev = severity(score)
                else:
                    score = float(ridge_pred[i])
                    sev = p["severity"][i]
                records.append({
                    "system": system, "cve_id": row["cve_id"], "parsed": True,
                    "true_severity": row["severity"], "pred_severity": sev,
                    "true_cwe": row["cwe"], "pred_cwe": p["cwe"][i],
                    "true_cvss_score": row["cvss_score"], "pred_cvss_score": score,
                    "true_components": {f: row[f] for f in COMPONENTS},
                    "pred_components": comps,
                })
            write_preds(RESULTS_DIR / f"preds_{system}.jsonl.gz", records)
            s = summarize(records)
            summary["systems"][system] = s
            e = s["est"]
            log(f"{system:14s} sev {e['sev']:.3f}  macroF1 {e['sev_macro_f1']:.3f}  "
                f"cwe {e['cwe']:.3f}  MAE {e['ae']:.3f}  partial {e['partial']:.3f}  vector {e['vector']:.3f}")

    (RESULTS_DIR / "summary_classical.json").write_text(json.dumps(summary, indent=2, default=float))
    (RESULTS_DIR / "tuning_classical.json").write_text(json.dumps(tuning, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main()
