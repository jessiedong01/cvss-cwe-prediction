"""
Metrics with bootstrap 95% confidence intervals.

Every metric is computed per example, so one bootstrap over example indices
gives intervals for all of them at once.
"""
import numpy as np

from cvss import SEVERITIES

COMPONENTS = [
    "attack_vector", "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality", "integrity", "availability",
]


def _norm(x) -> str:
    x = "" if x is None else str(x).strip().upper()
    return "ADJACENT_NETWORK" if x == "ADJACENT" else x


def per_example(r: dict) -> dict:
    """Per-example metric values for one prediction record."""
    tc, pc = r["true_components"], r.get("pred_components") or {}
    comp_hits = [_norm(tc[f]) == _norm(pc.get(f)) for f in COMPONENTS]
    score = r.get("pred_cvss_score")
    return {
        "sev": float(_norm(r["true_severity"]) == _norm(r.get("pred_severity"))),
        "cwe": float(_norm(r["true_cwe"]) == _norm(r.get("pred_cwe"))),
        "ae": abs(r["true_cvss_score"] - score) if score is not None else np.nan,
        "partial": float(np.mean(comp_hits)),
        "vector": float(all(comp_hits)),
        **{f"c_{f}": float(h) for f, h in zip(COMPONENTS, comp_hits)},
    }


def macro_f1(y_true, y_pred, labels=SEVERITIES) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in labels:
        t, p = y_true == c, y_pred == c
        tp = np.sum(t & p)
        denom = t.sum() + p.sum()
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def summarize(records: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates and 95% bootstrap CIs. Unparsed records count as wrong."""
    rows = [per_example(r) for r in records]
    keys = rows[0].keys()
    M = {k: np.array([row[k] for row in rows], dtype=float) for k in keys}
    yt = np.array([_norm(r["true_severity"]) for r in records])
    yp = np.array([_norm(r.get("pred_severity")) for r in records])
    n = len(records)

    def point(idx):
        out = {k: float(np.nanmean(v[idx])) for k, v in M.items()}
        out["sev_macro_f1"] = macro_f1(yt[idx], yp[idx])
        return out

    est = point(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [point(rng.integers(0, n, n)) for _ in range(n_boot)]
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)),
              float(np.percentile([b[k] for b in boots], 97.5))) for k in est}
    parsed = sum(bool(r.get("parsed", True)) for r in records)
    return {"n": n, "parse_rate": parsed / n, "est": est, "ci": ci}
