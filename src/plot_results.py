"""
Paper figures, all generated from real result files in results/.

  fig_confusion.pdf        severity confusion matrices, TF-IDF vs fine-tuned Phi-3.5
  fig_components.pdf       per-component accuracy, three systems
  fig_score_dist.pdf       predicted vs. true CVSS score distributions
  fig_cwe_frequency.pdf    CWE accuracy vs. how often the CWE appears in training
  fig_val_curve.pdf        validation metrics across LoRA checkpoints

Any figure whose inputs are missing is skipped with a message.

Usage:
  python src/plot_results.py
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cvss import SEVERITIES
from metrics import COMPONENTS, _norm

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
# Colour follows the system everywhere in the paper.
SYSTEMS = {
    "tfidf": ("TF-IDF + LR, formula", "#2a78d6", RESULTS / "preds_lr_formula.jsonl.gz"),
    "lora": ("Phi-3.5 + LoRA", "#eb6834", RESULTS / "preds_phi35_lora_formula_test.jsonl.gz"),
    "zeroshot": ("Phi-3.5 zero-shot", "#1baf7a", RESULTS / "preds_phi35_zeroshot_formula_test.jsonl.gz"),
}
TRUTH = "#8a8985"      # neutral gray for NVD ground truth
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

COMPONENT_LABELS = {
    "attack_vector": "Attack vector", "attack_complexity": "Attack complexity",
    "privileges_required": "Privileges required", "user_interaction": "User interaction",
    "scope": "Scope", "confidentiality": "Confidentiality", "integrity": "Integrity",
    "availability": "Availability",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, FULL = 3.3, 6.9  # inches: one column, full width


def load(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {name}")


def available() -> dict[str, list[dict]]:
    out = {}
    for key, (_, _, path) in SYSTEMS.items():
        recs = load(path)
        if recs is None:
            print(f"  (missing {path.name}; {key} left out)")
        else:
            out[key] = recs
    return out


# ── 0. Headline: fine-tuned model vs. best TF-IDF on the full test set ───────

def fig_headline(data: dict) -> None:
    from metrics import summarize
    keys = [k for k in ("tfidf", "lora") if k in data]
    if len(keys) < 2:
        return
    metrics = [("sev", "Severity\naccuracy"), ("sev_macro_f1", "Severity\nmacro-F1"),
               ("cwe", "CWE\naccuracy"), ("vector", "Exact CVSS\nvector")]
    summ = {k: summarize(data[k]) for k in keys}
    fig, ax = plt.subplots(figsize=(3.7, 2.15))
    x = np.arange(len(metrics))
    w = 0.36
    for j, k in enumerate(keys):
        est = np.array([summ[k]["est"][m] for m, _ in metrics])
        lo = est - np.array([summ[k]["ci"][m][0] for m, _ in metrics])
        hi = np.array([summ[k]["ci"][m][1] for m, _ in metrics]) - est
        xs = x + (j - 0.5) * (w + 0.03)
        ax.bar(xs, est, w, color=SYSTEMS[k][1], label=SYSTEMS[k][0], zorder=3)
        ax.errorbar(xs, est, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        for xi, v in zip(xs, est):
            ax.text(xi, v + 0.02, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold" if k == "lora" else "normal", color=INK)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.5, borderaxespad=0.2)
    save(fig, "fig_headline")


# ── 1. Confusion matrices ─────────────────────────────────────────────────────

def confusion(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Counts by (true, predicted) tier, and the number of CVEs in each true tier.

    Row totals include predictions outside the four tiers (a formula score of 0 has
    severity NONE), so each row's shares match the recall reported in the tables."""
    idx = {s: i for i, s in enumerate(SEVERITIES)}
    cm, totals = np.zeros((4, 4)), np.zeros(4)
    for r in records:
        t, p = _norm(r["true_severity"]), _norm(r["pred_severity"])
        if t in idx:
            totals[idx[t]] += 1
            if p in idx:
                cm[idx[t], idx[p]] += 1
    return cm, totals


def fig_confusion(data: dict) -> None:
    keys = [k for k in ("tfidf", "lora") if k in data]
    fig, axes = plt.subplots(1, len(keys), figsize=(COL * len(keys) * 0.95, 2.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        cm, totals = confusion(data[key])
        frac = cm / totals[:, None]
        ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{frac[i, j]:.0%}\n{int(cm[i, j]):,}", ha="center", va="center",
                        fontsize=6.5, color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), [s.title() for s in SEVERITIES])
        ax.set_yticks(range(4), [s.title() for s in SEVERITIES])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("NVD")
        ax.set_title(SYSTEMS[key][0])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_confusion")


# ── 2. Per-component accuracy (dot plot) ─────────────────────────────────────

def fig_components(data: dict) -> None:
    keys = [k for k in SYSTEMS if k in data]
    acc = {k: {f: np.mean([_norm(r["true_components"][f]) == _norm(r["pred_components"].get(f))
                           for r in data[k]]) for f in COMPONENTS} for k in keys}
    order = sorted(COMPONENTS, key=lambda f: acc[keys[0]][f])  # ascending for the first system
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    y = np.arange(len(order))
    offsets = np.linspace(-0.18, 0.18, len(keys)) if len(keys) > 1 else [0.0]
    for k, dy in zip(keys, offsets):  # small vertical dodge so equal values stay visible
        xs = [acc[k][f] for f in order]
        ax.scatter(xs, y + dy, s=22, color=SYSTEMS[k][1], label=SYSTEMS[k][0], zorder=3,
                   edgecolors="white", linewidths=0.8)
    for i, f in enumerate(order):  # thin guide from worst to best system
        vals = [acc[k][f] for k in keys]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=1.5, zorder=1)
    ax.set_yticks(y, [COMPONENT_LABELS[f] for f in order])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Accuracy on 2024 CVEs")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2)
    save(fig, "fig_components")


# ── 3. Score distributions (small multiples, one axis each) ──────────────────

def fig_score_dist(data: dict) -> None:
    panels = []
    ridge = load(RESULTS / "preds_lr_direct.jsonl.gz")
    if ridge:
        panels.append(("TF-IDF ridge regression", SYSTEMS["tfidf"][1], ridge))
    if "lora" in data:
        direct = load(RESULTS / "preds_phi35_lora_direct_test.jsonl.gz")
        if direct:
            panels.append(("Phi-3.5 + LoRA, score as written", SYSTEMS["lora"][1], direct))
    if not panels:
        return
    bins = np.arange(0, 10.25, 0.25)
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL * len(panels) / 3, 1.9), sharey=True, squeeze=False)
    for ax, (title, color, recs) in zip(axes[0], panels):
        true = [r["true_cvss_score"] for r in recs]
        pred = [r["pred_cvss_score"] for r in recs if r["pred_cvss_score"] is not None]
        w_t, w_p = np.ones(len(true)) / len(true), np.ones(len(pred)) / len(pred)
        ax.hist(true, bins=bins, weights=w_t, color=TRUTH, alpha=0.35, label="NVD")
        ax.hist(pred, bins=bins, weights=w_p, histtype="step", color=color, lw=1.4, label="Predicted")
        ax.set_title(title)
        ax.set_xlabel("CVSS base score")
        ax.set_xlim(0, 10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, loc="upper left")
    axes[0][0].set_ylabel("Share of CVEs")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_score_dist")


# ── 4. CWE accuracy vs. training frequency ───────────────────────────────────

def fig_cwe_frequency(data: dict) -> None:
    import pandas as pd
    train = pd.read_pickle(ROOT / "data" / "splits.pkl")["train"]
    freq = train["cwe"].value_counts()
    keys = [k for k in ("tfidf", "lora") if k in data]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for k in keys:
        hits, n = defaultdict(int), Counter()
        for r in data[k]:
            n[r["true_cwe"]] += 1
            hits[r["true_cwe"]] += _norm(r["true_cwe"]) == _norm(r["pred_cwe"])
        cwes = [c for c in n if n[c] >= 20]  # stable per-class accuracy
        x = [max(freq.get(c, 0), 0.8) for c in cwes]
        y = [hits[c] / n[c] for c in cwes]
        sizes = [6 + 2.5 * np.sqrt(n[c]) / 3 for c in cwes]
        ax.scatter(x, y, s=sizes, color=SYSTEMS[k][1], alpha=0.75, label=SYSTEMS[k][0],
                   edgecolors="white", linewidths=0.5)
    top100 = freq.iloc[99] if len(freq) >= 100 else None
    if top100 is not None and "tfidf" in keys:
        ax.axvline(top100, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.text(top100 * 0.85, 0.99, "TF-IDF predicts\nonly the top 100", ha="right", va="top",
                fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Training examples with this CWE (log scale)")
    ax.set_ylabel("CWE accuracy on 2024")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.03, 1.05)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [matplotlib.lines.Line2D([], [], ls="", marker="o", ms=5, color=SYSTEMS[k][1],
                                       label=SYSTEMS[k][0]) for k in keys]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(handles), handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2)
    save(fig, "fig_cwe_frequency")


# ── 5. Validation curve across checkpoints (two panels, one axis each) ──────

def fig_val_curve() -> None:
    pts = []
    for p in sorted(RESULTS.glob("summary_ckpt*_val.json")):
        s = json.loads(p.read_text())
        it = int(s["tag"].replace("ckpt", ""))
        sysd = s["systems"][f"{s['tag']}_formula"]
        pts.append((it, sysd["est"]["sev"], sysd["est"]["sev_macro_f1"], sysd["est"]["ae"]))
    if not pts:
        print("  (no checkpoint summaries; skipping val curve)")
        return
    pts.sort()
    it, acc, f1, mae = map(np.array, zip(*pts))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
    for ax, y, label in ((axes[0], acc, "Severity accuracy"), (axes[1], mae, "CVSS score MAE")):
        ax.plot(it, y, color=SYSTEMS["lora"][1], lw=2, marker="o", ms=4)
        ax.set_title(label)
        ax.set_xlabel("Training step")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_val_curve")


if __name__ == "__main__":
    data = available()
    fig_headline(data)
    fig_confusion(data)
    fig_components(data)
    fig_score_dist(data)
    fig_cwe_frequency(data)
    fig_val_curve()
