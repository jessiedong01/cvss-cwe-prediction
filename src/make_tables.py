"""
Write every number the paper quotes, straight from data and result files.

  paper/numbers.tex          \\newcommand macros used in the text
  paper/tables/data.tex      label distribution per split
  paper/tables/main.tex      all systems on the full 2024 test set
  paper/tables/subset.tex    zero-shot comparison on a shared random subset
  paper/tables/tiers.tex     per-severity recall and F1
  paper/tables/consistency.tex   language-model self-consistency
  paper/tables/prompts.tex   the exact prompts, copied from the code

Nothing in the paper's results is typed by hand. Missing inputs become "--".

Usage:
  python src/make_tables.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cvss import SEVERITIES
from evaluate_lora import ZERO_SHOT_TEMPLATE
from metrics import _norm, summarize
from prepare_lora_data import USER_TEMPLATE

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
MISSING = "--"

macros: dict[str, str] = {}


def macro(name: str, value) -> None:
    assert name.isalpha(), f"LaTeX macro names must be letters only: {name}"
    macros[name] = str(value)


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pct(x: float | None, d: int = 1) -> str:
    return MISSING if x is None else f"{100 * x:.{d}f}"


def load_preds(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ── Dataset ──────────────────────────────────────────────────────────────────

def data_section() -> None:
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    allrec = pd.read_pickle(ROOT / "data" / "cve_parsed.pkl")
    tr, va, te = splits["train"], splits["val"], splits["test"]
    macro("nTrain", fmt_int(len(tr)))
    macro("nVal", fmt_int(len(va)))
    macro("nTrainVal", fmt_int(len(tr) + len(va)))
    macro("nTest", fmt_int(len(te)))
    macro("nAll", fmt_int(len(allrec)))
    macro("nVThreeZero", fmt_int(int((~allrec["cvss_vector"].str.startswith("CVSS:3.1")).sum())))
    macro("nTrainCwes", fmt_int(tr["cwe"].nunique()))
    macro("nTestCwes", fmt_int(te["cwe"].nunique()))
    top = set(tr["cwe"].value_counts().head(100).index)
    macro("nTopCov", pct(te["cwe"].isin(top).mean()))
    macro("nTestClipped", pct((te["description"].str.len() > 2000).mean()))

    rows = []
    for name, df in (("Train (2022--23)", tr), ("Validation (2022--23)", va), ("Test (2024)", te)):
        share = df["severity"].value_counts(normalize=True)
        rows.append(f"{name} & {fmt_int(len(df))} & " +
                    " & ".join(pct(share.get(s, 0.0)) for s in SEVERITIES) +
                    f" & {fmt_int(df['cwe'].nunique())} \\\\")
    (TABLES / "data.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Data splits, with severity columns giving the share of CVEs (\%) in each tier.}
\label{tab:data}
\small
\begin{tabular}{lrrrrrr}
\toprule
Split & CVEs & Critical & High & Medium & Low & CWEs \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ── Results ──────────────────────────────────────────────────────────────────

SYSTEMS = [  # (row label, prediction file stem, macro prefix)
    ("TF-IDF + LR, direct", "preds_lr_direct", "LRd"),
    ("TF-IDF + LR, formula", "preds_lr_formula", "LRf"),
    ("TF-IDF + LR balanced, direct", "preds_lr_bal_direct", "LRBd"),
    ("TF-IDF + LR balanced, formula", "preds_lr_bal_formula", "LRBf"),
    ("TF-IDF + RF, direct", "preds_rf_direct", "RFd"),
    ("TF-IDF + RF, formula", "preds_rf_formula", "RFf"),
    ("Phi-3.5 + LoRA, direct", "preds_phi35_lora_direct_test", "LoRAd"),
    ("Phi-3.5 + LoRA, formula", "preds_phi35_lora_formula_test", "LoRAf"),
]
METRICS = [("sev", "Sev"), ("sev_macro_f1", "Fone"), ("cwe", "Cwe"), ("ae", "Mae"),
           ("partial", "Part"), ("vector", "Vec")]


def cell(s: dict, key: str, best: bool) -> str:
    if s is None:
        return MISSING
    est = s["est"][key]
    lo, hi = s["ci"][key]
    if key == "ae":
        txt, hw = f"{est:.2f}", f"{(hi - lo) / 2:.2f}"
    else:
        txt, hw = f"{100 * est:.1f}", f"{100 * (hi - lo) / 2:.1f}"
    txt = rf"\textbf{{{txt}}}" if best else txt
    return rf"{txt}\,{{\scriptsize$\pm${hw}}}"


def results_table(rows: list[tuple[str, dict | None]], caption: str, label: str, path: Path) -> None:
    present = [s for _, s in rows if s is not None]
    shown = lambda s, k: round(s["est"][k], 2) if k == "ae" else round(100 * s["est"][k], 1)
    best = {}
    for key, _ in METRICS:
        vals = [shown(s, key) for s in present]
        best[key] = (min if key == "ae" else max)(vals) if vals else None
    body = []
    for name, s in rows:  # ties at the displayed precision are all bold
        cells = [cell(s, k, s is not None and shown(s, k) == best[k]) for k, _ in METRICS]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    path.write_text(r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Severity} & CWE & Score & \multicolumn{2}{c}{Vector} \\
\cmidrule(lr){2-3}\cmidrule(lr){6-7}
System & Acc. & Macro-F1 & Acc. & MAE $\downarrow$ & Partial & Exact \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def main_results() -> dict:
    out = {}
    rows = []
    for name, stem, prefix in SYSTEMS:
        recs = load_preds(RESULTS / f"{stem}.jsonl.gz")
        s = summarize(recs) if recs else None
        out[prefix] = (s, recs)
        rows.append((name, s))
        for key, suffix in METRICS:
            if s is None:
                macro(f"n{prefix}{suffix}", MISSING)
                continue
            est = s["est"][key]
            macro(f"n{prefix}{suffix}", f"{est:.2f}" if key == "ae" else pct(est))
    results_table(rows, r"All systems on the full 2024 test set (" + macros["nTest"] + r" CVEs). "
                  r"Values are percentages except score MAE. $\pm$ gives the half-width of the 95\% bootstrap interval. "
                  r"Bold marks the best value in each column. The TF-IDF direct systems share one ridge regression, "
                  r"so their score MAE is identical.",
                  "tab:main", TABLES / "main.tex")
    return out


def subset_results(full: dict) -> None:
    zs_direct = load_preds(RESULTS / "preds_phi35_zeroshot_direct_test.jsonl.gz")
    zs_formula = load_preds(RESULTS / "preds_phi35_zeroshot_formula_test.jsonl.gz")
    if not zs_direct:
        macro("nSub", MISSING)
        for p in ("ZSd", "ZSf", "LRfSub", "LoRAfSub"):
            for _, suffix in METRICS:
                macro(f"n{p}{suffix}", MISSING)
        macro("nZSParse", MISSING)
        (TABLES / "subset.tex").write_text("% zero-shot results not available yet\n")
        return
    ids = {r["cve_id"] for r in zs_direct}
    macro("nSub", fmt_int(len(ids)))
    macro("nZSParse", pct(np.mean([r["parsed"] for r in zs_direct])))
    n = len(zs_direct)
    macro("nZSCritShare", pct(sum(_norm(r["pred_severity"]) == "CRITICAL" for r in zs_direct) / n))
    macro("nSubCritShare", pct(sum(_norm(r["true_severity"]) == "CRITICAL" for r in zs_direct) / n))
    macro("nZSMediumCount", fmt_int(sum(_norm(r["pred_severity"]) == "MEDIUM" for r in zs_direct)))
    macro("nZSMediumShare", pct(sum(_norm(r["pred_severity"]) == "MEDIUM" for r in zs_direct) / n))
    macro("nSubMediumShare", pct(sum(_norm(r["true_severity"]) == "MEDIUM" for r in zs_direct) / n))

    def restrict(prefix):
        recs = full[prefix][1]
        return [r for r in recs if r["cve_id"] in ids] if recs else None

    rows = [
        ("TF-IDF + LR, formula", restrict("LRf"), "LRfSub"),
        ("Phi-3.5 zero-shot, direct", zs_direct, "ZSd"),
        ("Phi-3.5 zero-shot, formula", zs_formula, "ZSf"),
        ("Phi-3.5 + LoRA, formula", restrict("LoRAf"), "LoRAfSub"),
    ]
    table_rows = []
    for name, recs, prefix in rows:
        s = summarize(recs) if recs else None
        table_rows.append((name, s))
        for key, suffix in METRICS:
            macro(f"n{prefix}{suffix}", MISSING if s is None else
                  (f"{s['est'][key]:.2f}" if key == "ae" else pct(s["est"][key])))
    results_table(table_rows, r"Comparison on the same random sample of " + macros["nSub"] +
                  r" test CVEs, on which every system is scored.", "tab:subset",
                  TABLES / "subset.tex")


def tier_table(full: dict) -> None:
    def per_tier(recs):
        yt = np.array([_norm(r["true_severity"]) for r in recs])
        yp = np.array([_norm(r["pred_severity"]) for r in recs])
        out = {}
        for c in SEVERITIES:
            tp = np.sum((yt == c) & (yp == c))
            rec = tp / max(np.sum(yt == c), 1)
            prec = tp / max(np.sum(yp == c), 1)
            out[c] = (rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return out

    systems = [("TF-IDF + LR, direct", "LRd"), ("TF-IDF + LR, formula", "LRf"),
               ("TF-IDF + LR bal., formula", "LRBf"), ("Phi-3.5 + LoRA, formula", "LoRAf")]
    body = []
    for name, prefix in systems:
        recs = full[prefix][1]
        if not recs:
            body.append(f"{name} & " + " & ".join([MISSING] * 8) + r" \\")
            continue
        t = per_tier(recs)
        for c in SEVERITIES:
            macro(f"n{prefix}Rec{c.title()}", pct(t[c][0]))
            macro(f"n{prefix}Fone{c.title()}", pct(t[c][1]))
        n_pred = sum(_norm(r["pred_severity"]) == "CRITICAL" for r in recs)
        n_true = sum(_norm(r["true_severity"]) == "CRITICAL" for r in recs)
        n_hit = sum(_norm(r["pred_severity"]) == _norm(r["true_severity"]) == "CRITICAL" for r in recs)
        macro(f"n{prefix}CritPred", fmt_int(n_pred))
        macro(f"n{prefix}CritTrue", fmt_int(n_true))
        macro(f"n{prefix}CritPrec", pct(n_hit / n_pred if n_pred else None))
        body.append(f"{name} & " + " & ".join(f"{pct(t[c][0])} & {pct(t[c][1])}" for c in SEVERITIES) + r" \\")
    (TABLES / "tiers.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Recall and F1 (\%) by true severity tier on the 2024 test set.}
\label{tab:tiers}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{Critical} & \multicolumn{2}{c}{High} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Low} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
System & Rec. & F1 & Rec. & F1 & Rec. & F1 & Rec. & F1 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def consistency_table() -> None:
    rows = []
    for name, path, prefix in (("Phi-3.5 zero-shot", RESULTS / "summary_phi35_zeroshot_test.json", "ZS"),
                               ("Phi-3.5 + LoRA", RESULTS / "summary_phi35_lora_test.json", "LoRA")):
        s = load_json(path)
        c = s["consistency"] if s else {}
        sc, sv = c.get("score_matches_components"), c.get("severity_matches_score")
        parse = s["systems"][f"{s['tag']}_direct"]["parse_rate"] if s else None
        macro(f"n{prefix}ScoreCons", pct(sc))
        macro(f"n{prefix}SevCons", pct(sv))
        macro(f"n{prefix}Parse", pct(parse))
        rows.append(f"{name} & {pct(parse)} & {pct(sc)} & {pct(sv)} \\\\")
    (TABLES / "consistency.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Self-consistency of language-model outputs (\%), where \emph{Score = formula} is the share of outputs whose written score equals the CVSS v3.1 score of the written vector and \emph{Severity = tier of score} is the share whose written severity matches the tier of the written score.}
\label{tab:consistency}
\small
\begin{tabular}{lccc}
\toprule
Model & Valid JSON & Score = formula & Severity = tier of score \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def paired_differences(full: dict) -> None:
    """Paired bootstrap of LoRA minus the best TF-IDF system on the same test CVEs."""
    from metrics import per_example, macro_f1
    lora = full["LoRAf"][1]
    if not lora:
        return
    rng = np.random.default_rng(0)
    for base in ("LRf", "LRBf"):
        other = full[base][1]
        if not other:
            continue
        by_id = {r["cve_id"]: r for r in other}
        pairs = [(a, by_id[a["cve_id"]]) for a in lora if a["cve_id"] in by_id]
        A = [per_example(a) for a, _ in pairs]
        B = [per_example(b) for _, b in pairs]
        n = len(pairs)
        diffs = {k: np.array([a[k] - b[k] for a, b in zip(A, B)], float) for k in ("sev", "cwe", "ae", "vector")}
        yt = np.array([_norm(a["true_severity"]) for a, _ in pairs])
        pa = np.array([_norm(a["pred_severity"]) for a, _ in pairs])
        pb = np.array([_norm(b["pred_severity"]) for _, b in pairs])
        boots = {k: [] for k in list(diffs) + ["f1"]}
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            for k, d in diffs.items():
                boots[k].append(np.nanmean(d[idx]))
            boots["f1"].append(macro_f1(yt[idx], pa[idx]) - macro_f1(yt[idx], pb[idx]))
        point = {k: float(np.nanmean(d)) for k, d in diffs.items()}
        point["f1"] = macro_f1(yt, pa) - macro_f1(yt, pb)
        for k, name in (("sev", "Sev"), ("cwe", "Cwe"), ("ae", "Mae"), ("vector", "Vec"), ("f1", "Fone")):
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            scale = 1 if k == "ae" else 100
            d = 2 if k == "ae" else 1
            macro(f"nDiff{base}{name}", f"{scale * point[k]:+.{d}f}")
            macro(f"nDiff{base}{name}Lo", f"{scale * lo:+.{d}f}")
            macro(f"nDiff{base}{name}Hi", f"{scale * hi:+.{d}f}")


def component_and_rare_cwe(full: dict) -> None:
    from metrics import COMPONENTS
    names = {"attack_vector": "AV", "attack_complexity": "AC", "privileges_required": "PR",
             "user_interaction": "UI", "scope": "S", "confidentiality": "C", "integrity": "I", "availability": "A"}
    splits = pd.read_pickle(ROOT / "data" / "splits.pkl")
    top = set(splits["train"]["cwe"].value_counts().head(100).index)
    for prefix in ("LRf", "LoRAf"):
        recs = full[prefix][1]
        if not recs:
            continue
        for f in COMPONENTS:
            acc = np.mean([_norm(r["true_components"][f]) == _norm(r["pred_components"].get(f)) for r in recs])
            macro(f"n{prefix}Comp{''.join(c for c in names[f] if c.isalpha())}", pct(acc))
        rare = [r for r in recs if r["true_cwe"] not in top]
        macro(f"n{prefix}RareCwe", pct(np.mean([_norm(r["true_cwe"]) == _norm(r["pred_cwe"]) for r in rare])))
        macro("nRareCweN", fmt_int(len(rare)))


def compute_table() -> None:
    """Wall-clock times read from the run logs."""
    import re
    from datetime import datetime, timedelta

    def stamps(path: Path, start_pat: str, end_pat: str):
        if not path.exists():
            return None
        lines = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text()).splitlines()
        t = lambda l: datetime.strptime(l[1:9], "%H:%M:%S")
        a = next((t(l) for l in lines if l.startswith("[") and re.search(start_pat, l)), None)
        b = next((t(l) for l in lines if l.startswith("[") and re.search(end_pat, l) and a), None)
        if a is None or b is None:
            return None
        d = b - a
        return d + timedelta(days=1) if d.total_seconds() < 0 else d

    def hours(d):
        return MISSING if d is None else f"{d.total_seconds() / 3600:.1f}"

    train = stamps(RESULTS / "log_gpu_attempt1.txt", r"training from scratch", r"ckpt1000:")
    test = stamps(RESULTS / "log_gpu.txt", r"phi35_lora: [0-9,]+ test CVEs", r"phi35_lora_direct")
    tfidf = stamps(RESULTS / "log_classical.txt", r"train \d", r"\] done")
    macro("nLoraTrainHours", hours(train))
    macro("nLoraTestHours", hours(test))
    macro("nTfidfMinutes", MISSING if tfidf is None else f"{tfidf.total_seconds() / 60:.0f}")
    (TABLES / "compute.tex").write_text(r"""\begin{table}[htbp]
\centering
\caption{Wall-clock time on a single Apple M5 Pro (64\,GB), with inference run in small batches of at most 32 CVEs and without optimization for speed.}
\label{tab:compute}
\small
\begin{tabular}{lr}
\toprule
Step & Time \\
\midrule
All TF-IDF baselines, tuning and test (CPU) & """ + macros["nTfidfMinutes"] + r""" min \\
LoRA fine-tuning, 5{,}000 steps & """ + macros["nLoraTrainHours"] + r""" h \\
LoRA inference, """ + macros["nTest"] + r""" test CVEs & """ + macros["nLoraTestHours"] + r""" h \\
\bottomrule
\end{tabular}
\end{table}
""")


# ── Additional analyses for the discussion ───────────────────────────────────

def _wrong(r):
    from metrics import COMPONENTS
    return [f for f in COMPONENTS if _norm(r["true_components"][f]) != _norm(r["pred_components"].get(f))]


def tex_escape(t: str) -> str:
    t = " ".join(t.split())
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("<", r"\textless{}"), (">", r"\textgreater{}")):
        t = t.replace(a, b)
    return t


def vector_errors_and_overlap(full: dict) -> None:
    from collections import Counter
    lrf, lo = full["LRf"][1], full["LoRAf"][1]
    if not lrf or not lo:
        return
    by = {r["cve_id"]: r for r in lrf}
    pairs = [(by[r["cve_id"]], r) for r in lo if r["cve_id"] in by]
    n = len(pairs)
    for prefix, k in (("LRf", 0), ("LoRAf", 1)):
        recs = [p[k] for p in pairs]
        cnt = Counter(min(len(_wrong(r)), 3) for r in recs)
        for name, i in (("Zero", 0), ("One", 1), ("Two", 2), ("ThreePlus", 3)):
            macro(f"n{prefix}Vec{name}", pct(cnt[i] / n))
        sole = Counter(_wrong(r)[0] for r in recs if len(_wrong(r)) == 1)
        tot = sum(sole.values())
        macro(f"n{prefix}SolePR", pct(sole["privileges_required"] / tot, 0))
        macro(f"n{prefix}SoleAvail", pct(sole["availability"] / tot, 0))
    for field, key, tag in (("severity", "pred_severity", "Sev"), ("cwe", "pred_cwe", "Cwe")):
        a = np.array([_norm(x[key]) == _norm(x["true_" + field]) for x, _ in pairs])
        b = np.array([_norm(y[key]) == _norm(y["true_" + field]) for _, y in pairs])
        macro(f"nBoth{tag}", pct(np.mean(a & b))); macro(f"nLoRAOnly{tag}", pct(np.mean(b & ~a)))
        macro(f"nLROnly{tag}", pct(np.mean(a & ~b))); macro(f"nEither{tag}", pct(np.mean(a | b)))
        macro(f"nNeither{tag}", pct(np.mean(~a & ~b)))
    a = np.array([len(_wrong(x)) == 0 for x, _ in pairs]); b = np.array([len(_wrong(y)) == 0 for _, y in pairs])
    macro("nEitherVec", pct(np.mean(a | b))); macro("nLoRAOnlyVec", pct(np.mean(b & ~a))); macro("nLROnlyVec", pct(np.mean(a & ~b)))


def tier_mae_table(full: dict) -> None:
    rows = []
    for name, prefix in (("TF-IDF ridge regression (direct)", "LRd"), ("TF-IDF + LR, formula", "LRf"), ("Phi-3.5 + LoRA, formula", "LoRAf")):
        recs = full[prefix][1]
        if not recs:
            continue
        cells = []
        for c in SEVERITIES:
            d = [abs(r["pred_cvss_score"] - r["true_cvss_score"]) for r in recs
                 if _norm(r["true_severity"]) == c and r["pred_cvss_score"] is not None]
            cells.append(f"{np.mean(d):.2f}")
            macro(f"n{prefix}Mae{c.title()}", f"{np.mean(d):.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    (TABLES / "tiers_mae.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Mean absolute score error by true severity tier on the 2024 test set.}
\label{tab:tiermae}
\small
\begin{tabular}{lcccc}
\toprule
System & Critical & High & Medium & Low \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def length_table(full: dict) -> None:
    test = pd.read_pickle(ROOT / "data" / "splits.pkl")["test"].set_index("cve_id")
    lrf, lo = full["LRf"][1], full["LoRAf"][1]
    if not lrf or not lo:
        return
    lens = pd.Series({r["cve_id"]: len(test.loc[r["cve_id"], "description"]) for r in lo})
    q = pd.qcut(lens, 4, labels=[1, 2, 3, 4])
    bounds = [int(x) for x in lens.quantile([0, .25, .5, .75, 1])]
    for i, b in enumerate(bounds):
        macro(f"nLenB{'ABCDE'[i]}", fmt_int(b))
    rows = []
    for name, prefix, recs in (("TF-IDF + LR, formula", "LRf", lrf), ("Phi-3.5 + LoRA, formula", "LoRAf", lo)):
        by = {r["cve_id"]: r for r in recs}
        cells = []
        for k in (1, 2, 3, 4):
            ids = q[q == k].index
            sev = np.mean([_norm(by[i]["pred_severity"]) == _norm(by[i]["true_severity"]) for i in ids])
            vec = np.mean([len(_wrong(by[i])) == 0 for i in ids])
            macro(f"n{prefix}SevQ{'ABCD'[k-1]}", pct(sev)); macro(f"n{prefix}VecQ{'ABCD'[k-1]}", pct(vec))
            cells.append(f"{pct(sev)} & {pct(vec)}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    (TABLES / "length.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Severity accuracy and exact-vector rate (\%) by quartile of description length on the 2024 test set. Quartile boundaries are """ + f"{bounds[1]}, {bounds[2]}, and {bounds[3]}" + r""" characters.}
\label{tab:length}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{Shortest quartile} & \multicolumn{2}{c}{Second} & \multicolumn{2}{c}{Third} & \multicolumn{2}{c}{Longest quartile} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
System & Sev. & Vector & Sev. & Vector & Sev. & Vector & Sev. & Vector \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def zero_shot_components() -> None:
    from metrics import COMPONENTS
    zs = load_preds(RESULTS / "preds_phi35_zeroshot_formula_test.jsonl.gz")
    zd = load_preds(RESULTS / "preds_phi35_zeroshot_direct_test.jsonl.gz")
    if not zs:
        return
    names = {"attack_vector": "AV", "attack_complexity": "AC", "privileges_required": "PR", "user_interaction": "UI",
             "scope": "S", "confidentiality": "C", "integrity": "I", "availability": "A"}
    for f in COMPONENTS:
        acc = np.mean([_norm(r["true_components"][f]) == _norm(r["pred_components"].get(f)) for r in zs])
        macro(f"nZSComp{names[f]}", pct(acc, 0))
    allowed = {"NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL", "LOW", "HIGH", "NONE", "REQUIRED", "UNCHANGED", "CHANGED"}
    macro("nZSUnparsed", fmt_int(sum(not r["parsed"] for r in zd)))
    macro("nZSInvalid", fmt_int(sum(any(_norm(r["pred_components"].get(f)) not in allowed for f in COMPONENTS) for r in zd if r["parsed"])))


def checkpoint_table() -> None:
    rows = []
    best_f1 = None
    for p in sorted(RESULTS.glob("summary_ckpt*_val.json"), key=lambda x: int(x.stem[len("summary_ckpt"):-len("_val")])):
        s = json.loads(p.read_text())
        step = int(s["tag"][4:])
        e = s["systems"][f"{s['tag']}_formula"]["est"]
        rows.append(f"{fmt_int(step)} & {pct(e['sev'])} & {pct(e['sev_macro_f1'])} & {pct(e['cwe'])} & {e['ae']:.2f} & {pct(e['vector'])} \\\\")
        if best_f1 is None or e["sev_macro_f1"] > best_f1[1]:
            best_f1 = (step, e["sev_macro_f1"])
    if not rows:
        return
    macro("nBestFoneStep", fmt_int(best_f1[0]))
    macro("nBestFoneVal", pct(best_f1[1]))
    (TABLES / "checkpoints.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Validation metrics of the fine-tuned model under formula scoring at each saved checkpoint, on the same 1,000 validation CVEs.}
\label{tab:ckpt}
\small
\begin{tabular}{rccccc}
\toprule
Step & Severity acc. & Macro-F1 & CWE acc. & Score MAE & Exact vector \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


EXAMPLE_IDS = {"CVE-2024-0003": "Purity", "CVE-2024-0008": "PanOS", "CVE-2024-0257": "RoboDK",
               "CVE-2024-10214": "Mattermost", "CVE-2024-0204": "GoAnywhere", "CVE-2024-0149": "Nvidia"}


def examples_table(full: dict) -> None:
    lo = full["LoRAf"][1]
    if not lo:
        return
    by = {r["cve_id"]: r for r in lo}
    test = pd.read_pickle(ROOT / "data" / "splits.pkl")["test"].set_index("cve_id")
    short = {"attack_vector": "AV", "attack_complexity": "AC", "privileges_required": "PR", "user_interaction": "UI",
             "scope": "S", "confidentiality": "C", "integrity": "I", "availability": "A"}
    rows = []
    for cid, tag in EXAMPLE_IDS.items():
        r = by[cid]
        desc = tex_escape(test.loc[cid, "description"])
        if len(desc) > 210:
            desc = desc[:207].rsplit(" ", 1)[0] + "\\,\\ldots"
        wrong = ", ".join(short[f] for f in _wrong(r)) or "none"
        rows.append(rf"\texttt{{{cid}}} & {desc} & {r['true_cvss_score']} {r['true_severity'].title()}, {r['true_cwe']} & "
                    rf"{r['pred_cvss_score']} {str(r['pred_severity']).title()}, {r['pred_cwe']} & {wrong} \\")
        macro(f"nEx{tag}True", f"{r['true_cvss_score']}"); macro(f"nEx{tag}Pred", f"{r['pred_cvss_score']}")
    (TABLES / "examples.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Test CVEs discussed in Section~\ref{sec:discussion}, with the NVD labels and the output of the fine-tuned model. The last column lists the vector components the model got wrong.}
\label{tab:examples}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lp{5.3cm}p{1.9cm}p{1.9cm}l@{}}
\toprule
CVE & Description & NVD & Predicted & Wrong \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ── Tables referenced by the 2026-10-15 rewrite ──────────────────────────────

COMP_NAMES = {"attack_vector": "Attack vector", "attack_complexity": "Attack complexity",
              "privileges_required": "Privileges required", "user_interaction": "User interaction",
              "scope": "Scope", "confidentiality": "Confidentiality", "integrity": "Integrity",
              "availability": "Availability"}


def year_coverage() -> None:
    allrec = pd.read_pickle(ROOT / "data" / "cve_parsed.pkl")
    years = allrec["cve_id"].str.extract(r"CVE-(\d{4})-")[0].astype(int)
    for y, tag in ((2022, "A"), (2023, "B"), (2024, "C")):
        path = ROOT / "data" / f"CVE-{y}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        items = d.get("vulnerabilities", d.get("cve_items", []))
        live = sum(1 for v in items if v.get("cve", v).get("vulnStatus") != "Rejected")
        macro(f"nUsableShare{tag}", pct(int((years == y).sum()) / live))


def agreement_table(full: dict) -> None:
    lrf, lo = full["LRf"][1], full["LoRAf"][1]
    by = {r["cve_id"]: r for r in lrf}
    pairs = [(by[r["cve_id"]], r) for r in lo if r["cve_id"] in by]
    rows = []
    for name, ok in (("Severity", lambda r: _norm(r["pred_severity"]) == _norm(r["true_severity"])),
                     ("CWE", lambda r: _norm(r["pred_cwe"]) == _norm(r["true_cwe"])),
                     ("Exact CVSS vector", lambda r: len(_wrong(r)) == 0)):
        a = np.array([ok(x) for x, _ in pairs]); b = np.array([ok(y) for _, y in pairs])
        cells = [np.mean(a & b), np.mean(b & ~a), np.mean(a & ~b), np.mean(~a & ~b), np.mean(a | b)]
        rows.append(f"{name} & " + " & ".join(pct(c) for c in cells) + r" \\")
    (TABLES / "agreement.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Agreement between fine-tuned Phi-3.5 and TF-IDF logistic regression (formula scoring) on the 2024 test set (\%). The last column is the share of CVEs on which at least one of the two is correct.}
\label{tab:agreement}
\small
\begin{tabular}{lccccc}
\toprule
 & Both correct & Fine-tuned only & TF-IDF only & Neither & At least one \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def value_frequency_table(full: dict) -> None:
    from metrics import COMPONENTS
    lo = full["LoRAf"][1]
    short = {"NETWORK": "Network", "ADJACENT_NETWORK": "Adjacent", "LOCAL": "Local", "PHYSICAL": "Physical",
             "LOW": "Low", "HIGH": "High", "NONE": "None", "REQUIRED": "Required", "UNCHANGED": "Unchanged", "CHANGED": "Changed"}
    rows = []
    for f in COMPONENTS:
        true = pd.Series([_norm(r["true_components"][f]) for r in lo])
        pred = pd.Series([_norm(r["pred_components"].get(f)) for r in lo])
        vals = [v for v in short if (true == v).any()]
        for i, v in enumerate(vals):
            rows.append(f"{COMP_NAMES[f] if i == 0 else ''} & {short[v]} & {pct((true == v).mean())} & {pct((pred == v).mean())} \\\\")
        rows.append(r"\addlinespace[2pt]")
    (TABLES / "value_freq.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Share of 2024 test CVEs (\%) with each value of each CVSS v3.1 component, as assigned by NIST and as predicted by the fine-tuned model.}
\label{tab:valuefreq}
\small
\begin{tabular}{llrr}
\toprule
Component & Value & NVD & Fine-tuned \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def zeroshot_severity_table() -> None:
    zd = load_preds(RESULTS / "preds_phi35_zeroshot_direct_test.jsonl.gz")
    zf = load_preds(RESULTS / "preds_phi35_zeroshot_formula_test.jsonl.gz")
    if not zd:
        return
    n = len(zd)
    rows = []
    for c in SEVERITIES:
        t = sum(_norm(r["true_severity"]) == c for r in zd)
        w = sum(_norm(r["pred_severity"]) == c for r in zd)
        f = sum(_norm(r["pred_severity"]) == c for r in zf)
        rows.append(f"{c.title()} & {fmt_int(t)} ({pct(t / n)}) & {fmt_int(w)} ({pct(w / n)}) & {fmt_int(f)} ({pct(f / n)}) \\\\")
    other = n - sum(_norm(r["pred_severity"]) in SEVERITIES for r in zd)
    rows.append(f"No valid tier & -- & {fmt_int(other)} ({pct(other / n)}) & -- \\\\")
    (TABLES / "zs_severity.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Severity tiers on the """ + macros["nSub"] + r"""-CVE zero-shot sample: as assigned by NIST, as written by zero-shot Phi-3.5, and as computed from the vector it wrote. Counts with shares (\%) in parentheses.}
\label{tab:zssev}
\small
\begin{tabular}{lccc}
\toprule
Tier & NVD & Zero-shot, written & Zero-shot, from vector \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def sole_component_and_gap_tables(full: dict) -> None:
    from collections import Counter
    from metrics import COMPONENTS
    data = {}
    for prefix in ("LoRAf", "LRf"):
        recs = full[prefix][1]
        sole = Counter(_wrong(r)[0] for r in recs if len(_wrong(r)) == 1)
        tot = sum(sole.values())
        sev_ok = [r for r in recs if _norm(r["pred_severity"]) == _norm(r["true_severity"])]
        gap = Counter(min(len(_wrong(r)), 3) for r in sev_ok)
        data[prefix] = (sole, tot, gap, len(sev_ok), len(recs))
        miss = sum(v for k, v in gap.items() if k > 0)
        macro(f"n{prefix}GapOne", pct(gap[1] / miss))
        macro(f"n{prefix}GapShare", pct(miss / len(recs)))
    rows = [f"{COMP_NAMES[f]} & {pct(data['LoRAf'][0][f] / data['LoRAf'][1], 0)} & {pct(data['LRf'][0][f] / data['LRf'][1], 0)} \\\\" for f in COMPONENTS]
    (TABLES / "sole_component.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Which component is wrong when exactly one of the eight is wrong (\% of such CVEs), for the fine-tuned model and TF-IDF logistic regression on the 2024 test set.}
\label{tab:sole}
\small
\begin{tabular}{lcc}
\toprule
Component & Fine-tuned & TF-IDF \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    rows = []
    for name, prefix in (("Fine-tuned Phi-3.5", "LoRAf"), ("TF-IDF + LR, formula", "LRf")):
        sole, tot, gap, nok, n = data[prefix]
        cells = [gap[0] / n, gap[1] / n, gap[2] / n, gap[3] / n]
        rows.append(f"{name} & {pct(nok / n)} & " + " & ".join(pct(c) for c in cells) + r" \\")
    (TABLES / "gap.tex").write_text(r"""\begin{table}[t]
\centering
\caption{CVEs whose severity is predicted correctly, split by how many of the eight vector components are wrong (\% of all 2024 test CVEs). Severity can be right while the vector is not, and most such CVEs have exactly one wrong component.}
\label{tab:gap}
\small
\begin{tabular}{lccccc}
\toprule
 & Severity correct & 0 wrong & 1 wrong & 2 wrong & 3+ wrong \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# Words that can signal the true value of a component. Used only to bold text in the
# qualitative tables; a match is a possible clue, not a label.
CLUES = {
    ("attack_vector", "NETWORK"): ["remote attacker", "remotely", "network", "crafted request", "http", "url", "web"],
    ("attack_vector", "ADJACENT_NETWORK"): ["adjacent", "bluetooth", "wi-fi", "wifi", "same network", "local network"],
    ("attack_vector", "LOCAL"): ["local attacker", "local user", "locally", "local"],
    ("attack_vector", "PHYSICAL"): ["physical access", "physically", "physical", "usb"],
    ("attack_complexity", "HIGH"): ["race condition", "man-in-the-middle", "timing", "non-default", "specific configuration"],
    ("privileges_required", "NONE"): ["unauthenticated", "without authentication", "anonymous", "remote attacker"],
    ("privileges_required", "LOW"): ["authenticated", "low-privileged", "low privileged", "logged-in", "subscriber", "contributor"],
    ("privileges_required", "HIGH"): ["administrator", "administrative", "admin", "high privileges", "privileged", "root"],
    ("user_interaction", "REQUIRED"): ["crafted file", "crafted html page", "visit", "click", "open", "victim", "user"],
    ("scope", "CHANGED"): ["cross-site scripting", "xss", "sandbox", "escape", "guest", "host"],
    ("confidentiality", "HIGH"): ["arbitrary code", "execute", "sensitive", "credentials", "read arbitrary"],
    ("confidentiality", "LOW"): ["limited", "partial", "some information"],
    ("confidentiality", "NONE"): ["denial of service", "crash"],
    ("integrity", "HIGH"): ["arbitrary code", "execute", "modify", "write"],
    ("integrity", "LOW"): ["limited", "inject"],
    ("integrity", "NONE"): ["denial of service", "crash", "information disclosure"],
    ("availability", "HIGH"): ["denial of service", "dos", "crash", "arbitrary code", "execute"],
    ("availability", "LOW"): ["limited", "degrade"],
    ("availability", "NONE"): ["information disclosure", "read", "disclose"],
}


def _bold_clues(desc: str, field: str, value: str) -> str:
    import re as _re
    text = tex_escape(desc)
    for kw in sorted(CLUES.get((field, value), []), key=len, reverse=True):
        pat = _re.compile(r"(?<![\w\\])(" + _re.escape(kw) + r")(?!\w)", _re.I)
        if pat.search(text):
            return pat.sub(lambda m: r"\textbf{" + m.group(1) + "}", text, count=1)
    return text


def qualitative_tables(full: dict) -> None:
    from metrics import COMPONENTS
    test = pd.read_pickle(ROOT / "data" / "splits.pkl")["test"].set_index("cve_id")
    sys_names = [("LoRAf", "FT"), ("LRf", "LR"), ("LRBf", "CB"), ("RFf", "RF")]
    preds = {p: {r["cve_id"]: r for r in full[p][1]} for p, _ in sys_names}
    ids = sorted(preds["LoRAf"])
    short = {"NETWORK": "N", "ADJACENT_NETWORK": "A", "LOCAL": "L", "PHYSICAL": "P", "LOW": "L", "HIGH": "H",
             "NONE": "N", "REQUIRED": "R", "UNCHANGED": "U", "CHANGED": "C"}

    def pick(field, all_wrong):
        out = []
        seen_desc = set()
        for i in ids:
            d = test.loc[i, "description"]
            if len(d) > 230 or d[:60] in seen_desc:
                continue
            wrong = [_norm(preds[p][i]["pred_components"].get(field)) != _norm(preds[p][i]["true_components"][field]) for p, _ in sys_names]
            if (all(wrong) if all_wrong else (any(wrong) and not all(wrong))):
                out.append(i); seen_desc.add(d[:60])
            if len(out) == 4:
                break
        return out

    for all_wrong, tables in ((True, ("qual_all_a", "qual_all_b")), (False, ("qual_some_a", "qual_some_b"))):
        for half, name in zip((COMPONENTS[:4], COMPONENTS[4:]), tables):
            rows = []
            for f in half:
                for k, i in enumerate(pick(f, all_wrong)):
                    true = _norm(preds["LoRAf"][i]["true_components"][f])
                    got = " ".join(f"{s}:{short.get(_norm(preds[p][i]['pred_components'].get(f)), '?')}" for p, s in sys_names)
                    rows.append(f"{COMP_NAMES[f] if k == 0 else ''} & \\texttt{{{i}}} & {_bold_clues(test.loc[i, 'description'], f, true)} & {short.get(true, '?')} & {got} \\\\")
                rows.append(r"\addlinespace[3pt]")
            which = "all four trained systems" if all_wrong else "at least one, but not all, of the four trained systems"
            (TABLES / f"{name}.tex").write_text(r"""\begin{table}[p]
\centering
\caption{Test CVEs on which """ + which + r""" predicted the wrong value for the listed component (""" + ("attack vector, attack complexity, privileges required, and user interaction" if half == COMPONENTS[:4] else "scope and the three impacts") + r"""). Bold marks words in the NVD description that may signal the true value. FT: fine-tuned Phi-3.5, LR: logistic regression, CB: class-balanced logistic regression, RF: random forest, all with formula scoring. Values: N network or none, A adjacent, L local or low, P physical, H high, R required, U unchanged, C changed.}
\label{tab:""" + name.replace("_", "") + r"""}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llp{6.2cm}cl@{}}
\toprule
Component & CVE & Description & True & Predicted \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def summary_table(full: dict) -> None:
    """Main-text results table: no intervals, bold best and underline second best."""
    rows_def = [("TF-IDF + LR", "LRd", "LRf"), ("TF-IDF + CB-LR", "LRBd", "LRBf"), ("TF-IDF + RF", "RFd", "RFf"),
                ("Fine-tuned Phi-3.5", "LoRAd", "LoRAf")]
    metrics_ = [("sev", 1), ("sev_macro_f1", 1), ("cwe", 1), ("ae", -1), ("vector", 1)]
    S = {p: summarize(full[p][1]) for _, a, b in rows_def for p in (a, b) if full[p][1]}
    body = []
    for mode, idx in (("direct", 1), ("formula", 2)):
        prefixes = [r[idx] for r in rows_def]
        for key, sign in metrics_:
            vals = sorted({round(S[p]["est"][key], 3) for p in prefixes}, key=lambda v: -sign * v)
            S.setdefault("_rank", {})[(mode, key)] = vals
        for name, *ps in rows_def:
            p = ps[idx - 1]
            cells = []
            for key, sign in metrics_:
                v = S[p]["est"][key]
                txt = f"{v:.2f}" if key == "ae" else f"{100 * v:.1f}"
                ranks = S["_rank"][(mode, key)]
                r = ranks.index(round(v, 3))
                cells.append(r"\textbf{" + txt + "}" if r == 0 else (r"\underline{" + txt + "}" if r == 1 else txt))
            mark = r"$^{\diamond}$" if mode == "formula" else ""
            body.append(f"{name}{mark} & " + " & ".join(cells) + r" \\")
        body.append(r"\midrule" if mode == "direct" else "")
    (TABLES / "summary.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Results on enriching test (2024) CVEs. Systems marked $^{\diamond}$ predict CVSS scores by predicting vectors and computing the score using the CVSS formula. For each scoring method, bold highlights the best performing system and underline indicates second best. See Table~\ref{tab:main} for all results.}
\label{tab:summary}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
System & Severity acc. & Severity macro-F1 & CWE acc. & Score MAE $\downarrow$ & Exact CVSS \\
\midrule
""" + "\n".join(b for b in body if b) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def misc_numbers() -> None:
    # Training ran in the first pipeline attempt; the second resumed at evaluation.
    log = next((p for p in (RESULTS / "log_gpu_attempt1.txt", RESULTS / "log_gpu.txt")
                if p.exists() and "Trainable parameters" in p.read_text()), RESULTS / "log_gpu.txt")
    params = pct_params = MISSING
    if log.exists():
        import re
        for line in re.sub(r"\x1b\[[0-9;]*m", "", log.read_text()).splitlines():
            if line.startswith("Trainable parameters"):
                pct_params = line.split("%")[0].split()[-1]
                params = f"{float(line.split('(')[1].split('M')[0]):.1f}"
    macro("nLoraParams", params)
    macro("nLoraPct", pct_params)
    n_train = len(pd.read_pickle(ROOT / "data" / "splits.pkl")["train"])
    lora_train = ROOT / "data" / "lora" / "train.jsonl"
    n_kept = sum(1 for _ in open(lora_train)) if lora_train.exists() else None
    macro("nDropped", fmt_int(n_train - n_kept) if n_kept is not None else MISSING)
    macro("nLoraTrain", fmt_int(n_kept) if n_kept is not None else MISSING)
    macro("nDroppedPct", pct((n_train - n_kept) / n_train) if n_kept is not None else MISSING)
    sel = load_json(RESULTS / "checkpoint_selection.json")
    macro("nBestStep", fmt_int(sel["best_step"]) if sel else MISSING)


def score_peak_numbers() -> None:
    """Share of test CVEs scored exactly 9.8 by the fine-tuned model and by NVD."""
    recs = load_preds(RESULTS / "preds_phi35_lora_formula_test.jsonl.gz")
    if not recs:
        macro("nLoRAfNineEight", MISSING); macro("nNvdNineEight", MISSING)
        return
    p = np.array([np.nan if r["pred_cvss_score"] is None else r["pred_cvss_score"] for r in recs])
    t = np.array([r["true_cvss_score"] for r in recs])
    macro("nLoRAfNineEight", pct(np.mean(p == 9.8)))
    macro("nNvdNineEight", pct(np.mean(t == 9.8)))
    for stem, prefix in (("preds_lr_direct", "LRd"), ("preds_lr_formula", "LRf"),
                         ("preds_phi35_lora_formula_test", "LoRAf")):
        rs = load_preds(RESULTS / f"{stem}.jsonl.gz")
        if not rs:
            continue
        pp = np.array([np.nan if r["pred_cvss_score"] is None else r["pred_cvss_score"] for r in rs])
        tt = np.array([r["true_cvss_score"] for r in rs])
        macro(f"n{prefix}Bias", f"{np.nanmean(pp - tt):+.2f}")  # mean signed score error
        if prefix == "LRf":
            macro("nLRfNineEight", pct(np.mean(pp == 9.8)))


def prompts() -> None:
    def verbatim(s: str) -> str:
        return "\\begin{verbatim}\n" + s.replace("{description}", "<CVE description>").replace("{{", "{").replace("}}", "}") + "\n\\end{verbatim}"
    import textwrap
    wrap = lambda s: "\n".join(textwrap.fill(p, 78) if p else "" for p in s.split("\n"))
    (TABLES / "prompts.tex").write_text(
        "\\paragraph{Fine-tuned model (user turn).}\n{\\footnotesize " + verbatim(wrap(USER_TEMPLATE)) + "}\n\n"
        "\\paragraph{Zero-shot model (user turn).}\n{\\footnotesize " + verbatim(wrap(ZERO_SHOT_TEMPLATE)) + "}\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    data_section()
    full = main_results()
    subset_results(full)
    tier_table(full)
    consistency_table()
    paired_differences(full)
    vector_errors_and_overlap(full)
    tier_mae_table(full)
    length_table(full)
    zero_shot_components()
    checkpoint_table()
    examples_table(full)
    year_coverage()
    agreement_table(full)
    value_frequency_table(full)
    zeroshot_severity_table()
    sole_component_and_gap_tables(full)
    qualitative_tables(full)
    summary_table(full)
    component_and_rare_cwe(full)
    compute_table()
    misc_numbers()
    prompts()
    score_peak_numbers()
    lines = ["% Generated by src/make_tables.py from results/. Do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros and {len(list(TABLES.glob('*.tex')))} tables")


if __name__ == "__main__":
    main()
