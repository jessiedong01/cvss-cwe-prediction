"""
Evaluate Phi-3.5 (fine-tuned with a LoRA adapter, or zero-shot) with MLX.

Each run writes two systems:
  <tag>_direct   the score and severity the model wrote
  <tag>_formula  score and severity computed from the model's own components

It also measures self-consistency: does the model's score match the CVSS
formula applied to its own components, and does its severity match its score?

Usage:
  # pick a checkpoint on validation
  python src/evaluate_lora.py --adapter-file adapters/phi35/0002000_adapters.safetensors --split val --n 1000
  # final test run
  python src/evaluate_lora.py --adapter-file adapters/phi35/adapters.safetensors --split test --n 0 --tag phi35_lora
  # zero-shot baseline on a random test subset
  python src/evaluate_lora.py --zero-shot --split test --n 2000 --tag phi35_zeroshot
"""
import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cvss import base_score, severity
from metrics import COMPONENTS, summarize
from prepare_lora_data import USER_TEMPLATE, clip
from prepare_splits import make_splits

RESULTS_DIR = Path(__file__).parent.parent / "results"
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

ZERO_SHOT_TEMPLATE = (
    "You are a cybersecurity expert specializing in vulnerability analysis. "
    "Given a CVE description, predict the NVD CVSS v3.1 enrichment fields.\n\n"
    "CVE Description:\n{description}\n\n"
    "Respond ONLY with one JSON object with exactly these keys and allowed values:\n"
    '{{"attack_vector": "NETWORK"|"ADJACENT_NETWORK"|"LOCAL"|"PHYSICAL", '
    '"attack_complexity": "LOW"|"HIGH", "privileges_required": "NONE"|"LOW"|"HIGH", '
    '"user_interaction": "NONE"|"REQUIRED", "scope": "UNCHANGED"|"CHANGED", '
    '"confidentiality": "NONE"|"LOW"|"HIGH", "integrity": "NONE"|"LOW"|"HIGH", '
    '"availability": "NONE"|"LOW"|"HIGH", "cwe": "CWE-<number>", '
    '"cvss_score": <number 0.0-10.0>, "severity": "CRITICAL"|"HIGH"|"MEDIUM"|"LOW"}}'
)

TOKEN_BUDGET = 12_000  # max (batch size x longest prompt) per batch; kept small so other GPU jobs fit
MAX_BATCH = 32
OOM_WAIT_SECONDS = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse(text: str) -> dict:
    """First JSON object in the text, or {}."""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    for candidate in ([m.group(0)] if m else []) + [text.strip()]:
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(adapter_file: str | None):
    from mlx_lm import load as mlx_load
    if adapter_file is None:
        return mlx_load(BASE_MODEL)
    # mlx_lm loads <dir>/adapters.safetensors; stage the chosen checkpoint there.
    src = Path(adapter_file)
    tmp = Path(tempfile.mkdtemp(prefix="adapter_"))
    shutil.copy(src.parent / "adapter_config.json", tmp / "adapter_config.json")
    shutil.copy(src, tmp / "adapters.safetensors")
    return mlx_load(BASE_MODEL, adapter_path=str(tmp))


def batches(lengths: list[int]):
    """Yield index batches of similar-length prompts within the token budget."""
    order = np.argsort(lengths)
    batch: list[int] = []
    for i in order:
        if batch and (len(batch) + 1) * lengths[i] > TOKEN_BUDGET or len(batch) == MAX_BATCH:
            yield batch
            batch = []
        batch.append(int(i))
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-file", default=None)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n", type=int, default=1000, help="random subset size (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    assert args.zero_shot != bool(args.adapter_file), "pass --zero-shot or --adapter-file"

    from mlx_lm.generate import batch_generate

    df = make_splits()[args.split]
    if args.n:
        df = df.sample(min(args.n, len(df)), random_state=args.seed)
    rows = df.to_dict("records")
    tag = args.tag or ("zeroshot" if args.zero_shot else Path(args.adapter_file).stem)
    log(f"{tag}: {len(rows):,} {args.split} CVEs")

    model, tok = load(None if args.zero_shot else args.adapter_file)
    template = ZERO_SHOT_TEMPLATE if args.zero_shot else USER_TEMPLATE
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": template.format(description=clip(r["description"]))}],
                                add_generation_prompt=True)
        for r in rows
    ]

    # Partial results survive crashes: each finished batch is appended here and skipped on restart.
    RESULTS_DIR.mkdir(exist_ok=True)
    partial = RESULTS_DIR / f"partial_{tag}_{args.split}.jsonl"
    done_text: dict[str, str] = {}
    if partial.exists():
        for line in open(partial):
            r = json.loads(line)
            done_text[r["cve_id"]] = r["raw"]
        log(f"resuming: {len(done_text):,} CVEs already done")
    texts = [done_text.get(r["cve_id"]) for r in rows]
    todo = [i for i, t in enumerate(texts) if t is None]

    def generate(idx: list[int]) -> list[str]:
        """Generate for idx; on GPU out-of-memory, free memory and retry in halves."""
        import mlx.core as mx
        try:
            return batch_generate(model, tok, [prompts[i] for i in idx], max_tokens=args.max_tokens).texts
        except RuntimeError as e:
            if "Insufficient Memory" not in str(e) and "OutOfMemory" not in str(e):
                raise
            mx.clear_cache()
            if len(idx) == 1:
                log(f"  out of GPU memory on a single CVE; waiting {OOM_WAIT_SECONDS}s")
                time.sleep(OOM_WAIT_SECONDS)
                return generate(idx)
            log(f"  out of GPU memory at batch {len(idx)}; retrying in halves")
            mid = len(idx) // 2
            return generate(idx[:mid]) + generate(idx[mid:])

    done, t0 = 0, time.time()
    with open(partial, "a") as pf:
        for idx in batches([len(prompts[i]) for i in todo]):
            idx = [todo[i] for i in idx]
            for i, t in zip(idx, generate(idx)):
                texts[i] = t
                pf.write(json.dumps({"cve_id": rows[i]["cve_id"], "raw": t}) + "\n")
            pf.flush()
            done += len(idx)
            if done % 1000 < len(idx) or done == len(todo):
                rate = done / (time.time() - t0)
                log(f"  {done:,}/{len(todo):,}  {rate:.1f} CVE/s  eta {(len(todo) - done) / rate / 60:.0f} min")

    direct, formula, consist = [], [], {"score_matches_components": [], "severity_matches_score": []}
    for row, text in zip(rows, texts):
        pred = parse(text)
        comps = {f: str(pred.get(f, "")).upper() for f in COMPONENTS}
        own_score = to_float(pred.get("cvss_score"))
        own_sev = str(pred.get("severity", "")).upper()
        f_score = base_score(comps)
        base = {
            "cve_id": row["cve_id"], "parsed": bool(pred), "raw": text,
            "true_severity": row["severity"], "true_cwe": row["cwe"],
            "true_cvss_score": row["cvss_score"],
            "true_components": {f: row[f] for f in COMPONENTS},
            "pred_cwe": str(pred.get("cwe", "")).upper(), "pred_components": comps,
        }
        direct.append({**base, "system": f"{tag}_direct", "pred_cvss_score": own_score, "pred_severity": own_sev})
        formula.append({**base, "system": f"{tag}_formula", "pred_cvss_score": f_score, "pred_severity": severity(f_score)})
        if own_score is not None and f_score is not None:
            consist["score_matches_components"].append(own_score == f_score)
        if own_score is not None and own_sev:
            consist["severity_matches_score"].append(severity(own_score) == own_sev)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {"tag": tag, "split": args.split, "n": len(rows), "seed": args.seed,
               "adapter_file": args.adapter_file,
               "consistency": {k: float(np.mean(v)) if v else None for k, v in consist.items()},
               "systems": {}}
    for records in (direct, formula):
        name = records[0]["system"]
        with gzip.open(RESULTS_DIR / f"preds_{name}_{args.split}.jsonl.gz", "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        s = summarize(records)
        summary["systems"][name] = s
        e = s["est"]
        log(f"{name:28s} parse {s['parse_rate']:.3f}  sev {e['sev']:.3f}  macroF1 {e['sev_macro_f1']:.3f}  "
            f"cwe {e['cwe']:.3f}  MAE {e['ae']:.3f}  partial {e['partial']:.3f}  vector {e['vector']:.3f}")
    log(f"consistency: {summary['consistency']}")
    (RESULTS_DIR / f"summary_{tag}_{args.split}.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
