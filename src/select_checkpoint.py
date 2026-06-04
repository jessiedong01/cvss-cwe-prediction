"""
Choose the LoRA checkpoint on validation data only.

Evaluates every saved checkpoint on the same 1,000 validation CVEs and picks
the one with the highest severity accuracy of the formula system. Writes
results/checkpoint_selection.json.

Usage:
  python src/select_checkpoint.py --adapter-dir adapters/phi35
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default="adapters/phi35")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--every", type=int, default=1000, help="only evaluate steps divisible by this")
    args = ap.parse_args()

    ckpts = sorted(Path(args.adapter_dir).glob("*_adapters.safetensors"))
    scores = {}
    for ck in ckpts:
        step = int(ck.name.split("_")[0])
        if step % args.every:
            continue
        tag = f"ckpt{step}"
        summary = RESULTS / f"summary_{tag}_val.json"
        if not summary.exists():
            subprocess.run([sys.executable, str(ROOT / "src" / "evaluate_lora.py"),
                            "--adapter-file", str(ck), "--split", "val", "--n", str(args.n),
                            "--tag", tag], check=True)
        s = json.loads(summary.read_text())
        scores[step] = s["systems"][f"{tag}_formula"]["est"]["sev"]
        print(f"step {step}: val severity accuracy (formula) {scores[step]:.4f}", flush=True)

    best = max(scores, key=scores.get)
    best_file = Path(args.adapter_dir) / f"{best:07d}_adapters.safetensors"
    out = {"criterion": "validation severity accuracy, formula system", "every": args.every,
           "n_val": args.n, "scores": scores, "best_step": best, "best_file": str(best_file)}
    (RESULTS / "checkpoint_selection.json").write_text(json.dumps(out, indent=2))
    print(f"best: step {best} -> {best_file}")


if __name__ == "__main__":
    main()
