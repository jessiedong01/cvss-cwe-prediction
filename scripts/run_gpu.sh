#!/usr/bin/env bash
# All GPU steps for the paper, in order. Safe to rerun: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*"; }

ADIR=adapters/phi35
if [ ! -f "$ADIR/0005000_adapters.safetensors" ]; then
  # A partial run cannot be resumed faithfully (mlx_lm restarts the LR schedule), so start over.
  log "training from scratch"
  rm -rf "$ADIR"
  python -m mlx_lm.lora --config scripts/lora_phi35.yaml
fi
log "training done"

python src/select_checkpoint.py --adapter-dir "$ADIR"
BEST=$(python -c "import json; print(json.load(open('results/checkpoint_selection.json'))['best_file'])")
log "best checkpoint: $BEST"

[ -f results/summary_phi35_lora_test.json ] || \
  python src/evaluate_lora.py --adapter-file "$BEST" --split test --n 0 --tag phi35_lora
[ -f results/summary_phi35_zeroshot_test.json ] || \
  python src/evaluate_lora.py --zero-shot --split test --n 2000 --seed 0 --tag phi35_zeroshot

python src/make_tables.py
python src/plot_results.py
log "ALL DONE"
