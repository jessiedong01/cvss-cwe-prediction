#!/usr/bin/env bash
# Full pipeline for the paper. Run from the repo root on Apple silicon.
set -euo pipefail

python src/download_data.py 2022 2023 2024
python src/prepare_splits.py
python src/baselines_classical.py

python src/prepare_lora_data.py
python -m mlx_lm.lora --config scripts/lora_phi35.yaml
python src/select_checkpoint.py --adapter-dir adapters/phi35
BEST=$(python -c "import json; print(json.load(open('results/checkpoint_selection.json'))['best_file'])")
python src/evaluate_lora.py --adapter-file "$BEST" --split test --n 0 --tag phi35_lora
python src/evaluate_lora.py --zero-shot --split test --n 2000 --seed 0 --tag phi35_zeroshot

python src/make_tables.py
python src/plot_results.py
(cd paper && tectonic main.tex)
