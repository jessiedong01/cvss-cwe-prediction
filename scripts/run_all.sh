#!/usr/bin/env bash
# Full experiment pipeline — run from the repo root.
# Edit flags as needed (--n controls sample size for LLM baselines).
set -euo pipefail

echo "=== Step 1: Download NVD data ==="
python src/download_data.py 2022 2023 2024

echo ""
echo "=== Step 2: Parse & cache CVE records ==="
python src/parse_data.py

echo ""
echo "=== Step 3: EDA ==="
python src/eda.py

echo ""
echo "=== Step 4: Create train/val/test splits ==="
python src/prepare_splits.py

echo ""
echo "=== Step 5a: Classical baselines (TF-IDF + LR and RF) ==="
python src/baselines_classical.py

echo ""
echo "=== Step 5b: LLM zero-shot baseline (Qwen 72B via OpenRouter) ==="
python src/baselines_llm.py --model qwen/qwen-2.5-72b-instruct --n 500

echo ""
echo "=== Step 5c: Frontier model upper bound (Claude via OpenRouter) ==="
python src/baselines_llm.py --model anthropic/claude-opus-4 --n 500

echo ""
echo "=== Step 6: Prepare LoRA training data ==="
python src/prepare_lora_data.py

echo ""
echo "=== Step 7: Train Phi-3.5-mini LoRA adapter ==="
echo "  (requires: pip install mlx-lm, Apple Silicon Mac)"
python -m mlx_lm.lora --config scripts/lora_config_phi35.yaml

echo ""
echo "=== Step 8: Evaluate Phi-3.5 LoRA adapter ==="
python src/evaluate_lora.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --adapter adapters/phi35 \
    --n 500

echo ""
echo "=== Step 9: Train Qwen 0.5B LoRA adapter (lower bound) ==="
python -m mlx_lm.lora --config scripts/lora_config_qwen.yaml

echo ""
echo "=== Step 10: Evaluate Qwen 0.5B LoRA adapter ==="
python src/evaluate_lora.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter adapters/qwen05b \
    --n 500

echo ""
echo "=== Step 11: Comparison table ==="
python src/compare_results.py --latex

echo ""
echo "Pipeline complete. Results in results/"
