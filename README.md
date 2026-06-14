# CVE Triage — SLM-Based Vulnerability Enrichment

**CS229 Final Project** | Stanford University

Trains small language models to predict NVD enrichment fields (CVSS score, severity, CWE, CVSS vector components) directly from a raw CVE description — replicating what NIST's NVD produces, but instantly.

## Motivation

NIST's National Vulnerability Database fell catastrophically behind in 2023 and in April 2026 announced it would stop adding severity scores entirely. With 131 new CVEs per day, attackers don't wait for triage. This project builds open-source ML tooling to fill that gap using only publicly available data and consumer hardware (16GB MacBook M1).

## Task

| Input | Raw CVE description text |
|-------|--------------------------|
| Output 1 | CVSS base score (regression, 0.0–10.0) |
| Output 2 | 8 CVSS vector components (AV, AC, PR, UI, S, C, I, A) |
| Output 3 | CWE weakness class (~150 categories) |
| Output 4 | Severity bucket (CRITICAL / HIGH / MEDIUM / LOW) |

## Models

| Approach | Details |
|----------|---------|
| TF-IDF + Logistic Regression | Classical baseline (interpretable, fast) |
| TF-IDF + Random Forest | Classical baseline (non-linear) |
| Zero-shot LLM (Qwen 72B, Llama 3.3 70B) | Open-source via OpenRouter |
| Zero-shot frontier (Claude) | Practical upper bound |
| LoRA fine-tuned Phi-3.5-mini (3.8B) | Primary fine-tuned model |
| LoRA fine-tuned Qwen 2.5 0.5B | Lower-bound scale probe |

## Data

Source: [fkie-cad/nvd-json-data-feeds](https://github.com/fkie-cad/nvd-json-data-feeds) — NVD API 2.0 JSON, updated every 2 hours, free and public.

**Temporal split (no leakage):**
- Train + val: 2022–2023 CVEs (~160K usable records, 80/20 split)
- Test: 2024 CVEs (held out entirely, ~55K records)

## Setup

```bash
git clone https://github.com/<you>/cve-triage
cd cve-triage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Apple Silicon LoRA training:
pip install mlx-lm

# API keys (for LLM baselines):
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY
```

## Running Experiments

```bash
# Full pipeline (Steps 1–11):
bash scripts/run_all.sh

# Or step by step:
python src/download_data.py              # download 2022/2023/2024 feeds
python src/parse_data.py                 # parse → data/cve_parsed.pkl
python src/eda.py                        # plots → results/eda_*.png
python src/prepare_splits.py             # splits → data/splits.pkl

python src/baselines_classical.py        # TF-IDF + LR + RF

python src/baselines_llm.py \
    --model qwen/qwen-2.5-72b-instruct \
    --n 500                              # zero-shot open-source model

python src/prepare_lora_data.py          # → data/train.jsonl, data/valid.jsonl

python -m mlx_lm.lora \
    --config scripts/lora_config_phi35.yaml   # train Phi-3.5 adapter

python src/evaluate_lora.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --adapter adapters/phi35 \
    --n 500                              # evaluate on held-out 2024 CVEs

python src/compare_results.py --latex    # → results/comparison.csv + .tex
```

## Evaluation Metrics

- **CVSS MAE** — mean absolute error on 0–10 score
- **Severity accuracy** — exact match on 4-class label
- **CWE accuracy** — top-1 exact match across ~150 classes
- **CVSS vector accuracy** — full vector string exact match
- **Per-component accuracy** — separate accuracy for each of 8 CVSS fields

## Project Structure

```
cve-triage/
├── src/
│   ├── download_data.py        # Download NVD JSON feeds
│   ├── parse_data.py           # Parse & clean CVE records
│   ├── prepare_splits.py       # Temporal train/val/test splits
│   ├── eda.py                  # Exploratory data analysis
│   ├── baselines_classical.py  # TF-IDF + LR/RF
│   ├── baselines_llm.py        # Zero-shot LLM prompting
│   ├── prepare_lora_data.py    # Format JSONL for MLX LoRA
│   ├── evaluate_lora.py        # Evaluate trained adapter
│   ├── evaluate.py             # Shared metrics utilities
│   └── compare_results.py      # Comparison table + failure analysis
├── scripts/
│   ├── lora_config_phi35.yaml  # MLX-LM config for Phi-3.5
│   ├── lora_config_qwen.yaml   # MLX-LM config for Qwen 0.5B
│   └── run_all.sh              # Full pipeline script
├── data/                       # Downloaded and parsed data (gitignored)
└── results/                    # JSONL results + plots (gitignored)
```

## Prior Work

- [CVE-LLM (AAAI 2025, Siemens)](https://arxiv.org/abs/...) — fine-tuned LLMs run 50–100× faster than human analysts
- [arxiv 2603.14911](https://arxiv.org/abs/2603.14911) — fine-tuned RoBERTa-125M matches 8B accuracy on CVE-to-CWE mapping
- [CIRCL vulnerability-scores](https://huggingface.co/CIRCL/vulnerability-scores) — RoBERTa severity baseline
