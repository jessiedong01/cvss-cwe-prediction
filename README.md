# CVSS and CWE Prediction

This repository predicts the NVD fields for a CVE from its description alone. The model outputs the CVSS v3.1 vector, the base score, the severity, and the CWE. Since April 2026 the NVD fills in these fields only for high-priority CVEs. We fine-tune Phi-3.5-mini with LoRA and compare it with TF-IDF classifiers and zero-shot prompting. Models train on CVEs from 2022 and 2023 and are tested on every usable CVE from 2024.

## Reproducing the paper

Training and evaluation of the language model need Apple silicon. The TF-IDF baselines run anywhere. Building the PDF needs [tectonic](https://tectonic-typesetting.github.io).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install pandas scikit-learn numpy tqdm matplotlib requests mlx-lm transformers
bash scripts/run_all.sh
```

`run_all.sh` downloads a pinned NVD snapshot, builds the splits, runs every baseline, fine-tunes the model, picks a checkpoint on validation data, evaluates on the test set, and rebuilds the paper's tables, figures, and PDF. Per-CVE predictions for every system are in `results/`.

## Citation

```bibtex
@misc{dixit2026predictingcvss,
    title={Predicting CVSS and CWE from CVE Descriptions with Fine-Tuned Small Language Models},
    author={Tara Dixit and Jessie Dong and Summer Li},
    year={2026},
}
```
