"""
Zero-shot LLM baseline via OpenRouter (open-source models) and Claude API (frontier upper bound).

Usage:
  # Open-source model via OpenRouter
  python src/baselines_llm.py --model qwen/qwen-2.5-72b-instruct --n 200
  python src/baselines_llm.py --model meta-llama/llama-3.3-70b-instruct --n 200

  # Frontier model upper bound
  python src/baselines_llm.py --model claude-opus-4-7 --provider anthropic --n 200

Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY in .env
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from prepare_splits import make_splits
from evaluate import (
    parse_json_response, summarize, print_summary, save_results,
    CVSS_VECTOR_FIELDS,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in vulnerability analysis.
Given a CVE description, predict the structured NVD enrichment fields.
Respond ONLY with a valid JSON object — no explanation, no markdown, no extra text."""

USER_TEMPLATE = """CVE Description:
{description}

Predict the following fields and return them as JSON:
{{
  "cvss_score": <float 0.0-10.0>,
  "severity": <"CRITICAL"|"HIGH"|"MEDIUM"|"LOW">,
  "cwe": <"CWE-NNN">,
  "attack_vector": <"NETWORK"|"ADJACENT"|"LOCAL"|"PHYSICAL">,
  "attack_complexity": <"LOW"|"HIGH">,
  "privileges_required": <"NONE"|"LOW"|"HIGH">,
  "user_interaction": <"NONE"|"REQUIRED">,
  "scope": <"UNCHANGED"|"CHANGED">,
  "confidentiality": <"NONE"|"LOW"|"HIGH">,
  "integrity": <"NONE"|"LOW"|"HIGH">,
  "availability": <"NONE"|"LOW"|"HIGH">
}}"""


def _query_openrouter(desc: str, model: str, client) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(description=desc)},
        ],
        temperature=0,
        max_tokens=512,
        extra_headers={"HTTP-Referer": "https://github.com/cve-triage"},
    )
    return response.choices[0].message.content or ""


def _query_anthropic(desc: str, model: str, client) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(description=desc)}],
    )
    return response.content[0].text if response.content else ""


def _make_result(row, pred: dict, raw: str, model: str) -> dict:
    return {
        "model":            model,
        "cve_id":           row["cve_id"],
        "parsed":           bool(pred),
        "raw_response":     raw,
        "true_severity":    row["severity"],
        "pred_severity":    str(pred.get("severity", "")).upper(),
        "true_cwe":         row["cwe"],
        "pred_cwe":         str(pred.get("cwe", "")),
        "true_cvss_score":  float(row["cvss_score"]),
        "pred_cvss_score":  float(pred.get("cvss_score", 5.0)),
        "true_cvss_vector": row["cvss_vector"],
        "pred_cvss_vector": "",
        "true_components":  {f: row[f] for f in CVSS_VECTOR_FIELDS},
        "pred_components": {
            "attack_vector":       str(pred.get("attack_vector", "")).upper(),
            "attack_complexity":   str(pred.get("attack_complexity", "")).upper(),
            "privileges_required": str(pred.get("privileges_required", "")).upper(),
            "user_interaction":    str(pred.get("user_interaction", "")).upper(),
            "scope":               str(pred.get("scope", "")).upper(),
            "confidentiality":     str(pred.get("confidentiality", "")).upper(),
            "integrity":           str(pred.get("integrity", "")).upper(),
            "availability":        str(pred.get("availability", "")).upper(),
        },
    }


def run(model: str, provider: str, n_samples: int, sample_seed: int = 42) -> None:
    splits = make_splits()
    test = splits["test"]
    if n_samples > 0:
        test = test.sample(min(n_samples, len(test)), random_state=sample_seed).reset_index(drop=True)
    print(f"\nEvaluating {model} on {len(test)} test CVEs ...")

    # Build client
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        query_fn = lambda desc: _query_anthropic(desc, model, client)
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        query_fn = lambda desc: _query_openrouter(desc, model, client)

    results = []
    model_slug = model.replace("/", "_").replace(".", "_")

    for _, row in tqdm(test.iterrows(), total=len(test)):
        raw = ""
        pred = {}
        try:
            raw = query_fn(row["description"])
            pred = parse_json_response(raw)
        except Exception as e:
            print(f"\n[warn] {row['cve_id']}: {e}")
            time.sleep(2)

        results.append(_make_result(row, pred, raw, model))
        time.sleep(0.3)  # rate limit

    out_path = RESULTS_DIR / f"results_llm_{model_slug}.jsonl"
    save_results(results, out_path)

    summary = summarize(model, results)
    print_summary(summary)

    with open(RESULTS_DIR / f"summary_llm_{model_slug}.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "per_component_acc"}, f, indent=2)


OPEN_SOURCE_MODELS = [
    ("qwen/qwen-2.5-72b-instruct",        "openrouter"),
    ("meta-llama/llama-3.3-70b-instruct", "openrouter"),
    ("mistralai/mistral-7b-instruct",     "openrouter"),
    ("google/gemini-flash-1.5",           "openrouter"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "anthropic"])
    parser.add_argument("--n",        type=int, default=200, help="number of test samples (0=all)")
    parser.add_argument("--all-open-source", action="store_true",
                        help="run all open-source models sequentially")
    args = parser.parse_args()

    if args.all_open_source:
        for model, provider in OPEN_SOURCE_MODELS:
            run(model, provider, args.n)
    else:
        run(args.model, args.provider, args.n)
