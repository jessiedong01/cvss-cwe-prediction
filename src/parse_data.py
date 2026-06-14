"""
Parse NVD API 2.0 JSON feeds into a flat pandas DataFrame.
Handles both cvssMetricV31 and cvssMetricV30; prefers V31 Primary source.
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "cve_parsed.pkl"


def _en_description(descriptions: list) -> Optional[str]:
    for d in descriptions:
        if d.get("lang") == "en":
            val = d.get("value", "").strip()
            return val if val else None
    return None


def _cvss_data(metrics: dict) -> dict:
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        for source_pref in ("nvd@nist.gov", None):
            for e in entries:
                if source_pref is None or e.get("source") == source_pref:
                    if e.get("type") == "Primary":
                        return e["cvssData"]
        if entries:
            return entries[0]["cvssData"]
    return {}


def _cwe(weaknesses: list) -> Optional[str]:
    candidates = []
    for w in weaknesses:
        for d in w.get("description", []):
            if d.get("lang") == "en":
                val = d.get("value", "")
                if val.startswith("CWE-") and val not in ("NVD-CWE-noinfo", "NVD-CWE-Other"):
                    candidates.append((w.get("type") == "Primary", val))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def parse_file(json_path: Path) -> pd.DataFrame:
    print(f"Parsing {json_path.name} ...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    # Support both NVD API format (vulnerabilities[].cve) and fkie-cad feed (cve_items[])
    items = data.get("vulnerabilities", data.get("cve_items", []))
    for vuln in items:
        cve = vuln.get("cve", vuln)  # unwrap if nested, else use directly

        desc = _en_description(cve.get("descriptions", []))
        if not desc or len(desc) < 20:
            continue

        cvss = _cvss_data(cve.get("metrics", {}))
        if not cvss or "baseScore" not in cvss:
            continue
        # Score 0.0 has severity NONE, which is not a triage tier.
        if cvss.get("baseSeverity", "").upper() not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            continue

        cwe = _cwe(cve.get("weaknesses", []))
        if not cwe:
            continue

        records.append({
            "cve_id":                cve.get("id", ""),
            "description":           desc,
            "cvss_score":            float(cvss.get("baseScore", 0.0)),
            "cvss_vector":           cvss.get("vectorString", ""),
            "severity":              cvss.get("baseSeverity", "").upper(),
            "cwe":                   cwe,
            "attack_vector":         cvss.get("attackVector", ""),
            "attack_complexity":     cvss.get("attackComplexity", ""),
            "privileges_required":   cvss.get("privilegesRequired", ""),
            "user_interaction":      cvss.get("userInteraction", ""),
            "scope":                 cvss.get("scope", ""),
            "confidentiality":       cvss.get("confidentialityImpact", ""),
            "integrity":             cvss.get("integrityImpact", ""),
            "availability":          cvss.get("availabilityImpact", ""),
        })

    df = pd.DataFrame(records)
    print(f"  -> {len(df):,} usable records (desc+cvss+cwe complete)")
    return df


def load_all(years: list[int], force: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached data from {CACHE_PATH.name}")
        return pd.read_pickle(CACHE_PATH)

    dfs = [parse_file(DATA_DIR / f"CVE-{y}.json") for y in years]
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(CACHE_PATH)
    print(f"Cached {len(df):,} total records -> {CACHE_PATH.name}")
    return df


if __name__ == "__main__":
    df = load_all([2022, 2023, 2024])
    print(f"\nTotal records: {len(df):,}")
    print(df["severity"].value_counts())
    print(df["cwe"].value_counts().head(15))
    print(f"CVSS score range: {df['cvss_score'].min()} – {df['cvss_score'].max()}")
    print(f"Unique CWEs: {df['cwe'].nunique()}")
