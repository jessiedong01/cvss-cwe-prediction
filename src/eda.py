"""
Exploratory data analysis — run once after parse_data.py to understand the dataset.
Outputs plots to results/eda_*.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from parse_data import load_all

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_severity_dist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sns.countplot(data=df, x="severity", order=order, ax=axes[0])
    axes[0].set_title("Severity distribution")
    axes[0].set_xlabel("")

    year_sev = df.groupby(["year", "severity"]).size().unstack(fill_value=0)
    year_sev[order].plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Severity by year")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_severity.png", dpi=150)
    plt.close()
    print("Saved eda_severity.png")


def plot_cvss_dist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["cvss_score"], bins=40, edgecolor="black")
    axes[0].set_title("CVSS score distribution")
    axes[0].set_xlabel("CVSS base score")

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        subset = df[df["severity"] == sev]["cvss_score"]
        axes[1].hist(subset, bins=20, alpha=0.6, label=sev)
    axes[1].set_title("CVSS score by severity")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_cvss.png", dpi=150)
    plt.close()
    print("Saved eda_cvss.png")


def plot_cwe_dist(df: pd.DataFrame, top_n: int = 20) -> None:
    top_cwes = df["cwe"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_cwes.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} CWEs")
    ax.set_xlabel("Count")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_cwe.png", dpi=150)
    plt.close()
    print("Saved eda_cwe.png")


def plot_description_lengths(df: pd.DataFrame) -> None:
    df = df.copy()
    df["desc_len"] = df["description"].str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["desc_len"].clip(upper=2000), bins=50, edgecolor="black")
    ax.set_title("CVE description length (chars)")
    ax.set_xlabel("Characters")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_desc_length.png", dpi=150)
    plt.close()
    print("Saved eda_desc_length.png")


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"Total records:      {len(df):,}")
    print(f"Years:              {sorted(df['year'].unique())}")
    print(f"Unique CWEs:        {df['cwe'].nunique():,}")
    print(f"CVSS score range:   {df['cvss_score'].min():.1f} – {df['cvss_score'].max():.1f}")
    print(f"Avg desc length:    {df['description'].str.len().mean():.0f} chars")
    print(f"\nSeverity distribution:")
    print(df["severity"].value_counts().to_string())
    print(f"\nTop 10 CWEs:")
    print(df["cwe"].value_counts().head(10).to_string())
    print(f"\nCVSS vector components:")
    for col in ["attack_vector", "attack_complexity", "privileges_required",
                "user_interaction", "scope"]:
        print(f"  {col}: {dict(df[col].value_counts())}")


if __name__ == "__main__":
    df = load_all([2022, 2023, 2024])
    df["year"] = df["cve_id"].str.extract(r"CVE-(\d{4})-").astype(int)

    print_stats(df)
    plot_severity_dist(df)
    plot_cvss_dist(df)
    plot_cwe_dist(df)
    plot_description_lengths(df)
    print("\nEDA complete. Plots saved to results/")
