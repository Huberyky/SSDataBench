#!/usr/bin/env python3
"""
Strength vs. structure comparison for Types 2, 3, and 5.
Reads summary_type*.csv files and plots mean ± std of pass rates across datasets.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASETS = ["acs_1980", "cfps", "nlsy"]


def normalize_model_name(model_dir_name: str) -> str:
    base = model_dir_name.split("_")[0]
    parts = base.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else base


def safe_load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _extract_strength_structure(df: pd.DataFrame) -> Dict[str, float]:
    """
    Return dict with strength/structure rates if present.
    Falls back to averaging all rows with mode=strength/structure when
    explicit summary rows are missing.
    """
    cols = {c.lower(): c for c in df.columns}
    rate_col = cols.get("insignificant_rate")
    key_col = cols.get("key")
    mode_col = cols.get("mode")
    if not rate_col:
        return {}

    def _pull_avg(row: pd.DataFrame) -> float:
        if row.empty:
            return np.nan
        # If this is a bundle of rows (no explicit summary), average them.
        vals = pd.to_numeric(row[rate_col], errors="coerce").dropna()
        if vals.empty:
            return np.nan
        return float(vals.mean())

    row_strength = pd.DataFrame()
    row_structure = pd.DataFrame()
    if key_col:
        row_strength = df[df[key_col] == "avg_strength"]
        row_structure = df[df[key_col] == "avg_structure"]
    if mode_col and row_strength.empty:
        row_strength = df[df[mode_col].astype(str).str.lower() == "strength"]
    if mode_col and row_structure.empty:
        row_structure = df[df[mode_col].astype(str).str.lower() == "structure"]

    strength_val = _pull_avg(row_strength)
    structure_val = _pull_avg(row_structure)
    if np.isnan(strength_val) and np.isnan(structure_val):
        return {}
    return {"strength": strength_val, "structure": structure_val}


def process_type(type_id: int, datasets: List[str], root_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []

    for ds in datasets:
        ds_dir = root_dir / ds
        if not ds_dir.is_dir():
            continue

        # Direct summary inside dataset folder (single-run mode)
        direct_summary = ds_dir / f"summary_type{type_id}.csv"
        if direct_summary.exists():
            df = safe_load(direct_summary)
            rates = _extract_strength_structure(df)
            if rates:
                records.append(
                    {
                        "model": ds,
                        "dataset": ds,
                        "strength": rates["strength"],
                        "structure": rates["structure"],
                    }
                )

        for model_dir_name in sorted(p.name for p in ds_dir.iterdir() if p.is_dir()):
            df = safe_load(ds_dir / model_dir_name / f"summary_type{type_id}.csv")
            if df.empty:
                continue
            rates = _extract_strength_structure(df)
            if not rates:
                continue
            records.append(
                {
                    "model": normalize_model_name(model_dir_name),
                    "dataset": ds,
                    "strength": rates["strength"],
                    "structure": rates["structure"],
                }
            )
    return pd.DataFrame(records)


def plot(df: pd.DataFrame, type_id: int, out_dir: Path):
    if df.empty:
        print(f"No data found for type {type_id}")
        return

    models = sorted(df["model"].unique())
    plot_records: List[Dict] = []
    for model in models:
        s_vals = [v for v in df[df["model"] == model]["strength"] if not np.isnan(v)]
        t_vals = [v for v in df[df["model"] == model]["structure"] if not np.isnan(v)]
        s_mean = np.mean(s_vals) if s_vals else np.nan
        t_mean = np.mean(t_vals) if t_vals else np.nan
        s_std = np.std(s_vals) if s_vals else np.nan
        t_std = np.std(t_vals) if t_vals else np.nan
        plot_records.append(
            {
                "model": model,
                "strength_mean": s_mean,
                "strength_std": s_std,
                "structure_mean": t_mean,
                "structure_std": t_std,
            }
        )
    df_plot = pd.DataFrame(plot_records).sort_values("model").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(df_plot))
    offset = 0.15
    strength_color = "#ED9B82"
    structure_color = "#9C89B8"

    plt.errorbar(
        x - offset,
        df_plot["strength_mean"],
        yerr=df_plot["strength_std"],
        fmt="o",
        markersize=8,
        capsize=4,
        color=strength_color,
        label="Strength",
    )
    plt.errorbar(
        x + offset,
        df_plot["structure_mean"],
        yerr=df_plot["structure_std"],
        fmt="s",
        markersize=8,
        capsize=4,
        color=structure_color,
        label="Structure",
    )

    plt.xticks(x, df_plot["model"], rotation=45, ha="right")
    plt.ylabel("Pass Rate", fontsize=14)
    plt.title(f"Type {type_id}: Association Strength vs Structure (Mean ± Std)", fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ss_type{type_id}.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_dir / f"ss_type{type_id}.pdf", dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Strength vs. structure plots for types 2/3/5.")
    parser.add_argument("--root", type=Path, default=Path("evaluation_results"), help="Root directory of evaluation outputs.")
    parser.add_argument("--datasets", type=str, default=os.getenv("VIS_DATASETS", ""), help="Comma-separated dataset names to include.")
    parser.add_argument("--single", action="store_true", help="Treat root as a single run folder (dataset name = folder name).")
    args = parser.parse_args()

    if args.single:
        datasets = [args.root.name]
        base_root = args.root.parent
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else DEFAULT_DATASETS
        base_root = args.root

    out_dir = Path("visualization_figures")
    for t in (2, 3, 5):
        df = process_type(t, datasets, base_root)
        plot(df, t, out_dir)


if __name__ == "__main__":
    main()
