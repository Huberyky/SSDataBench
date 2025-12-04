#!/usr/bin/env python3
"""
Overview plots for evaluation results:
 - Heatmap of average insignificant rates (model × type × dataset)
 - Box/strip plots for per-type metrics (types 1–5)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

EXCLUDED_MODELS = {"qwen-qwen2.5-vl"}

PALETTE_REAL = "#2C5AA0"
PALETTE_SIM = "#C65E1A"
BOX_REAL = "#A8C0E0"
BOX_SIM = "#F1BEA7"


def normalize_model_name(model_dir_name: str) -> str:
    base = model_dir_name.split("_")[0]
    parts = base.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else base


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect_summary_records(root: Path, single: bool) -> Tuple[pd.DataFrame, List[str]]:
    records: List[Dict] = []
    datasets = []
    dataset_paths = [root] if single else [p for p in root.iterdir() if p.is_dir()]

    for dpath in sorted(dataset_paths, key=lambda p: p.name):
        dataset = dpath.name
        datasets.append(dataset)

        if single:
            # Treat this folder as a single run with summary at its root.
            summary_path = dpath / "overall_insignificant_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                model = dataset
                if model not in EXCLUDED_MODELS:
                    for _, r in df.iterrows():
                        t = str(r["type"])
                        if t.startswith("type"):
                            records.append(
                                {
                                    "dataset": dataset,
                                    "model": model,
                                    "type": t,
                                    "avg_insignificant_rate": float(r["avg_insignificant_rate"]),
                                }
                            )
            continue

        # Model subfolders
        for model_dir in sorted(p.name for p in dpath.iterdir() if p.is_dir()):
            csv_path = dpath / model_dir / "overall_insignificant_summary.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            model = normalize_model_name(model_dir)
            if model in EXCLUDED_MODELS:
                continue
            for _, r in df.iterrows():
                t = str(r["type"])
                if t.startswith("type"):
                    records.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "type": t,
                            "avg_insignificant_rate": float(r["avg_insignificant_rate"]),
                        }
                    )
    return pd.DataFrame(records), datasets


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> Tuple[List[str], List[str]]:
    piv = df.pivot_table(index="model", columns=["type", "dataset"], values="avg_insignificant_rate")
    types = [f"type{i}" for i in range(1, 6) if f"type{i}" in piv.columns.get_level_values(0)]
    datasets = sorted(df["dataset"].unique())
    cols = [(t, d) for t in types for d in datasets if (t, d) in piv.columns]
    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(cols))
    piv["__avg__"] = piv.mean(axis=1)
    piv = piv.sort_values("__avg__", ascending=True)
    piv_heat = piv.drop(columns="__avg__")

    n = len(piv_heat)
    fig = plt.figure(figsize=(16, max(4, n * 0.6)))
    gs = GridSpec(1, 2, width_ratios=[2.3, 8.5], wspace=0.02, left=0.10, right=0.98, bottom=0.25, top=0.90)
    ax_labels = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1], sharey=ax_labels)

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        piv_heat,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": "Average Pass Rate"},
        yticklabels=False,
        ax=ax_heat,
    )

    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title("Average Pass Rate by Model × Type × Dataset", fontweight="bold")

    ax_heat.set_xticks(np.arange(len(piv_heat.columns)) + 0.5)
    ax_heat.set_xticklabels([c[1] for c in piv_heat.columns], rotation=45, ha="right")

    for t in types:
        idx = [i for i, c in enumerate(piv_heat.columns) if c[0] == t]
        if idx:
            mid = (min(idx) + max(idx) + 1) / 2
            ax_heat.text(
                mid,
                -0.28,
                t,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                transform=ax_heat.get_xaxis_transform(),
            )
    for t in types[:-1]:
        idx = [i for i, c in enumerate(piv_heat.columns) if c[0] == t]
        if idx:
            ax_heat.axvline(max(idx) + 1, color="gray", lw=1.5, ls="--")

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(ax_heat.get_ylim())
    ax_labels.invert_yaxis()
    ax_labels.axis("off")

    ys = np.arange(n) + 0.5
    models = piv.index.tolist()
    avgs = piv["__avg__"].tolist()

    # Compute left offset for the (Avg) label aligned to the heatmap
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_labels = ax_labels.get_window_extent(renderer=renderer)
    bbox_heat = ax_heat.get_window_extent(renderer=renderer)
    heat_left_rel = (bbox_heat.x0 - bbox_labels.x0) / bbox_labels.width
    avg_x = heat_left_rel - 0.01
    ax_labels.axvline(heat_left_rel - 0.005, color="lightgray", lw=0.8)

    for y, model, avg in zip(ys, models, avgs):
        ax_labels.text(0.00, y, model, ha="left", va="center", fontsize=11, color="black", clip_on=False)
        ax_labels.text(
            avg_x,
            y,
            f"(Avg: {avg:.2f})",
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "heatmap_auto_align_labels.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "heatmap_auto_align_labels.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_dir}")

    return models, datasets


def collect_metric_records(
    root: Path,
    datasets: Iterable[str],
    models_order: List[str],
    rel_path: str,
    real_col: str,
    sim_col: str,
) -> pd.DataFrame:
    records: List[Dict] = []

    def _pick_cols(df: pd.DataFrame, real_key: str, sim_key: str) -> Tuple[str, str]:
        if real_key in df.columns and sim_key in df.columns:
            return real_key, sim_key
        real_alt = next((c for c in df.columns if "real" in c.lower()), "")
        sim_alt = next((c for c in df.columns if "sim" in c.lower()), "")
        if real_alt and sim_alt:
            return real_alt, sim_alt
        return "", ""

    for ds in datasets:
        ds_dir = root / ds
        if not ds_dir.is_dir():
            continue

        # Case: metrics saved directly under dataset folder (single-run mode)
        direct_path = ds_dir / rel_path
        if direct_path.exists():
            df = safe_read_csv(direct_path)
            if not df.empty:
                rcol, scol = _pick_cols(df, real_col, sim_col)
                if rcol and scol:
                    for v in df[rcol].dropna():
                        records.append({"model": ds, "type": "Real", "value": v})
                    for v in df[scol].dropna():
                        records.append({"model": ds, "type": "Simulated", "value": v})

        # Case: metrics inside model subfolders
        for model_dir_name in sorted(p.name for p in ds_dir.iterdir() if p.is_dir()):
            model = normalize_model_name(model_dir_name)
            csv_path = ds_dir / model_dir_name / rel_path
            df = safe_read_csv(csv_path)
            if df.empty:
                continue
            rcol, scol = _pick_cols(df, real_col, sim_col)
            if not rcol or not scol:
                continue
            for v in df[rcol].dropna():
                records.append({"model": model, "type": "Real", "value": v})
            for v in df[scol].dropna():
                records.append({"model": ds, "type": "Real", "value": v})
            for v in df[scol].dropna():
                records.append({"model": model, "type": "Simulated", "value": v})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["model"] = pd.Categorical(df["model"], categories=models_order, ordered=True)
    return df.sort_values("model")


def plot_boxstrip(df: pd.DataFrame, title: str, ylabel: str, out_path: Path):
    if df.empty:
        print(f"Skipping {title}: no data")
        return
    plt.figure(figsize=(17, 6))
    sns.boxplot(
        data=df,
        x="model",
        y="value",
        hue="type",
        hue_order=["Real", "Simulated"],
        showcaps=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        linewidth=1.3,
        palette={"Real": BOX_REAL, "Simulated": BOX_SIM},
        dodge=True,
    )
    sns.stripplot(
        data=df,
        x="model",
        y="value",
        hue="type",
        hue_order=["Real", "Simulated"],
        dodge=True,
        alpha=0.7,
        size=3.5,
        palette={"Real": PALETTE_REAL, "Simulated": PALETTE_SIM},
    )
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(-0.02, max(1.02, df["value"].max() + 0.1))
    plt.legend([], [], frameon=False)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_REAL, markersize=9, label="Real"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_SIM, markersize=9, label="Simulated"),
        ],
        fontsize=14,
        loc="upper right",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close()
    print(f"Saved {title} → {out_path.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(description="Overview visualizations for evaluation results.")
    parser.add_argument("--root", type=Path, default=Path("evaluation_results"), help="Root directory of evaluation outputs.")
    parser.add_argument("--single", action="store_true", help="Treat root as a single run folder (no subfolders).")
    args = parser.parse_args()

    sns.set(style="whitegrid", font_scale=1.2)
    df_summary, datasets = collect_summary_records(args.root, args.single)
    if df_summary.empty:
        raise ValueError(f"No valid summary files found under {args.root}")

    out_dir = Path("visualization_figures")
    models_order, datasets_in_summary = plot_heatmap(df_summary, out_dir)

    # In single mode, dataset folder holds files directly; base_root should be its parent.
    base_root = args.root if not args.single else args.root.parent

    # Type 2: Cramer's V pairs
    df_type2 = collect_metric_records(
        base_root,
        datasets_in_summary,
        models_order,
        "Data_type2/cramers_v_pairs.csv",
        real_col="real_v",
        sim_col="sim_v",
    )
    plot_boxstrip(df_type2, "Type 2: Cramer's V (paired associations)", "Cramer's V", out_dir / "type2")

    # Type 1: entropy
    df_type1 = collect_metric_records(
        base_root, datasets_in_summary, models_order, "Data_type1/entropy.csv", real_col="real", sim_col="sim"
    )
    plot_boxstrip(df_type1, "Type 1: Entropy distributions", "Entropy", out_dir / "type1")

    # Type 3: regression R²
    df_type3 = collect_metric_records(
        base_root,
        datasets_in_summary,
        models_order,
        "Data_type3/regression_r2.csv",
        real_col="r2_real",
        sim_col="r2_sim",
    )
    plot_boxstrip(df_type3, "Type 3: Regression R² distributions", "R²", out_dir / "type3")

    # Type 4: event order entropy
    df_type4 = collect_metric_records(
        base_root,
        datasets_in_summary,
        models_order,
        "Data_type4/event_order_entropy.csv",
        real_col="real_entropy",
        sim_col="sim_entropy",
    )
    plot_boxstrip(df_type4, "Type 4: Event order entropy", "Entropy", out_dir / "type4")

    # Type 5: Cramer's V for event sequences
    df_type5 = collect_metric_records(
        base_root,
        datasets_in_summary,
        models_order,
        "Data_type5/CramersV_pairs_all.csv",
        real_col="real",
        sim_col="sim",
    )
    plot_boxstrip(df_type5, "Type 5: Cramer's V (event sequences vs predictors)", "Cramer's V", out_dir / "type5")


if __name__ == "__main__":
    main()
