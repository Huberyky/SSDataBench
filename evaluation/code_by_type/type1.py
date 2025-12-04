#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type1.py

Compare real vs. simulated distributions for type-1 variables
with bootstrap-based insignificance frequency.

Now includes pre-bootstrap visualization for each variable:
  - categorical: bar plot of real vs. simulated percentage
  - numeric: KDE plot of real vs. simulated density
"""

import os, json, math, argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

from .common import (
    read_config,
    clean_str,
    apply_value_map,
    drop_values,
    to_numeric_clean,
)


def _entropy(series):
    v = series.dropna().value_counts(normalize=True)
    return entropy(v, base=2)
import matplotlib.colors as mcolors

def _plot_categorical_distribution(var, r, s, out_dir, display_name=None, allowed=None, rate=None):
    """Bar plot comparing categorical distributions."""
    base_dir = Path(out_dir)
    r = r.dropna()
    s = s.dropna()
    if len(r) == 0 or len(s) == 0:
        return

    r_pct = r.value_counts(normalize=True) * 100
    s_pct = s.value_counts(normalize=True) * 100

    if allowed and isinstance(allowed, (list, tuple)) and len(allowed) > 0:
        cats = [c for c in allowed if (c in r_pct.index or c in s_pct.index)]
        cats += [c for c in r_pct.index if c not in cats]
        cats += [c for c in s_pct.index if c not in cats]
    else:
        cats = list(r_pct.index) + [c for c in s_pct.index if c not in r_pct.index]

    comp = pd.DataFrame({
        "category": cats,
        "Real": [r_pct.get(c, 0.0) for c in cats],
        "Simulated": [s_pct.get(c, 0.0) for c in cats],
    })
    comp_melt = comp.melt(id_vars="category", var_name="source", value_name="percent")

    palette = {"Real": "#4C72B0", "Simulated": "#DD8452"}
    hue_order = ["Real", "Simulated"]

    plt.figure(figsize=(7.5, 4))
    ax = sns.barplot(
        data=comp_melt,
        x="category",
        y="percent",
        hue="source",
        hue_order=hue_order,
        order=cats,
        palette=palette,
        saturation=1.0,
        edgecolor="none",
        legend=False,
        errorbar=None,
    )

    bars = [p for p in ax.patches]
    bars = sorted(bars, key=lambda p: (p.get_x(), p.get_y()))

    n_hue = len(hue_order)
    for i, patch in enumerate(bars):
        hue = hue_order[i % n_hue]
        rgb = mcolors.to_rgb(palette[hue])
        patch.set_facecolor((*rgb, 0.45))
        patch.set_edgecolor((*rgb, 1.0))
        patch.set_linewidth(1.6)
    plt.text(
     0.95, 0.75,  f"Pass Rate = {rate:.2f}",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=9, color="gray",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2")
)
    title = display_name or var
    plt.title(f"Distribution Comparison: {title}", fontsize=12, weight='bold')
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xlabel(title, fontsize=11)
    plt.xticks(rotation=40, ha="right")
    # plt.legend(title="", loc="upper right", frameon=False)
    sns.despine()
    plt.tight_layout()

    fig_dir = base_dir / "Figures_type1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{var}_distribution.png", dpi=300)
    plt.savefig(fig_dir / f"{var}_distribution.pdf", dpi=300)
    plt.close()

    print(f"📊 Saved categorical plot: {fig_dir}")

def _plot_numeric_distribution(var, r, s, out_dir, display_name=None, rate=None):
    """KDE comparison for numeric variables."""
    base_dir = Path(out_dir)

    r = r.dropna()
    s = s.dropna()
    # print(len(r), len(s))
    if len(r) == 0 or len(s) == 0:
        return

    real_color = "#4C72B0"
    sim_color = "#DD8452"

    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        r, label="Real",
        color=real_color,
        linewidth=2.0,
        fill=True,
        alpha=0.4,
    )
    sns.kdeplot(
        s, label="Simulated",
        color=sim_color,
        linewidth=2.0,
        fill=True,
        alpha=0.4,
    )
    title = display_name or var
    plt.title(f"Distribution Comparison: {title}", fontsize=12, weight='bold')
    plt.text(
     0.95, 0.75, f"Pass Rate = {rate:.2f}",
    transform=plt.gca().transAxes,
    ha="right", va="top",
    fontsize=9, color="gray",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2")
)
    plt.xlabel(title, fontsize=11)
    plt.ylabel("Density")
    plt.legend(title="", loc="upper right", frameon=False)
    sns.despine()
    plt.tight_layout()
    fig_dir = base_dir / "Figures_type1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{var}_distribution.png", dpi=300)
    plt.savefig(fig_dir / f"{var}_distribution.pdf", dpi=300)
    plt.close()
    print(f"📊 Saved numerical plot: {fig_dir}")


# ---------- Bootstrap Tests ----------
def bootstrap_categorical_insignificance(r, s, B=1000, alpha=0.05, id_col="profile_id",
                                         sample_n=None, ratio=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r_df = r.copy().dropna()
    s_df = s.copy().dropna()

    common_ids = np.intersect1d(r_df[id_col].dropna(), s_df[id_col].dropna())
    if len(common_ids) == 0:
        return np.nan
    n_total = len(common_ids)
    # import pdb; pdb.set_trace()
    # print(n_total)
    n_b = int(sample_n or round(n_total * ratio))
    not_sig = 0
    r_map = r_df.set_index(id_col).iloc[:, -1]
    s_map = s_df.set_index(id_col).iloc[:, -1]
    for _ in range(B):
        sampled_ids = rng.choice(common_ids, n_b, replace=True)
        rb = r_map.loc[sampled_ids].dropna().values
        sb = s_map.loc[sampled_ids].dropna().values
        # print(len(rb), len(sb))
        if len(rb) < 2 or len(sb) < 2:
            continue
        cats = sorted(set(rb) | set(sb))
        obs = np.array([[np.sum(rb == c), np.sum(sb == c)] for c in cats])
        try:
            # print(obs.T)
            _, p, _, _ = chi2_contingency(obs.T)
            if p > alpha:
                not_sig += 1
            # print(chi2, p, dof, expected)
        except Exception:
            continue
    return not_sig / B if B > 0 else np.nan


def bootstrap_numeric_insignificance(r, s, B=1000, alpha=0.05, id_col="profile_id",
                                     sample_n=None, ratio=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r_df = r.copy().dropna()
    s_df = s.copy().dropna()


    common_ids = np.intersect1d(r_df[id_col], s_df[id_col])
    if len(common_ids) == 0:
        return np.nan
    n_total = len(common_ids)
    # print(n_total)
    n_b = int(sample_n or round(n_total * ratio))
    r_map = r_df.set_index(id_col).iloc[:, -1]
    s_map = s_df.set_index(id_col).iloc[:, -1]

    not_sig = 0
    for _ in range(B):
        sampled_ids = rng.choice(common_ids, n_b, replace=True)
        rb = r_map.loc[sampled_ids].dropna().values
        sb = s_map.loc[sampled_ids].dropna().values
        # print(len(rb),len(sb))
        if len(rb) < 3 or len(sb) < 3:
            continue
        _, p_ks = ks_2samp(rb, sb)
        if p_ks > alpha:
            not_sig += 1
    return not_sig / B if B > 0 else np.nan


# ---------- Main ----------
def run_type1_eval(config: Union[str, Dict[str, Any]],
                   real_csv: Optional[str] = None,
                   sim_csv: Optional[str] = None,
                   out_dir: Optional[str] = None,
                   bootstrap_B: Optional[int] = None,
                   bootstrap_sample_n: Optional[int] = None,
                   alpha: Optional[float] = None,
                   verbose: bool = True):
    if isinstance(config, str):
        cfg = read_config(config)
    else:
        cfg = dict(config)

    if real_csv: cfg["real_csv"] = real_csv
    if sim_csv: cfg["sim_csv"] = sim_csv
    if out_dir: cfg["out_dir"] = out_dir
    if bootstrap_B: cfg["bootstrap_B"] = bootstrap_B
    if alpha: cfg["alpha"] = alpha
    if bootstrap_sample_n: cfg["bootstrap_sample_n"] = bootstrap_sample_n

    df_real = pd.read_csv(cfg["real_csv"], low_memory=False)
    df_sim = pd.read_csv(cfg["sim_csv"], low_memory=False)
    out_dir = Path(cfg.get("out_dir", "./Compare_Type1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    B = cfg.get("bootstrap_B", 1000)
    a = cfg.get("alpha", 0.05)
    S = cfg.get("bootstrap_sample_n", 500)
    results = []
    entropy_rows = []
    if verbose:
        print(f"\n================ TYPE 1 COMPARISON ================")
        print(f"Bootstrap {B} iterations | α={a}\n")

    for var, vcfg in cfg["variables"].items():
        vtype = (vcfg.get("type") or "").lower()
        display_name = vcfg.get("name", var)
        print(f"--- {var} ({vtype}) {display_name}---")

        if vtype == "categorical":
            r = df_real[["profile_id", var]].copy()
            s = df_sim[["profile_id", var]].copy()
            r[var] = apply_value_map(r[var].map(clean_str), vcfg.get("value_map", {}))
            s[var] = apply_value_map(s[var].map(clean_str), vcfg.get("value_map", {}))
            r[var] = drop_values(r[var], vcfg.get("drop_values", []))
            s[var] = drop_values(s[var], vcfg.get("drop_values", []))
            # print(len(r), len(s))
            real_entropy = _entropy(r[var])
            sim_entropy  = _entropy(s[var])
            # import pdb; pdb.set_trace()
    
            entropy_rows.append({
            "var": var,
            "real": real_entropy,
            "sim": sim_entropy
        })
            rate = bootstrap_categorical_insignificance(r, s, B=B, alpha=a, id_col="profile_id",
                                         sample_n=S, ratio=1.0, rng=None)
            _plot_categorical_distribution(
            var,
            r[var],
            s[var],
            out_dir,
            display_name,
            allowed=vcfg.get("allowed", None),
            rate=rate
        )

        else:
            r = df_real[["profile_id", var]].copy()
            s = df_sim[["profile_id", var]].copy()
            r[var] = apply_value_map(r[var].map(clean_str), vcfg.get("value_map", {}))
            s[var] = apply_value_map(s[var].map(clean_str), vcfg.get("value_map", {}))
            r[var] = drop_values(r[var], vcfg.get("drop_values", []))
            s[var] = drop_values(s[var], vcfg.get("drop_values", []))
            r[var] = to_numeric_clean(r[var])
            s[var] = to_numeric_clean(s[var])
            # print(len(r), len(s))
            rate = bootstrap_numeric_insignificance(r, s, B=B, alpha=a, id_col="profile_id",
                                         sample_n=S, ratio=1.0, rng=None)
            _plot_numeric_distribution(
            var,
            r[var],
            s[var],
            out_dir,
            display_name,
            rate=rate
        )

        results.append({
            "variable": var,
            "type": vtype,
            "insignificant_rate": rate
        })

        print(f"  Insignificant rate = {rate:.3f}")
        fig_path = out_dir / f"{var}_distribution.png"
        if fig_path.exists():
            print(f"  📊 Plot saved: {fig_path}")
    summary = pd.DataFrame(results)

    if not summary.empty:
        avg_row = summary.select_dtypes(include=[float, int]).mean(numeric_only=True)
        avg_row = avg_row.to_dict()
        avg_row["key"] = "avg"
        for col in summary.columns:
            if col not in avg_row:
                avg_row[col] = ""
        summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)
    out_entropy_dir = out_dir / "Data_type1"
    out_entropy_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(entropy_rows).to_csv(out_entropy_dir / "entropy.csv", index=False)
    summary_path = out_dir / "summary_type1.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path} (with average row).")
    overall = np.nanmean(summary["insignificant_rate"])
    print(f"\n================ SUMMARY =================")
    print(f"Average Insignificant Rate = {overall:.3f}")
    print(f"📄 Summary saved to {summary_path}\n")

    return {"avg_insignificant_rate": overall, "summary_path": summary_path, "summary_df": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap_B", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    run_type1_eval(args.config, bootstrap_B=args.bootstrap_B, alpha=args.alpha)
