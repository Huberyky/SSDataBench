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


# ---------- Bootstrap Tests ----------
def bootstrap_categorical_insignificance(r, s, B=1000, alpha=0.05, id_col="profile_id",
                                         sample_n=None, ratio=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r_df = r.copy().dropna()
    s_df = s.copy().dropna()

    not_sig = 0
    r_map = r_df.set_index(id_col).iloc[:, -1].dropna()
    s_map = s_df.set_index(id_col).iloc[:, -1].dropna()
    # print(len(r_map), len(s_map))
    for _ in range(B):
        rb = r_map.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    ).values
        sb = s_map.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    ).values
        # print(len(rb), len(sb))
        cats = sorted(set(rb) | set(sb))
        obs = np.array([[np.sum(rb == c), np.sum(sb == c)] for c in cats])

        _, p, _, _ = chi2_contingency(obs.T,  correction=False)
        if p > alpha:
            not_sig += 1
    # print(p)
    return not_sig / B if B > 0 else np.nan


def bootstrap_numeric_insignificance(r, s, B=1000, alpha=0.05, id_col="profile_id",
                                     sample_n=None, ratio=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r_df = r.copy().dropna()
    s_df = s.copy().dropna()

    not_sig = 0
    r_map = r_df.set_index(id_col).iloc[:, -1].dropna()
    s_map = s_df.set_index(id_col).iloc[:, -1].dropna()

    not_sig = 0
    # print(len(r_map), len(s_map))
    for _ in range(B):
        rb = r_map.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    ).values
        sb = s_map.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    ).values
        # print(len(rb), len(sb))
        _, p_ks = ks_2samp(rb, sb)
        # print(p_ks)
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
