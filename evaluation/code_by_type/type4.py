#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type4.py

Compare REAL vs. SIMULATED datasets in terms of the
relative ORDER of key life-course events
via bootstrap significance tests.

"""

import os, argparse, json, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import entropy

from .common import read_config

def _entropy(series):
    v = series.dropna().value_counts(normalize=True)
    return entropy(v, base=2)


def _compute_order_label(row, event_vars, name_map=None):
    """Build a human-readable order tag (e.g., Edu→Marriage→Birth) from event timings."""
    events = []
    for v in event_vars:
        val = row[v]
        if pd.isna(val):
            continue
        events.append((v, float(val)))
    if not events:
        return np.nan

    ordered = sorted(events, key=lambda x: x[1])

    labeled = [name_map.get(v, v) for v, _ in ordered] if name_map else [v for v, _ in ordered]

    return "→".join(labeled)

def _clean_series(s, cfg):
    s = s.astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
    if "value_map" in cfg:
        for k, v in cfg["value_map"].items():
            s = s.replace(k, v)
    s = s.replace(["inf", "Inf", "INF"], np.inf)
    return pd.to_numeric(s, errors="coerce")


# ---------- Core Bootstrap ----------
def _bootstrap_order_consistency(
    df_real, df_sim, event_vars,
    B=1000, alpha=0.05,
    sample_n=None, ratio=1.0,
    id_col="profile_id", verbose=True,
    out_dir=None
):
    """
    Matched bootstrap test on event order patterns between REAL and SIM datasets.

    Returns:
        - insignificant_rate
        - mean_p
        - mean_dissimilarity (Index of Dissimilarity)
    """
    if id_col not in df_real.columns or id_col not in df_sim.columns:
        raise ValueError(f"❌ Both datasets must contain '{id_col}' for matched bootstrap.")
    df_real = df_real.copy().dropna(subset=event_vars, how="any")
    df_sim = df_sim.copy().dropna(subset=event_vars, how="any")

    if verbose:
        print(f"\n================ TYPE 4 COMPARISON ================")
        print(f"Bootstrap {B} iterations | α={alpha}")
        print(f"Event variables: {event_vars}")

    rng = np.random.default_rng()
    pvals, dissim_vals = [], []
    example_tbl = None 

    for b in range(B):
        try:
            rb = df_real.sample(
            n=sample_n, replace=True,
            random_state=rng.integers(1e9)
        )
            sb = df_sim.sample(
            n=sample_n, replace=True,
            random_state=rng.integers(1e9)
        )
            # print(rb, sb)

            all_orders = sorted(set(rb["event_order"]) | set(sb["event_order"]))
            # print(all_orders)
            if not all_orders:
                continue

            tbl = pd.DataFrame(0, index=["real", "sim"], columns=all_orders)
            tbl.loc["real"] = rb["event_order"].value_counts().reindex(all_orders, fill_value=0)
            tbl.loc["sim"]  = sb["event_order"].value_counts().reindex(all_orders, fill_value=0)
            # print(tbl)
            p_real = tbl.loc["real"] / tbl.loc["real"].sum()
            p_sim  = tbl.loc["sim"]  / tbl.loc["sim"].sum()

            D = 0.5 * np.sum(np.abs(p_real - p_sim))
            dissim_vals.append(D)
            chi2, p, _, _ = chi2_contingency(tbl, correction=False)
            # print(p)
            if np.isnan(p):
                pvals.append(0)   # NaN 视为 fail
            else:
                pvals.append(p)

            if example_tbl is None:
                example_tbl = tbl.copy()
        except Exception:
            continue


    if not pvals:
        return {
            "insignificant_rate": np.nan,
            "mean_p": np.nan,
            "mean_dissimilarity": np.nan,
            "pass_count": 0,
            "iterations": 0,
            "comment": "No valid bootstrap samples."
        }
    non_sig = np.sum(np.array(pvals) >= alpha)
    insignificant_rate = non_sig / len(pvals)

    return {
        "insignificant_rate": insignificant_rate,
        "mean_p": np.mean(pvals),
        "mean_dissimilarity": np.mean(dissim_vals) if dissim_vals else np.nan,
        "pass_count": int(non_sig),
        "iterations": len(pvals)
    }

# ---------- Main ----------
def run_type4_eval(
    config,
    real_csv=None,
    sim_csv=None,
    out_dir=None,
    bootstrap_B=None,
    bootstrap_sample_n=None,
    alpha=None,
    verbose=True,
):
    """
    Type4 evaluation entrypoint.
    Priority of parameter sources (highest to lowest):
      1. Function arguments
      2. Config file (type4.yaml)
      3. Built-in defaults
    """
    cfg = read_config(config) if isinstance(config, str) else dict(config)

    cfg["real_csv"] = real_csv or cfg.get("real_csv")
    cfg["sim_csv"] = sim_csv or cfg.get("sim_csv")
    cfg["out_dir"] = out_dir or cfg.get("out_dir", "./results_type4")
    cfg["bootstrap_B"] = bootstrap_B or cfg.get("bootstrap_B", 10)
    cfg["bootstrap_sample_n"] = bootstrap_sample_n or cfg.get("bootstrap_sample_n", None)
    cfg["alpha"] = alpha or cfg.get("alpha", 0.05)

    real = pd.read_csv(cfg["real_csv"], low_memory=False)
    sim  = pd.read_csv(cfg["sim_csv"],  low_memory=False)
    out_dir = cfg.get("out_dir", "./Compare_Type4")
    os.makedirs(out_dir, exist_ok=True)

    event_cfgs = cfg["event_variables"]
    event_vars = list(event_cfgs.keys())
    all_combos = []
    for k in (3,):
        all_combos += list(itertools.combinations(event_vars, k))
    B = cfg.get("bootstrap_B", 1000)
    a = cfg.get("alpha", 0.05)
    ratio = cfg.get("bootstrap_sample_ratio", 1.0)
    sample_n = cfg.get("bootstrap_sample_n", None)

    for v, spec in event_cfgs.items():
        real[v] = _clean_series(real[v], spec)
        sim[v]  = _clean_series(sim[v], spec)

    results = []
    entropy_rows = []
    for combo in all_combos:
        combo = list(combo)
        print(f"\n▶️ Testing combination: {combo}")
        r=real.copy()
        s=sim.copy()
        r=r.dropna(subset=combo,how="any")
        s =s.dropna(subset=combo,how="any")

        r["event_order"]=r.apply(_compute_order_label,axis=1,args=(combo,))
        s["event_order"] =s.apply(_compute_order_label,axis=1,args=(combo,))
        real_entropy = _entropy(r["event_order"])
        sim_entropy  = _entropy(s["event_order"])

        entropy_rows.append({
            "combo": "→".join(combo),
            "real_entropy": real_entropy,
            "sim_entropy": sim_entropy
        }) 
        # print(entropy_rows)
        res = _bootstrap_order_consistency(
            r, s, combo,
            B=B, alpha=a, sample_n=sample_n, ratio=ratio,
            id_col="profile_id", verbose=False, out_dir=out_dir
        )
        print(res)
        # import pdb; pdb.set_trace()
        res["combo"] = "→".join(combo)
        results.append(res)

    save_dir = os.path.join(out_dir, "Data_type4")
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(entropy_rows).to_csv(
        os.path.join(save_dir, "event_order_entropy.csv"),
        index=False
    )

    summary = pd.DataFrame(results)

    if not summary.empty:
        avg_row = summary.select_dtypes(include=[float, int]).mean(numeric_only=True).to_dict()
        avg_row["combo"] = "avg"
        summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

    summary_path = os.path.join(out_dir, "summary_type4.csv")
    summary.to_csv(summary_path, index=False)

    avg_insig = np.nanmean(summary.loc[summary["combo"] != "avg", "insignificant_rate"])
    avg_dissim = np.nanmean(summary.loc[summary["combo"] != "avg", "mean_dissimilarity"])
    print(f"\n================ SUMMARY =================")
    print(f"Average Insignificant Rate = {avg_insig:.3f}")
    print(f"Average Dissimilarity Index = {avg_dissim:.3f}")
    print(f"📄 Summary saved to {summary_path}")

    return {
        "avg_insignificant_rate": avg_insig,
        "summary_path": summary_path,
        "summary_df": summary
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap_B", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    run_type4_eval(args.config, bootstrap_B=args.bootstrap_B, alpha=args.alpha)
