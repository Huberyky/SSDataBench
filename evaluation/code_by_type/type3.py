#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type3.py

Bootstrap significance comparison of regression models (REAL vs. SIMULATED)
via bootstrap significance tests.
Supports: ols
"""

import os, math, argparse, json, io, contextlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import norm

from .common import read_config

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
def _check_real_model_success(df_real, y, Xs, model_type):
    """Return True if the regression can be reliably fit on the real dataset."""
    formula = f"{y} ~ {' + '.join(Xs)}"
    try:
        m = _fit_model(df_real.dropna(subset=[y] + Xs), formula, model_type)
        if m is None:
            return False
        if not hasattr(m, "params") or len(m.params) < 2:
            return False
        if np.all(pd.isna(m.params)):
            return False
        if hasattr(m, "df_resid") and m.df_resid <= 0:
            return False

        r2 = None
        if hasattr(m, "rsquared"):
            r2 = m.rsquared
            print(f"[✅ {y}] R² = {r2:.3f}")
        elif hasattr(m, "prsquared"):
            r2 = m.prsquared
            print(f"[✅ {y}] pseudo-R² = {r2:.3f}")

        if r2 is None or np.isnan(r2):
            return False

        return True

    except Exception as e:
        print(f"[⚠️ Skip {y}: Real model check failed → {e}]")
        return False
# ================= Cleaning =================
def _clean_series(s, cfg):
    s = s.copy()
    s = s.astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
    if "value_map" in cfg and isinstance(cfg["value_map"], dict):
        lower_map = {str(k).lower(): v for k, v in cfg["value_map"].items()}
        s = s.map(lambda x: lower_map.get(x.lower(), x) if isinstance(x, str) else x)
    if "drop_values" in cfg and cfg["drop_values"]:
        drops = set(str(d).strip() for d in cfg["drop_values"])
        s = s.mask(s.astype(str).isin(drops))
    if cfg.get("type","").lower()=="numeric":
        s = pd.to_numeric(s, errors="coerce")
    elif cfg.get("type","").lower()=="categorical":
        s = s.astype("category")
    if cfg.get("log_transform", False) is True:
        s = np.where(s >= 0, np.log1p(s), np.nan)
    return s


def _prepare_predictors(df_real, df_sim, pred_cfg):
    for v, spec in pred_cfg.items():
        if v in df_real.columns: df_real[v]=_clean_series(df_real[v],spec)
        if v in df_sim.columns:  df_sim[v]=_clean_series(df_sim[v],spec)
    return df_real, df_sim, list(pred_cfg.keys())


def _prepare_responses(df_real, df_sim, resp_cfg):
    clean_map={}
    for y,spec in resp_cfg.items():
        if y in df_real.columns: df_real[y]=_clean_series(df_real[y],spec)
        if y in df_sim.columns:  df_sim[y]=_clean_series(df_sim[y],spec)
        clean_map[y]=spec
    return df_real, df_sim, list(resp_cfg.keys()), clean_map


# ================= Model helpers =================
def _fit_model(df, formula, model_type):
    """
    Fit regression model (OLS / Logit / MNLogit) with automatic
    categorical encoding for non-numeric Y.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mt = model_type.lower()
        df = df.copy()
        if mt == "ols":
            return smf.ols(formula, data=df).fit()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")


def _bootstrap_regression_significance(
    df_real, df_sim, model_type, y, Xs, B, alpha, sample_n, ratio,
    id_col="profile_id", out_dir="./Compare_Type3", mode="strength"
):
    """Bootstrap regression comparison under strength mode."""
    if id_col not in df_real.columns or id_col not in df_sim.columns:
        raise ValueError(f"❌ Both df_real and df_sim must contain '{id_col}' for matched bootstrap.")

    common_ids = sorted(set(df_real[id_col]) & set(df_sim[id_col]))
    if len(common_ids) < 5:
        return {
            "response": y, "model_type": model_type, "mode": mode,
            "insignificant_rate": np.nan, "iterations": 0, "pass_count": 0,
            "fit_success": 0, "fit_fail": 0
        }

    n = len(common_ids)
    n_b = int(sample_n or round(n * ratio))
    rng = np.random.default_rng()
    points, beta_logs = [], []
    success_count, fail_count = 0, 0
    formula = f"{y} ~ {' + '.join(Xs)}"

    # === Clean data ===
    def _clean_for_bootstrap(df):
        df = df.copy()
        df[y] = pd.to_numeric(df[y], errors="coerce")

        for x in Xs:
            if x in df.columns:
                if pd.api.types.is_numeric_dtype(df[x]):
                    df[x] = pd.to_numeric(df[x], errors="coerce")
                else:
                    df[x] = df[x].astype(str).replace({"": np.nan, "nan": np.nan, "NA": np.nan}).astype("category")
        return df

    df_real = _clean_for_bootstrap(df_real)
    df_sim = _clean_for_bootstrap(df_sim)

    from statsmodels.stats.anova import anova_lm

    # === Bootstrap iterations ===
    for b in range(B):

        rb = df_real.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    )
        sb = df_sim.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
        )
        # print(len(rb), len(sb))
        rb["__group__"] = "Real"
        sb["__group__"] = "Sim"
        df_comb = pd.concat([rb, sb], ignore_index=True)

        drop_cols = [y] + Xs
        df_comb = df_comb.dropna(subset=drop_cols)
        if len(df_comb) < len(Xs) * 2 + 2:
            points.append(0)
            continue

        try:
            mr = smf.ols(formula, data=rb).fit()
            ms = smf.ols(formula, data=sb).fit()
            
            R2_real = mr.rsquared
            R2_sim  = ms.rsquared
            n1 = len(rb)
            n2 = len(sb)
            
            se1 = np.sqrt(4 * R2_real * (1 - R2_real)**2 / n1)
            se2 = np.sqrt(4 * R2_sim * (1 - R2_sim)**2 / n2)
            # print(R2_real , R2_sim, se1, se2,p)
            delta = R2_real - R2_sim
            se_delta = np.sqrt(se1**2 + se2**2)
            z = delta / se_delta
            p = 2 * norm.sf(abs(z))  
            success_count += 1


        except Exception as e:
            fail_count += 1
            p = np.nan

        if not np.isnan(p) and p > alpha:
            points.append(1)
        else:
            points.append(0)

    # === Save β logs ===
    # os.makedirs(out_dir, exist_ok=True)
    # betafile = os.path.join(out_dir, f"betas_{y}_{mode}.jsonl")
    # with open(betafile, "w", encoding="utf-8") as f:
    #     for row in beta_logs:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "response": y,
        "model_type": model_type,
        "mode": mode,
        "insignificant_rate": np.mean(points) if points else np.nan,
        "iterations": len(points),
        "pass_count": sum(points),
        "fit_success": success_count,
        "fit_fail": fail_count
    }

# ---------- Main ----------
def run_type3_eval(
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
    Type3 evaluation entrypoint.
    Priority of parameter sources (highest to lowest):
      1. Function arguments
      2. Config file (e.g., type3.yaml)
      3. Built-in defaults
    """
    cfg = read_config(config) if isinstance(config, str) else dict(config)

    cfg["real_csv"] = real_csv or cfg.get("real_csv")
    cfg["sim_csv"] = sim_csv or cfg.get("sim_csv")
    cfg["out_dir"] = out_dir or cfg.get("out_dir", "./results_type3")
    cfg["bootstrap_B"] = bootstrap_B or cfg.get("bootstrap_B", 10)
    cfg["bootstrap_sample_n"] = bootstrap_sample_n or cfg.get("bootstrap_sample_n", None)
    cfg["alpha"] = alpha or cfg.get("alpha", 0.05)

    df_real = pd.read_csv(cfg["real_csv"], low_memory=False)
    df_sim  = pd.read_csv(cfg["sim_csv"],  low_memory=False)
    out_dir = Path(cfg.get("out_dir", "./Compare_Type3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    B = cfg.get("bootstrap_B", 1000)
    a = cfg.get("alpha", 0.05)
    ratio = cfg.get("bootstrap_sample_ratio", 1.0)
    sample_n = cfg.get("bootstrap_sample_n", None)

    df_real, df_sim, Xs = _prepare_predictors(df_real, df_sim, cfg["predictors"])
    df_real, df_sim, responses, _ = _prepare_responses(df_real, df_sim, cfg["response"])
    mt_cfg = cfg.get("model_type", "ols")

    if verbose:
        print(f"\n================ TYPE 3 COMPARISON ================")
        print(f"Bootstrap {B} iterations | α={a}\n")

    if isinstance(mt_cfg, dict):
        pairs = [(y, mt_cfg.get(y, "ols")) for y in responses]
    elif isinstance(mt_cfg, list):
        if len(mt_cfg) != len(responses):
            raise ValueError(f"Length mismatch: {len(mt_cfg)} model types but {len(responses)} responses")
        pairs = list(zip(responses, mt_cfg))
    else:
        pairs = [(y, mt_cfg) for y in responses]

    success_outcomes = []
    print("🔍 Checking real-model fit success...")
    for y, mt in pairs:
        ok = _check_real_model_success(df_real, y, Xs, mt)
        if ok:
            success_outcomes.append((y, mt))
            print(f"✅ {y} ({mt}) — Real model fits successfully")
        else:
            print(f"⚠️ Skipped {y} ({mt}) — Real model failed to fit")

    if not success_outcomes:
        print("❌ No valid real-model fits. Exiting.")
        return

    print(f"\n------ Evaluating STRENGTH mode ------")
    results = []
    for y, mt in success_outcomes:
        res = _bootstrap_regression_significance(
            df_real, df_sim, mt, y, Xs,
            B, a, sample_n, ratio,
            mode="strength"
        )
        results.append(res)
        if verbose:
            print(f"{y} ({mt}, strength) → insignificant_rate={res['insignificant_rate']:.3f}")

    summary_all = pd.DataFrame(results)
    summary_all["mode"] = "strength"

    summary_strength_path = os.path.join(out_dir, "summary_type3_strength.csv")
    summary_all.to_csv(summary_strength_path, index=False)
    avg_strength = np.nanmean(summary_all["insignificant_rate"])
    print(f"✅ Saved strength summary → {summary_strength_path}")
    print(f"   Average Insignificant Rate (strength) = {avg_strength:.3f}")

    print("\n📊 Aggregating strength summary...")
    avg_rows = []
    if not summary_all.empty:
        avg_row = {c: "" for c in summary_all.columns}
        avg_row["key"] = "avg_strength"
        avg_row["mode"] = "strength"
        avg_row["insignificant_rate"] = avg_strength
        avg_rows.append(avg_row)

        overall_row = {c: "" for c in summary_all.columns}
        overall_row["key"] = "avg_all"
        overall_row["mode"] = "overall"
        overall_row["insignificant_rate"] = avg_strength
        avg_rows.append(overall_row)

    summary_all_out = pd.concat([summary_all, pd.DataFrame(avg_rows)], ignore_index=True) if avg_rows else summary_all

    summary_all_path = os.path.join(out_dir, "summary_type3.csv")
    summary_all_out.to_csv(summary_all_path, index=False)

    print(f"\n================ SUMMARY =================")
    print(f"Average (Strength)  = {avg_strength:.3f}")
    print(f"Overall Average     = {avg_strength:.3f}")
    print(f"📄 Summary saved to {summary_all_path}\n")

    avg_total = avg_strength
    return {
        "avg_strength": avg_strength,
        "avg_insignificant_rate": avg_total,
        "summary_path": summary_all_path,
        "summary_df": summary_all
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap_B", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    run_type3_eval(args.config, bootstrap_B=args.bootstrap_B, alpha=args.alpha)
