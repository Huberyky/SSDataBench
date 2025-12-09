#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type3.py

Bootstrap significance comparison of regression models (REAL vs. SIMULATED).

Each bootstrap iteration fits the same regression model to both datasets and
tests whether all coefficients are statistically equal (H0: β_real = β_sim).

Scoring rule:
  - 1 point if *all* coefficients have p > α (not significant)
  - 0 otherwise

Final similarity_score = mean(points across B iterations)
------------------------------------------------------------
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
def _plot_full_regression_coefficients(
    df_real, df_sim, y, Xs, model_type, out_dir,
    rate_strength=None
):
    """Compare full-data regression coefficients between real and simulated datasets."""
    formula = f"{y} ~ {' + '.join(Xs)}"
    y_series = df_real[y]
    if not pd.api.types.is_numeric_dtype(y_series):
        n_classes = y_series.nunique()
        if n_classes == 2:
            model_type = "logit"
        elif n_classes > 2:
            model_type = "mnlogit"
        else:
            print(f"[⚠️ Skipping {y}: not enough categories]")
            return

    print(f"📈 Fitting full {model_type.upper()} regression for {y} ...")

    try:
        mr = _fit_model(df_real, formula, model_type)
        ms = _fit_model(df_sim,  formula, model_type)
    except Exception as e:
        print(f"[⚠️ Full-data regression failed for {y}: {e}]")
        return

    def _get_r2(model):
        if hasattr(model, "rsquared"):
            return model.rsquared
        elif hasattr(model, "prsquared"):
            return model.prsquared
        else:
            return np.nan

    r2_real = _get_r2(mr)
    r2_sim  = _get_r2(ms)
    print(f"   ↳ Real R² = {r2_real:.3f}, Sim R² = {r2_sim:.3f}")

    pr, ps = _flatten_params(mr), _flatten_params(ms)
    if isinstance(pr.index, pd.MultiIndex):
        pr.index = pr.index.map(lambda x: f"{x[0]}:{x[1]}")
    if isinstance(ps.index, pd.MultiIndex):
        ps.index = ps.index.map(lambda x: f"{x[0]}:{x[1]}")

    common = sorted(set(pr.index) & set(ps.index))
    if not common:
        print(f"[⚠️ No overlapping coefficients for {y}]")
        return

    df_beta = pd.DataFrame({
        "β_real": pr[common],
        "β_sim":  ps[common]
    }).reset_index().rename(columns={"index": "coef"})
    df_beta_melt = df_beta.melt(id_vars="coef", var_name="source", value_name="beta")

    plt.figure(figsize=(max(6, len(common)*0.6), 4))
    palette = {"β_real": "#4C72B0", "β_sim": "#DD8452"}
    hue_order = ["β_real", "β_sim"]

    ax = sns.barplot(
        data=df_beta_melt,
        x="coef", y="beta",
        hue="source", hue_order=hue_order,
        palette=palette,
        saturation=1.0,
        edgecolor="none",
        errorbar=None,
        legend=False
    )

    bars = [p for p in ax.patches]
    bars = sorted(bars, key=lambda p: (p.get_x(), p.get_y()))
    n_hue = len(hue_order)
    for i, patch in enumerate(bars):
        hue = hue_order[i % n_hue]
        rgb = mcolors.to_rgb(palette[hue])
        patch.set_facecolor((*rgb, 0.45))
        patch.set_edgecolor((*rgb, 1.0))
        patch.set_linewidth(1.5)

    plt.title(f"Full Regression Coefficients — {y} ({model_type.upper()})", fontsize=11, weight="bold")
    plt.ylabel("Coefficient (β)", fontsize=10)
    plt.xlabel("Variable", fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    info_lines = [
        f"Real R² = {r2_real:.3f}",
        f"Sim R² = {r2_sim:.3f}"
    ]
    if rate_strength is not None and not np.isnan(rate_strength):
        info_lines.append(f"Pass Rate (Strength)  = {rate_strength:.2f}")

    info_text = "\n".join(info_lines)

    plt.text(
        0.98, 0.95, info_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9, color="gray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3")
    )

    fig_dir = os.path.join(out_dir, "Figures_type3")
    os.makedirs(fig_dir, exist_ok=True)
    out_path_png = os.path.join(fig_dir, f"full_regression_{y}.png")
    out_path_pdf = os.path.join(fig_dir, f"full_regression_{y}.pdf")
    plt.savefig(out_path_png, dpi=300)
    plt.savefig(out_path_pdf, dpi=300)
    plt.close()
    print(f"📊 Full regression plot saved: {out_path_pdf}")
    return r2_real, r2_sim
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

        y_name = formula.split("~")[0].strip()

        if mt in ["logit", "mnlogit"]:
            if y_name in df.columns and not pd.api.types.is_numeric_dtype(df[y_name]):
                df[y_name] = pd.Categorical(df[y_name]).codes

        if mt == "ols":
            return smf.ols(formula, data=df).fit()
        elif mt == "logit":
            return smf.logit(formula, data=df).fit(disp=False, maxiter=100)
        elif mt == "mnlogit":
            return smf.mnlogit(formula, data=df).fit(disp=False, maxiter=200)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")


def _flatten_params(model):
    p=model.params
    if isinstance(p,pd.DataFrame): p=p.stack()
    return p


def _wald_pval(b1,se1,b2,se2):
    """Wald z test: H0 β1=β2"""
    try:
        se=np.sqrt(se1**2+se2**2)
        z=(b1-b2)/se
        return 2*norm.sf(abs(z))
    except Exception:
        return np.nan


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
        if model_type.lower() in ["logit", "mnlogit"]:
            all_cats = pd.Index(pd.Categorical(df_real[y]).categories).union(
                pd.Index(pd.Categorical(df_sim[y]).categories))
            dtype = pd.api.types.CategoricalDtype(categories=list(all_cats), ordered=False)
            df[y] = pd.Categorical(df[y], dtype=dtype).codes
        elif model_type.lower() == "ols":
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
        sampled_ids = rng.choice(common_ids, size=n_b, replace=True)
        rb = df_real.set_index(id_col).loc[sampled_ids].reset_index()
        sb = df_sim.set_index(id_col).loc[sampled_ids].reset_index()
        rb["__group__"] = "Real"
        sb["__group__"] = "Sim"
        df_comb = pd.concat([rb, sb], ignore_index=True)

        drop_cols = [y] + Xs
        df_comb = df_comb.dropna(subset=drop_cols)
        if len(df_comb) < len(Xs) * 2 + 2:
            points.append(0)
            continue

        try:
            # === Fisher’s z test on predictive R² (strength) ===
            mr = smf.ols(formula, data=rb).fit()
            ms = smf.ols(formula, data=sb).fit()
            r_real = np.sqrt(max(0, mr.rsquared))
            r_sim  = np.sqrt(max(0, ms.rsquared))
            z1, z2 = np.arctanh(r_real), np.arctanh(r_sim)
            se = np.sqrt(1 / (len(rb) - 3) + 1 / (len(sb) - 3))
            z = (z1 - z2) / se
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

    r2_rows = []
    for y, mt in success_outcomes:
        res_stren  = next((r for r in summary_all.to_dict("records")
                        if r["response"] == y and r["mode"] == "strength"), None)
        rate_stren  = res_stren["insignificant_rate"] if res_stren else np.nan

        r2_real, r2_sim = _plot_full_regression_coefficients(
            df_real, df_sim, y, Xs, mt, out_dir,
            rate_strength=rate_stren
        )
        r2_rows.append({
            "response": y,
            "r2_real": r2_real,
            "r2_sim": r2_sim
        })
    r2_df = pd.DataFrame(r2_rows)

    save_dir = out_dir / "Data_type3"
    save_dir.mkdir(parents=True, exist_ok=True)

    r2_df.to_csv(save_dir / "regression_r2.csv", index=False)
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
