#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
type2_.py — Compare REAL vs. SIMULATED pairwise associations
via bootstrap significance tests for STRENGTH only.

----------------------------------------------------------------------
Association patterns supported:
  • categorical × categorical
  • numerical × numerical
  • numerical × categorical

Output:
  summary_type2_strength.csv (plus merged summary_type2.csv)
"""
import os, math, argparse, json, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2, norm, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from .common import read_config, clean_str, apply_value_map, drop_values, to_numeric_clean

# ========== Global warning suppress ==========
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="statsmodels")
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _test_equal_assoc_cat_cat_strength(df, v1, v2, group_col="__grp__"):
    """
    Delta-method z-test comparing Cramér’s V (association strength)
    between real and simulated categorical associations.

    H0: V_real = V_sim
    where V = sqrt( (chi2 / n) / min(r-1, c-1) )
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency, norm


    def _cramers_v_and_var_autograd(tab):
        try:
            import autograd.numpy as anp
            from autograd import jacobian
        except ImportError:
            raise ImportError("Please first install autograd: pip install autograd")
        cm = np.array(tab, dtype=float) # 混淆矩阵
        n = np.sum(cm)
        r, c = cm.shape
        q = min(r, c)
        if n == 0 or q <= 1:
            return np.nan, np.nan
        p_hat = (cm / n).flatten()
        def cramers_v_func(p_vec):

            p_mat = p_vec.reshape((r, c))

            p_row = anp.sum(p_mat, axis=1, keepdims=True)
            p_col = anp.sum(p_mat, axis=0, keepdims=True)
            expected = anp.dot(p_row, p_col)
            
            term = (p_mat - expected)**2 / (expected + 1e-20)
            phi2 = anp.sum(term)
            
            return anp.sqrt(phi2 / (q - 1))

        try:
            V_est = cramers_v_func(p_hat)
            
            if V_est < 1e-6:
                return 0.0, 0.0 


            calculate_jacobian = jacobian(cramers_v_func)
            J = calculate_jacobian(p_hat)
            

            Sigma = (np.diag(p_hat) - np.outer(p_hat, p_hat)) / n
            
            var_V = np.dot(J, np.dot(Sigma, J.T))
            
            return float(V_est), float(var_V)
            
        except Exception as e:
            return np.nan, np.nan

    groups = df[group_col].unique()
    if len(groups) != 2:
        return np.nan
    g0, g1 = sorted(groups)
    df0 = df[df[group_col] == g0]
    df1 = df[df[group_col] == g1]

    cats_v1 = sorted(set(df0[v1].dropna()) | set(df1[v1].dropna()))
    cats_v2 = sorted(set(df0[v2].dropna()) | set(df1[v2].dropna()))

    x0 = pd.Categorical(df0[v1], categories=cats_v1)
    y0 = pd.Categorical(df0[v2], categories=cats_v2)
    x1 = pd.Categorical(df1[v1], categories=cats_v1)
    y1 = pd.Categorical(df1[v2], categories=cats_v2)

    ct0 = pd.crosstab(x0, y0)
    ct1 = pd.crosstab(x1, y1)
    ct0 = ct0.reindex(index=cats_v1, columns=cats_v2, fill_value=0)
    ct1 = ct1.reindex(index=cats_v1, columns=cats_v2, fill_value=0)    
    # import pdb; pdb.set_trace()
    # print(ct0.shape, ct1.shape)
    def _eff_dim(ct):
        n_rows = (ct.sum(axis=1) > 0).sum()
        n_cols = (ct.sum(axis=0) > 0).sum()
        return n_rows, n_cols

    r0, c0 = _eff_dim(ct0)
    r1, c1 = _eff_dim(ct1)
    if r0 < 2 or c0 < 2 or r1 < 2 or c1 < 2:
        return np.nan

    V0, var0 = _cramers_v_and_var_autograd(ct0)
    V1, var1 = _cramers_v_and_var_autograd(ct1)    
    if np.isnan(V0) or np.isnan(V1):
        return np.nan

    z = (V0 - V1) / np.sqrt(var0 + var1)
    p = 2 * norm.sf(abs(z))
    # print(p)
    return float(p)

def _test_equal_assoc_num_num_strength(x1, y1, x2, y2):
    r1 = np.corrcoef(x1, y1)[0, 1]
    r2 = np.corrcoef(x2, y2)[0, 1]
    if np.isnan(r1) or np.isnan(r2):
        return np.nan
    r1, r2 = np.clip(r1, -0.999999, 0.999999), np.clip(r2, -0.999999, 0.999999)
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (len(x1) - 3) + 1 / (len(x2) - 3))
    z = (z1 - z2) / se
    return 2 * norm.sf(abs(z))

def _test_equal_assoc_num_cat_strength(num1, cat1, num2, cat2):
    """
    Delta-method z-test comparing η² (eta squared) effect sizes between
    real and simulated data.

    H0: η²_real = η²_sim
    where η² = SS_between / SS_total
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import norm

    def _eta_sq(num, cat):
        df = pd.DataFrame({"num": num, "cat": cat}).dropna()
        if df["cat"].nunique() < 2 or len(df) < 5:
            return np.nan, np.nan, np.nan
        try:
            model = smf.ols("num ~ C(cat)", data=df).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            ss_between = anova.loc["C(cat)", "sum_sq"]
            ss_total = ss_between + anova.loc["Residual", "sum_sq"]
            eta2 = ss_between / ss_total
            n = len(df)
            k = df["cat"].nunique()
            var_eta2 = (4 * eta2 * (1 - eta2)**2) / n
            return eta2, var_eta2, n
        except Exception:
            return np.nan, np.nan, np.nan
    eta1, var1, n1 = _eta_sq(num1, cat1)
    eta2, var2, n2 = _eta_sq(num2, cat2)
    if np.isnan(eta1) or np.isnan(eta2) or eta1 == eta2:
        return np.nan

    # Delta-method z test
    z = (eta1 - eta2) / np.sqrt(var1 + var2)
    p = 2 * norm.sf(abs(z))
    # print(eta1, eta2, p)
    return float(p)


# ============================================================
# === Unified Bootstrap
# ============================================================
def _bootstrap_assoc_significance(df_real, df_sim, v1, v2, cfg1, cfg2,
                                  B, alpha, sample_n, ratio, rng=None,
                                  id_col="profile_id", mode="strength"):
    """Bootstrap testing under strength mode."""
    if rng is None:
        rng = np.random.default_rng()
    if id_col not in df_real.columns or id_col not in df_sim.columns:
        raise ValueError(f"❌ Both datasets must contain column '{id_col}' for matched sampling.")
    t1 = cfg1.get("type", "categorical").lower()
    t2 = cfg2.get("type", "categorical").lower()

    def _prep(df, v, cfg):
        s = df[v].map(clean_str)
        s = apply_value_map(s, cfg.get("value_map", {}))
        s = drop_values(s, cfg.get("drop_values", []))
        if cfg.get("type", "").lower() == "numeric":
            s = to_numeric_clean(s)
        return s

    if v1 in df_real.columns: df_real[v1] = _prep(df_real, v1, cfg1)
    if v2 in df_real.columns: df_real[v2] = _prep(df_real, v2, cfg2)
    if v1 in df_sim.columns:  df_sim[v1] = _prep(df_sim, v1, cfg1)
    if v2 in df_sim.columns:  df_sim[v2] = _prep(df_sim, v2, cfg2)

    df_real_valid = df_real.dropna(subset=[v1, v2])
    df_sim_valid  = df_sim.dropna(subset=[v1, v2])
    wins = 0
    for _ in range(B):
        rb = df_real_valid.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    )
        sb = df_sim_valid.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    )
        if t1 == "categorical" and t2 == "categorical":
            tmp = pd.concat([rb.assign(__grp__=0), sb.assign(__grp__=1)], axis=0)
            p = _test_equal_assoc_cat_cat_strength(tmp, v1, v2, "__grp__")
        elif t1 == "numeric" and t2 == "numeric":
            p = _test_equal_assoc_num_num_strength(
                    rb[v1].values, rb[v2].values,
                    sb[v1].values, sb[v2].values)
        else:
            num = v1 if t1=="numeric" else v2
            cat = v2 if t1=="numeric" else v1
            p = _test_equal_assoc_num_cat_strength(
                    rb[num].values, rb[cat].values,
                    sb[num].values, sb[cat].values)

        if not np.isnan(p) and p > alpha:
            wins += 1

    return {"var1": v1, "var2": v2, "type1": t1, "type2": t2,
            "mode": mode, "insignificant_rate": wins / B}

# ============================================================
# === Evaluation Routines
# ============================================================
def run_type2_eval_one(config, real_csv, sim_csv, out_dir,
                   bootstrap_B=1000, bootstrap_sample_n=500,
                   alpha=0.05, ratio=1.0, verbose=True):
    """Run Type2 evaluation under strength mode."""
    df_real = pd.read_csv(real_csv, low_memory=False)
    df_sim  = pd.read_csv(sim_csv,  low_memory=False)
    os.makedirs(out_dir, exist_ok=True)

    B = bootstrap_B
    rng = np.random.default_rng()
    results = []
    var_list = list(config["variables"].keys())
    print(f"\n================ TYPE2 STRENGTH TESTS ================")
    print(f"Bootstrap {B} iterations | α={alpha}\n")

    for i in range(len(var_list)):
        for j in range(i + 1, len(var_list)):
            v1, v2 = var_list[i], var_list[j]
            cfg1, cfg2 = config["variables"][v1], config["variables"][v2]
            res = _bootstrap_assoc_significance(
                df_real, df_sim, v1, v2, cfg1, cfg2,
                B, alpha, bootstrap_sample_n, ratio, rng)
            if res is not None:
                results.append(res)
                if verbose:
                    print(f"{v1} × {v2}: rate={res['insignificant_rate']:.3f}")

    summary = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, "summary_type2_strength.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Saved strength summary → {summary_path}")
    if "insignificant_rate" in summary.columns:
        overall = np.nanmean(summary["insignificant_rate"])
        print(f"Average Insignificant Rate (strength) = {overall:.3f}")
    else:
        overall = np.nan
    return {"avg_insignificant_rate": overall, "summary_path": summary_path, "summary_df": summary}

# ================== Main ==================
def run_type2_eval(config, real_csv=None, sim_csv=None, out_dir=None,
                        bootstrap_B=None, bootstrap_sample_n=None, alpha=None,
                        ratio=1.0, verbose=True):
    """
    Run Type2 evaluation for STRENGTH mode.
    Runs bootstrap tests for all variable pairs and outputs summary CSVs and plots.
    """
    if isinstance(config, str):
        cfg = read_config(config)
    else:
        cfg = dict(config)
    if real_csv: cfg["real_csv"] = real_csv
    if sim_csv:  cfg["sim_csv"]  = sim_csv
    if out_dir:  cfg["out_dir"]  = out_dir
    if bootstrap_B: cfg["bootstrap_B"] = bootstrap_B
    if alpha:       cfg["alpha"] = alpha
    if bootstrap_sample_n: cfg["bootstrap_sample_n"] = bootstrap_sample_n

    df_real = pd.read_csv(cfg["real_csv"], low_memory=False)
    df_sim  = pd.read_csv(cfg["sim_csv"],  low_memory=False)
    out_dir = cfg.get("out_dir", "./Compare_Type2")
    os.makedirs(out_dir, exist_ok=True)
    B = cfg.get("bootstrap_B", 1000)
    a = cfg.get("alpha", 0.05)
    sample_n = cfg.get("bootstrap_sample_n", 500)
    ratio = cfg.get("bootstrap_sample_ratio", ratio)


    print(f"\n================ TYPE 2 (STRENGTH) COMPARISON ================")
    print(f"Bootstrap {B} iterations | α={a}\n")

    results = []
    var_list = list(cfg["variables"].keys())
    for i in range(len(var_list)):
        for j in range(i + 1, len(var_list)):
            v1, v2 = var_list[i], var_list[j]
            cfg1, cfg2 = cfg["variables"][v1], cfg["variables"][v2]

            if cfg1.get("input", False) and cfg2.get("input", False):
                if verbose:
                    print(f"⚪ Skipped bootstrap test for {v1} × {v2} (both input)")
                continue

            res = _bootstrap_assoc_significance(
                df_real, df_sim, v1, v2, cfg1, cfg2,
                B, a, sample_n, ratio
            )
            if res is not None:
                results.append(res)
                if verbose:
                    print(f"[strength] {v1} × {v2}: insignificant_rate={res['insignificant_rate']:.3f}")

    # ---- summary + avg ----
    summary = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, "summary_type2_strength.csv")
    summary.to_csv(summary_path, index=False)
    overall = np.nanmean(summary["insignificant_rate"]) if "insignificant_rate" in summary.columns else np.nan

    print(f"✅ Saved summary (strength) → {summary_path}")
    print(f"Average Insignificant Rate (strength) = {overall:.3f}\n")



    # =======================================================
    # === Aggregate summary into single CSV
    # =======================================================
    print("\n📊 Aggregating strength summary...")

    merged = summary.copy()
    if not merged.empty and "mode" not in merged.columns:
        merged["mode"] = "strength"

    avg_rows = []
    if "insignificant_rate" in merged.columns and not merged.empty:
        strength_avg = np.nanmean(merged["insignificant_rate"])
        avg_row = {c: "" for c in merged.columns}
        avg_row["key"] = "avg_strength"
        avg_row["mode"] = "strength"
        avg_row["insignificant_rate"] = strength_avg
        avg_rows.append(avg_row)

        overall_row = {c: "" for c in merged.columns}
        overall_row["key"] = "avg"
        overall_row["mode"] = "all"
        overall_row["insignificant_rate"] = strength_avg
        avg_rows.append(overall_row)

    if avg_rows:
        merged = pd.concat([merged, pd.DataFrame(avg_rows)], ignore_index=True)

    merged_path = os.path.join(out_dir, "summary_type2.csv")
    merged.to_csv(merged_path, index=False)
    print(f"✅ Saved merged summary → {merged_path}")
    if avg_rows:
        print(f"  strength avg = {avg_rows[0]['insignificant_rate']:.3f}")
    return {
        "avg_strength": avg_rows[0]["insignificant_rate"] if avg_rows else np.nan,
        "avg_insignificant_rate": avg_rows[0]["insignificant_rate"] if avg_rows else np.nan,
        "summary_path": merged_path,
        "summary_df": merged
    }

# ============================================================
# === CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap_B", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    run_type2_eval(args.config, bootstrap_B=args.bootstrap_B, alpha=args.alpha)
