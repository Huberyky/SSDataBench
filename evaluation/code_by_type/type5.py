#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type5.py
-----------------------------------------------------------
Bootstrap test of EQUAL ASSOCIATION STRENGTH between event_order
and each predictor across REAL vs SIMULATED datasets
via bootstrap significance tests.

"""

import os, argparse, json, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2

from .common import read_config

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _clean_series(s, cfg):
    s = s.astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
    if "value_map" in cfg:
        for k, v in cfg["value_map"].items():
            s = s.replace(k, v)
    s = s.replace(["inf", "Inf", "INF"], np.inf)
    if cfg.get("type", "").lower() == "numeric":
        return pd.to_numeric(s, errors="coerce")
    return s.astype("category")


def _compute_order_label(row, event_vars):
    events = []
    for v in event_vars:
        val = row[v]
        if pd.isna(val):
            continue
        events.append((v, float(val)))
    if not events:
        return np.nan
    ordered = sorted(events, key=lambda x: x[1])
    return "-".join([x[0] for x in ordered])
from scipy.stats import chi2, norm, chi2_contingency

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

# ---------- Bootstrap ----------
def _bootstrap_assoc_significance(df_real, df_sim, event_order_col, pred, pred_cfg,
                                  B, alpha, sample_n, ratio,
                                  rng=None, id_col="profile_id", mode="strength"):
    """
    Bootstrap test for equal association between event_order (categorical)
    and a predictor (categorical or numeric), comparing REAL vs SIMULATED.
    """
    if rng is None:
        rng = np.random.default_rng()
    if id_col not in df_real.columns or id_col not in df_sim.columns:
        raise ValueError(f"❌ Both datasets must contain '{id_col}'.")

    # ---- Prepare predictor ----
    def _prep(df, v, cfg):
        s = df[v].astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
        if "value_map" in cfg:
            for k, vmap in cfg["value_map"].items():
                s = s.replace(k, vmap)
        if "drop_values" in cfg:
            s = s.replace(cfg["drop_values"], np.nan)
        if cfg.get("type", "").lower() == "numeric":
            s = pd.to_numeric(s, errors="coerce")
        else:
            s = s.astype("category")
        return s

    for df in (df_real, df_sim):
        if pred in df.columns:
            df[pred] = _prep(df, pred, pred_cfg)
    # import pdb; pdb.set_trace()
    # ---- Matched IDs ----

    pvals = []
    # print(len(df_real), len(df_sim))
    # ---- Determine predictor type ----
    pred_type = pred_cfg.get("type", "categorical").lower()
    # print(sample_n)
    for _ in range(B):
        rb = df_real.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    )
        sb = df_sim.sample(
        n=sample_n, replace=True,
        random_state=rng.integers(1e9)
    )

        # import pdb; pdb.set_trace()
        # if len(rb) < 3 or len(sb) < 3:
        #     continue
        # print(len(rb),len(sb))
        if pred_type == "categorical":
            tmp = pd.concat([rb.assign(__grp__=0), sb.assign(__grp__=1)], axis=0)
            p = _test_equal_assoc_cat_cat_strength(tmp, event_order_col, pred, "__grp__")
                # print(p)
        else:

            p = _test_equal_assoc_num_cat_strength(
                rb[pred].values, rb[event_order_col].astype(str).values,
                sb[pred].values, sb[event_order_col].astype(str).values
            )
            # print(p)
        if np.isnan(p):
            pvals.append(0)   
        else:
            pvals.append(p)
        # print(pvals)
    if not pvals:
        return None

    insignificant_rate = np.mean(np.array(pvals) > alpha)
    # print(insignificant_rate)
    return {
        "predictor": pred,
        "type": pred_type,
        "insignificant_rate": insignificant_rate,
        "iterations": len(pvals)
    }


# ---------- Main ----------
def run_type5_eval(
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
    Type5 Equal Association Evaluation Entry
    Priority of parameter sources (highest to lowest):
      1. Function arguments
      2. Config file (type5.yaml)
      3. Built-in defaults
    """
    cfg = read_config(config) if isinstance(config, str) else dict(config)

    cfg["real_csv"] = real_csv or cfg.get("real_csv")
    cfg["sim_csv"] = sim_csv or cfg.get("sim_csv")
    cfg["out_dir"] = out_dir or cfg.get("out_dir", "./results_type5")
    cfg["bootstrap_B"] = bootstrap_B or cfg.get("bootstrap_B", 10)
    cfg["bootstrap_sample_n"] = bootstrap_sample_n or cfg.get("bootstrap_sample_n", None)
    cfg["alpha"] = alpha or cfg.get("alpha", 0.05)

    real=pd.read_csv(cfg["real_csv"], low_memory=False)
    sim=pd.read_csv(cfg["sim_csv"],  low_memory=False)
    out_dir=cfg.get("out_dir","./Compare_Type5")
    os.makedirs(out_dir,exist_ok=True)

    event_cfgs=cfg["event_variables"]
    event_vars=list(event_cfgs.keys())
    for v,spec in event_cfgs.items():
        real[v]=_clean_series(real[v],spec)
        sim[v] =_clean_series(sim[v],spec)
        real[v] = pd.to_numeric(real[v], errors="coerce")
        sim[v]  = pd.to_numeric(sim[v],  errors="coerce")

    import itertools
    results_all = {}
    for mode in ["strength"]:
        print(f"\n================ TYPE 5 ({mode.upper()}) COMPARISON ================")
        print(f"Bootstrap {cfg.get('bootstrap_B',1000)} iterations | α={cfg.get('alpha',0.05)}")

        all_combos = []
        for k in (3,):
            all_combos += list(itertools.combinations(event_vars, k))

        combo_results = []
        all_pairs = []
        for combo in all_combos:
            combo = list(combo)
            combo_tag = "→".join(combo)
            print(f"\n▶️ Evaluating event combo: {combo}")
            r=real.copy()
            s=sim.copy()
            r=r.dropna(subset=combo,how="any")
            s =s.dropna(subset=combo,how="any")

            r["event_order"]=r.apply(_compute_order_label,axis=1,args=(combo,))
            s["event_order"] =s.apply(_compute_order_label,axis=1,args=(combo,))
            # print("Unique event_order (real):", r["event_order"].value_counts(dropna=False))
            # print("Unique event_order (sim):", s["event_order"].value_counts(dropna=False))
            real_valid = r.dropna(subset=["event_order"]).copy()
            sim_valid  = s.dropna(subset=["event_order"]).copy()
            min_n = 5

            real_n = len(real_valid)
            sim_n  = len(sim_valid)

            if real_n < min_n:
                print(f"⚠️ Skipping combo {combo_tag} (real < {min_n} valid samples)")
                continue
            if verbose:
                print(f"\n================ TYPE 5 COMPARISON ================")
                print(f"Bootstrap {cfg.get('bootstrap_B',1000)} iterations | α={cfg.get('alpha',0.05)}")
                # print("Top Order Patterns (Real):")
                # print(real["event_order"].value_counts(normalize=True).head(),"\n")
                # print("Top Order Patterns (Sim):")
                # print(sim["event_order"].value_counts(normalize=True).head(),"\n")

            # predictors

            preds = cfg["predictors"]
            for v, spec in preds.items():
                for df in (real_valid, sim_valid):
                    df[v] = _clean_series(df[v], spec)
            Xs = list(preds.keys())
            # print(Xs)
            # ---- Association significance for each predictor ----
            results = []
            for pred in Xs:
                pred_cfg = cfg["predictors"].get(pred, {"type": "categorical"})
                res = _bootstrap_assoc_significance(
                    df_real=real_valid,
                    df_sim=sim_valid,
                    event_order_col="event_order",   # ✅ event sequence 是 outcome
                    pred=pred,                       # ✅ 单个 predictor
                    pred_cfg=pred_cfg,
                    B=cfg.get("bootstrap_B", 1000),
                    alpha=cfg.get("alpha", 0.05),
                    sample_n=cfg.get("bootstrap_sample_n"),
                    ratio=cfg.get("bootstrap_sample_ratio", 1.0),
                    id_col="profile_id"
                )
                # print(res)
                if res is not None:
                    res["combo"] = "→".join(combo)
                    res["predictor"] = pred
                    results.append(res)
                    print(f"[strength] {combo} × {pred}: insignificant_rate={res['insignificant_rate']:.3f}")

            combo_results.extend(results)
        if all_pairs:
            save_dir = os.path.join(out_dir, "Data_type5")
            os.makedirs(save_dir, exist_ok=True)

            pd.DataFrame(all_pairs).to_csv(
                os.path.join(save_dir, "CramersV_pairs_all.csv"),
                index=False
            )

        df_summary = pd.DataFrame(combo_results)

        # --- Add avg row only when summary is not empty ---
        if not df_summary.empty:
            numeric_means = df_summary.select_dtypes(include=[float, int]).mean(numeric_only=True).to_dict()

            avg_row = {col: "" for col in df_summary.columns}
            avg_row.update(numeric_means)

            avg_row["combo"] = "avg"
            avg_row["mode"] = mode

            df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)


        # --- Save summary CSV ---
        summary_path = os.path.join(out_dir, f"summary_type5_{mode}.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"✅ Saved summary ({mode}) → {summary_path}")


        # --- Robust error control for 'insignificant_rate' ---
        if "insignificant_rate" not in df_summary.columns:
            print(f"[⚠️ Warning] 'insignificant_rate' column missing in Type 5 summary ({mode}).")
            print("Possible causes:")
            print("  • P-values were never computed")
            print("  • Some variables had invalid contingency tables")
            print("  • Summary rows did not include insignificant_rate\n")
            print("Proceeding with avg_insig = NaN.\n")

            avg_insig = np.nan
        else:
            avg_insig = float(np.nanmean(df_summary["insignificant_rate"]))

        print(f"Average Insignificant Rate ({mode}) = {avg_insig:.3f}\n")


        # --- Save combined results ---
        results_all[mode] = {
            "avg_insignificant_rate": avg_insig,
            "summary_path": summary_path,
            "summary_df": df_summary
        }

    # =======================================================
    # === Aggregate strength summary into one CSV
    # =======================================================
    print("\n📊 Aggregating strength summary...")

    merged_all = []
    for mode, res in results_all.items():
        df_sum = res["summary_df"].copy()
        if "mode" not in df_sum.columns:
            df_sum["mode"] = mode
        merged_all.append(df_sum)

    merged = pd.concat(merged_all, ignore_index=True) if merged_all else pd.DataFrame()

    avg_rows = []
    if not merged.empty and "insignificant_rate" in merged.columns:
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
    else:
        strength_avg = np.nan

    if avg_rows:
        merged = pd.concat([merged, pd.DataFrame(avg_rows)], ignore_index=True)

    merged_path = os.path.join(out_dir, "summary_type5.csv")
    merged.to_csv(merged_path, index=False)

    print(f"✅ Saved merged summary → {merged_path}")
    if not np.isnan(strength_avg):
        print(f"  strength avg = {strength_avg:.3f}\n")

    return {
        "avg_strength": strength_avg,
        "avg_insignificant_rate": strength_avg,
        "summary_path": merged_path,
        "summary_df": merged
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap_B", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    run_type5_eval(args.config, bootstrap_B=args.bootstrap_B, alpha=args.alpha)
