#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
type5_event_order_equal_assoc_bootstrap.py
-----------------------------------------------------------
Bootstrap test of EQUAL ASSOCIATION STRENGTH between event_order
and each predictor across REAL vs SIMULATED datasets.

Method:
  - Use log-linear LRT: test A:B:Group 3-way interaction = 0
  - Repeat with bootstrap matched samples
  - Compute insignificant rate (p > α)

Outputs:
  • summary.csv — per-variable insignificant rates
  • Bar plot — Equal Association Pass Rate
-----------------------------------------------------------
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
# ================== Cramér’s V Pairwise Plot ==================
def _cramers_v(table):
    """Compute Cramér’s V from contingency table"""
    chi2_val = chi2_contingency(table, correction=False)[0]
    n = table.sum().sum()
    phi2 = chi2_val / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def compute_cramers_v_one_to_many(df_real, df_sim, core_var, vars_list):
    """
    Compute Cramér’s V between a core categorical variable and multiple other categorical variables.
    Returns a DataFrame of results for REAL and SIMULATED datasets.

    Parameters
    ----------
    df_real, df_sim : pd.DataFrame
        Real and simulated datasets (must contain all variables)
    core_var : str
        The central variable (e.g., "event_order")
    vars_list : list of str
        List of other categorical variables to test against

    Returns
    -------
    pd.DataFrame
        Columns: core_var, var, real_v, sim_v
    """
    records = []
    for v in vars_list:
        if v == core_var:
            continue
        try:
            ct_r = pd.crosstab(df_real[core_var], df_real[v])
            ct_s = pd.crosstab(df_sim[core_var], df_sim[v])
            if (
                ct_r.shape[0] > 1 and ct_r.shape[1] > 1 and
                ct_s.shape[0] > 1 and ct_s.shape[1] > 1
            ):
                vr = _cramers_v(ct_r)
                vs = _cramers_v(ct_s)
                records.append({
                    "core_var": core_var,
                    "var": v,
                    "real_v": vr,
                    "sim_v": vs,
                    "abs_diff": abs(vr - vs)
                })
        except Exception:
            continue
    return pd.DataFrame(records)
def plot_cramers_v_one_to_many(df_pairs, out_dir, core_var, var_name_map=None, name=None, summary_df=None):
    """Draw a compact Cramér’s V comparison plot for the core variable."""
    if df_pairs.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    name_map = var_name_map or {}
    palette = {"Real": "#4C72B0", "Simulated": "#DD8452"}

    rate_map = {}
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            v = row.get("var") or row.get("predictor")
            rate_map[v] = row.get("insignificant_rate", np.nan)

    plt.figure(figsize=(4.2, max(3, 0.45 * len(df_pairs))))
    for i, row in df_pairs.iterrows():
        v = row["var"]
        rv, sv = row["real_v"], row["sim_v"]
        if not np.isnan(rv) and not np.isnan(sv):
            plt.plot([rv, sv], [i, i], color="gray", lw=1.0, alpha=0.8)
            plt.scatter(rv, i, color=palette["Real"], s=40, alpha=0.6,
                        label="Real" if i == 0 else "", zorder=3)
            plt.scatter(sv, i, color=palette["Simulated"], s=40, alpha=0.6,
                        label="Simulated" if i == 0 else "", zorder=3)
        rate = rate_map.get(v, np.nan)
        if not np.isnan(rate):
            plt.text(1.05, i, f"{rate:.2f}", fontsize=8, va="center", color="gray")

    ylabels = [name_map.get(v, v) for v in df_pairs["var"]]
    title = name_map.get(core_var, core_var)
    plt.yticks(range(len(df_pairs)), ylabels, fontsize=8)
    plt.title(f"{title}", fontsize=10)
    # import pdb;pdb.set_trace()
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.xlim(-0.05, 1)
    plt.xlabel("Cramér’s V", fontsize=9)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"cramersV_{name}.png")
    plt.savefig(out_path, dpi=300)
    out_path = os.path.join(out_dir, f"cramersV_{name}.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Cramér’s V (one-to-many) plot saved → {out_path}")
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
            # variance approximation (Cohen, 1988)
            var_eta2 = (2 * eta2 * (1 - eta2)**2) / max(1, n - k - 1)
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

    def _cramers_v_and_var(tab):
        """Compute Cramér’s V and its delta-method variance."""
        try:
            chi2_val, _, _, _ = chi2_contingency(tab, correction=False)
            n = tab.sum().sum()
            r, c = tab.shape
            if n == 0 or min(r, c) <= 1:
                return np.nan, np.nan
            V = np.sqrt((chi2_val / n) / min(r - 1, c - 1))
            # delta-method variance approximation
            var_V = ((1 - V ** 2) ** 2) / (2 * n * (min(r - 1, c - 1)) ** 2)
            return V, var_V
        except Exception:
            return np.nan, np.nan

    # === Split by group (0=real, 1=sim) ===
    groups = df[group_col].unique()
    if len(groups) != 2:
        return np.nan
    g0, g1 = sorted(groups)
    df0 = df[df[group_col] == g0]
    df1 = df[df[group_col] == g1]

    ct0 = pd.crosstab(df0[v1], df0[v2])
    ct1 = pd.crosstab(df1[v1], df1[v2])
    if ct0.shape[0] < 2 or ct0.shape[1] < 2 or ct1.shape[0] < 2 or ct1.shape[1] < 2:
        return np.nan

    V0, var0 = _cramers_v_and_var(ct0)
    V1, var1 = _cramers_v_and_var(ct1)
    if np.isnan(V0) or np.isnan(V1):
        return np.nan

    # delta-method z test
    z = (V0 - V1) / np.sqrt(var0 + var1)
    p = 2 * norm.sf(abs(z))
    return float(p)

# ---------- Bootstrap ----------
def _bootstrap_assoc_significance(df_real, df_sim, event_order_col, pred, pred_cfg,
                                  B, alpha, sample_n, ratio,
                                  rng=None, id_col="profile_id"):
    """
    Bootstrap test for equal association between event_order (categorical)
    and a predictor (categorical or numeric), comparing REAL vs SIMULATED.

    Supports:
      - categorical predictor → log-linear LRT
      - numeric predictor → two-way ANOVA
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
    common_ids = sorted(set(df_real[id_col]) & set(df_sim[id_col]))
    if len(common_ids) < 5:
        return None
    # print(len(common_ids))
    n_total = len(common_ids)
    n_b = int(sample_n or round(n_total * ratio))
    pvals = []

    # ---- Determine predictor type ----
    pred_type = pred_cfg.get("type", "categorical").lower()

    for _ in range(B):
        try:
            sampled_ids = rng.choice(common_ids, n_b, replace=True)
            rb = df_real.set_index(id_col).loc[sampled_ids].reset_index()
            sb = df_sim.set_index(id_col).loc[sampled_ids].reset_index()
            rb = rb.dropna(subset=[event_order_col, pred])
            sb = sb.dropna(subset=[event_order_col, pred])
            # import pdb; pdb.set_trace()
            if len(rb) < 3 or len(sb) < 3:
                continue
            # print(len(rb),len(sb))
            if pred_type == "categorical":
                tmp = pd.concat([rb.assign(__grp__=0), sb.assign(__grp__=1)], axis=0)
                p = _test_equal_assoc_cat_cat_strength(tmp, event_order_col, pred, "__grp__")
            else:
                p = _test_equal_assoc_num_cat_strength(
                    rb[pred].values, rb[event_order_col].astype(str).values,
                    sb[pred].values, sb[event_order_col].astype(str).values
                )
            if np.isnan(p):
                pvals.append(0)
            else:
                pvals.append(p)
            # print(pvals)
        except Exception:
            continue

    if not pvals:
        return None

    insignificant_rate = np.mean(np.array(pvals) > alpha)
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

            if real_n < min_n and sim_n < min_n:
                print(f"⚠️ Skipping combo {combo_tag} (real & sim < {min_n} samples)")
                continue

            if real_n >= min_n and sim_n < min_n:
                print(f"❗ Sim has < {min_n} valid samples for combo {combo_tag} → assigning score = 0")

                for pred in cfg['predictors'].keys():
                    combo_results.append({
                        'combo': combo_tag,
                        'predictor': pred,
                        'insignificant_rate': 0.0,
                        'mean_p': 0.0,
                        'pass_count': 0,
                        'iterations': 1,
                        'comment': f"Sim < {min_n} valid samples; scored as 0"
                    })
                continue

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

            # ---- Association significance for each predictor ----
            results = []
            for pred in Xs:
                pred_cfg = cfg["predictors"].get(pred, {"type": "categorical"})
                res = _bootstrap_assoc_significance(
                    df_real=real_valid,
                    df_sim=sim_valid,
                    event_order_col="event_order",
                    pred=pred,
                    pred_cfg=pred_cfg,
                    B=cfg.get("bootstrap_B", 1000),
                    alpha=cfg.get("alpha", 0.05),
                    sample_n=cfg.get("bootstrap_sample_n"),
                    ratio=cfg.get("bootstrap_sample_ratio", 1.0),
                    id_col="profile_id"
                )
                if res is not None:
                    res["combo"] = "→".join(combo)
                    res["predictor"] = pred
                    results.append(res)
                # import pdb; pdb.set_trace()
            if mode == "strength":
                fig_tag = combo_tag.replace("→", "_")
                plot_dir = os.path.join(out_dir, f"Figures_type5")
                os.makedirs(plot_dir, exist_ok=True)

                var_name_map = {v: (vcfg.get("name", v)) for v, vcfg in cfg["predictors"].items()}
                df_pairs = compute_cramers_v_one_to_many(real_valid, sim_valid, "event_order", Xs)
                for _, row in df_pairs.iterrows():
                    all_pairs.append({
                        "combo": combo_tag,
                        "var": row["var"],
                        "real": row["real_v"],
                        "sim": row["sim_v"]
                    })
                plot_cramers_v_one_to_many(
                    df_pairs=df_pairs,
                    out_dir=plot_dir,
                    core_var="event_order",
                    var_name_map=var_name_map,
                    name=fig_tag,
                    summary_df=pd.DataFrame(results) if results else None
                )

                print(f"✅ Saved per-variable Cramér’s V plots → {plot_dir}")
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
