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
def plot_numeric_correlations(df_real, df_sim, cfg, summary, out_dir):
    """Plot numeric-vs-numeric scatter/regression overlays for both datasets."""
    num_vars = [v for v, vcfg in cfg["variables"].items()
                if (vcfg.get("type") or "").lower() == "numeric"]
    if len(num_vars) < 2:
        print("ℹ️ Not enough numeric variables to draw correlation plots.")
        return

    print("\n📈 Drawing numeric × numeric correlation scatter plots...")
    corr_dir = os.path.join(out_dir, "Figures_type2", "NumNum")
    os.makedirs(corr_dir, exist_ok=True)

    rate_map = {}
    if "mode" in summary.columns:
        for _, row in summary.iterrows():
            key = frozenset([row["var1"], row["var2"]])
            mode = str(row.get("mode", "")).lower()
            val = row.get("insignificant_rate", np.nan)
            if key not in rate_map:
                rate_map[key] = {}
            rate_map[key][mode] = val
    else:
        # fallback for single-mode summaries
        rate_map = {frozenset([r["var1"], r["var2"]]): {"single": r["insignificant_rate"]}
                    for _, r in summary.iterrows()}

    def _safe_corrcoef(df, v1, v2):
        x = pd.to_numeric(df[v1], errors="coerce")
        y = pd.to_numeric(df[v2], errors="coerce")
        mask = ~x.isna() & ~y.isna()
        if mask.sum() < 3:
            return np.nan
        return np.corrcoef(x[mask], y[mask])[0, 1]

    for i in range(len(num_vars)):
        for j in range(i + 1, len(num_vars)):
            v1, v2 = num_vars[i], num_vars[j]
            name1 = cfg["variables"][v1].get("name", v1)
            name2 = cfg["variables"][v2].get("name", v2)
            rates = rate_map.get(frozenset([v1, v2]), {})

            r_real = _safe_corrcoef(df_real, v1, v2)
            r_sim  = _safe_corrcoef(df_sim, v1, v2)

            plt.figure(figsize=(4.2, 4.2))
            sns.regplot(
                x=v1, y=v2, data=df_real,
                scatter_kws={"alpha": 0.4, "color": "#4C72B0", "s": 20},
                line_kws={"color": "#4C72B0"}, label="Real"
            )
            sns.regplot(
                x=v1, y=v2, data=df_sim,
                scatter_kws={"alpha": 0.4, "color": "#DD8452", "s": 20},
                line_kws={"color": "#DD8452"}, label="Simulated"
            )

            plt.xlabel(name1, fontsize=10)
            plt.ylabel(name2, fontsize=10)
            plt.title(f"{name1} vs. {name2}", fontsize=10)
            plt.legend(frameon=False, fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.4)

            ax = plt.gca()
            text_lines = []
            if not np.isnan(r_real):
                text_lines.append(f"r(Real) = {r_real:.2f}")
            if not np.isnan(r_sim):
                text_lines.append(f"r(Sim) = {r_sim:.2f}")

            if "strength" in rates and not np.isnan(rates["strength"]):
                text_lines.append(f"Strength = {rates['strength']:.2f}")
            if "single" in rates and not np.isnan(rates["single"]):
                text_lines.append(f"Pass Rate = {rates['single']:.2f}")

            if text_lines:
                plt.text(
                    0.95, 0.95, "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=9, color="gray",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, boxstyle="round,pad=0.2")
                )

            plt.tight_layout()
            out_path = os.path.join(corr_dir, f"corr_{v1}_{v2}.png")
            plt.savefig(out_path, dpi=300)
            out_path = os.path.join(corr_dir, f"corr_{v1}_{v2}.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()

    print(f"✅ Saved numeric correlation figures → {corr_dir}")
# -------------------------------------------------------------------


def plot_categorical_numeric(df_real, df_sim, cfg, summary, out_dir):
    """Box/violin plots for categorical vs numeric pairs, annotated with pass rates."""
    cat_vars = [v for v, vcfg in cfg["variables"].items()
                if (vcfg.get("type") or "").lower() == "categorical"]
    num_vars = [v for v, vcfg in cfg["variables"].items()
                if (vcfg.get("type") or "").lower() == "numeric"]

    if len(cat_vars) == 0 or len(num_vars) == 0:
        print("ℹ️ No categorical × numeric variable combinations available.")
        return

    print("\n🎨 Drawing categorical × numeric distribution plots...")
    mix_dir = os.path.join(out_dir, "Figures_type2", "CatNum")
    os.makedirs(mix_dir, exist_ok=True)

    rate_map = {frozenset([row["var1"], row["var2"]]): row["insignificant_rate"]
                for _, row in summary.iterrows()}

    palette = {"Real": "#4C72B0", "Simulated": "#DD8452"}

    for cat in cat_vars:
        for num in num_vars:
            name_cat = cfg["variables"][cat].get("name", cat)
            name_num = cfg["variables"][num].get("name", num)
            rate = rate_map.get(frozenset([cat, num]), np.nan)

            df_r = df_real[[cat, num]].copy()
            df_r["source"] = "Real"
            df_s = df_sim[[cat, num]].copy()
            df_s["source"] = "Simulated"
            df_comb = pd.concat([df_r, df_s], axis=0, ignore_index=True)
            df_comb = df_comb.dropna(subset=[cat, num])

            plt.figure(figsize=(5, 4))
            sns.boxplot(
                data=df_comb,
                x=cat,
                y=num,
                hue="source",
                palette=palette,
                width=0.6,
                fliersize=2,
                boxprops=dict(alpha=0.45),
                linewidth=1.2
            )

            plt.xlabel(name_cat, fontsize=10)
            plt.ylabel(name_num, fontsize=10)
            plt.title(f"{name_num} by {name_cat}", fontsize=10)
            plt.xticks(rotation=30, ha='right')
            plt.legend(title="", frameon=False, fontsize=8)
            plt.grid(True, axis="y", linestyle="--", alpha=0.4)

            ax = plt.gca()
            if not np.isnan(rate):
                plt.text(
                    0.98, 0.95, f"Pass Rate = {rate:.2f}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=9, color="gray",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, boxstyle="round,pad=0.2")
                )

            sns.despine()
            plt.tight_layout()
            out_path = os.path.join(mix_dir, f"catnum_{cat}_{num}.png")
            plt.savefig(out_path, dpi=300)
            out_path = os.path.join(mix_dir, f"catnum_{cat}_{num}.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()

    print(f"✅ Saved categorical × numeric boxplots → {mix_dir}")
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

def compute_cramers_v_pairs(df_real, df_sim, vars_list):
    """Compute Cramér’s V for each categorical pair in REAL and SIMULATED datasets"""
    records = []
    for i, v1 in enumerate(vars_list):
        for j in range(i + 1, len(vars_list)):
            v2 = vars_list[j]
            try:
                ct_r = pd.crosstab(df_real[v1], df_real[v2])
                ct_s = pd.crosstab(df_sim[v1], df_sim[v2])
                if ct_r.shape[0] > 1 and ct_r.shape[1] > 1 and ct_s.shape[0] > 1 and ct_s.shape[1] > 1:
                    vr = _cramers_v(ct_r)
                    vs = _cramers_v(ct_s)
                    records.append({"pair": f"{v1} × {v2}", "var1": v1, "var2": v2,
                                    "real_v": vr, "sim_v": vs})
            except Exception:
                continue
    return pd.DataFrame(records)
def plot_cramers_v_per_variable(df_pairs, out_dir, title_prefix="", var_name_map=None, summary_df=None):
    """Draw a per-variable Cramér’s V comparison panel (Real vs Simulated)."""
    if df_pairs.empty:
        return

    name_map = var_name_map or {}
    vars_all = sorted(set(df_pairs["var1"]) | set(df_pairs["var2"]))
    os.makedirs(out_dir, exist_ok=True)

    palette = {"Real": "#4C72B0", "Simulated": "#DD8452"}

    rate_map = {}
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            k1, k2 = row["var1"], row["var2"]
            rate = row["insignificant_rate"]
            rate_map[frozenset([k1, k2])] = rate

    def _get_v(df, v1, v2, col):
        match = df[((df["var1"] == v1) & (df["var2"] == v2)) |
                   ((df["var1"] == v2) & (df["var2"] == v1))]
        if not match.empty:
            return match.iloc[0][col]
        return np.nan

    for v in vars_all:
        rows = []
        for other in vars_all:
            if v == other:
                rows.append({"other": v, "real_v": np.nan, "sim_v": np.nan})
            else:
                rows.append({
                    "other": other,
                    "real_v": _get_v(df_pairs, v, other, "real_v"),
                    "sim_v": _get_v(df_pairs, v, other, "sim_v"),
                    "rate": rate_map.get(frozenset([v, other]), np.nan)
                })

        sub = pd.DataFrame(rows)
        if sub.empty:
            continue

        plt.figure(figsize=(3.5, max(3, 0.35 * len(vars_all))))
        for i, row in sub.iterrows():
            if row["other"] == v:
                continue
            if not np.isnan(row["real_v"]) and not np.isnan(row["sim_v"]):
                plt.plot([row["real_v"], row["sim_v"]], [i, i],
                         color="gray", lw=1.0, alpha=0.8)
        for i, row in sub.iterrows():
            if row["other"] == v:
                continue
            if not np.isnan(row["real_v"]):
                plt.scatter(
                    row["real_v"], i,
                    color=palette["Real"], s=35, alpha=0.6,
                    label="Real" if i == 0 else "", zorder=3
                )
            if not np.isnan(row["sim_v"]):
                plt.scatter(
                    row["sim_v"], i,
                    color=palette["Simulated"], s=35, alpha=0.6,
                    label="Simulated" if i == 0 else "", zorder=3
                )
            if "rate" in row and not np.isnan(row["rate"]):
                plt.text(1.05, i, f"{row['rate']:.2f}", fontsize=8, va="center", color="gray")

        ylabels = [name_map.get(o, o) for o in sub["other"]]
        title = name_map.get(v, v)
        plt.yticks(range(len(sub)), ylabels, fontsize=8)
        plt.title(title if not title_prefix else f"{title_prefix}: {title}", fontsize=10)
        plt.grid(True, axis="x", linestyle="--", alpha=0.4)
        plt.xlim(-0.05, 1)
        plt.tight_layout()

        fpath = os.path.join(out_dir, f"cramers_{v}.png")
        plt.savefig(fpath, dpi=300)
        fpath = os.path.join(out_dir, f"cramers_{v}.pdf")
        plt.savefig(fpath, dpi=300)
        plt.close()
# ================== Equal Association Tests ==================

def _get_sizes(n1, n2, sample_n=None, ratio=1.0):
    if sample_n:
        n1b = n2b = int(sample_n)
    else:
        n1b, n2b = int(round(n1 * ratio)), int(round(n2 * ratio))
    return max(1, min(n1b, n1)), max(1, min(n2b, n2))


# ============================================================
# === 2. Association Tests: STRENGTH
# ============================================================

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

def _test_equal_assoc_num_num_strength(x1, y1, x2, y2):
    """Fisher z-test comparing correlation magnitudes (strength)."""
    if len(x1) < 4 or len(x2) < 4:
        return np.nan
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


# ============================================================
# === 4. Unified Bootstrap
# ============================================================
def _bootstrap_assoc_significance(df_real, df_sim, v1, v2, cfg1, cfg2,
                                  B, alpha, sample_n, ratio, rng=None,
                                  id_col="profile_id"):
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

    common_ids = sorted(set(df_real_valid[id_col]) & set(df_sim_valid[id_col]))
    if len(common_ids) < 5:
        return None
    # print(len(common_ids))
    wins = 0
    for _ in range(B):
        sampled_ids = rng.choice(common_ids, sample_n, replace=True)
        rb = df_real.set_index(id_col).loc[sampled_ids].reset_index()
        sb = df_sim.set_index(id_col).loc[sampled_ids].reset_index()
        rb, sb = rb.dropna(subset=[v1, v2]), sb.dropna(subset=[v1, v2])
        if len(rb) < 3 or len(sb) < 3:
            continue
        # print(len(rb),len(sb))
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
# === 5. Evaluation Routines
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

    # ---- Step 0: Compute Cramér’s V (for later plotting) ----
    cat_vars = [v for v, vcfg in cfg["variables"].items()
                if (vcfg.get("type") or "").lower() == "categorical"]
    if len(cat_vars) > 1:
        print("\n📊 Computing Cramér’s V for categorical pairs...")
        df_pairs = compute_cramers_v_pairs(df_real, df_sim, cat_vars)
        type2dir = os.path.join(out_dir, "Data_type2")
        os.makedirs(type2dir, exist_ok=True)
        pair_path = os.path.join(type2dir, "cramers_v_pairs.csv")
        df_pairs.to_csv(pair_path, index=False)
        print(f"✅ Saved pairwise Cramér’s V → {pair_path}")
    else:
        df_pairs = pd.DataFrame()

    # =======================================================
    # === Run STRENGTH evaluation only
    # =======================================================
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

    if len(cat_vars) > 1:
        print("📊 Drawing per-variable Cramér’s V comparison (with insignificant rates)...")
        plot_dir = os.path.join(out_dir, "Figures_type2/CatCat")
        var_name_map = {v: (vcfg.get("name", v)) for v, vcfg in cfg["variables"].items()}
        plot_cramers_v_per_variable(
            df_pairs,
            plot_dir,
            var_name_map=var_name_map,
            summary_df=summary
        )
        print(f"✅ Saved per-variable Cramér’s V plots → {plot_dir}")

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

    plot_numeric_correlations(df_real, df_sim, cfg, merged, out_dir)
    plot_categorical_numeric(df_real, df_sim, cfg, merged, out_dir)
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
