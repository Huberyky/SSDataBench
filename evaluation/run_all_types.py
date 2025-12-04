#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_all_types.py — Unified Evaluation Controller (direct function calls)
--------------------------------------------------
Imports and executes run_type*_eval functions directly,
aggregates all insignificant rates, and saves the overall summary.
"""

import os, sys, yaml, json, importlib, pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="evaluation_master.yaml")
parser.add_argument("--real_csv")
parser.add_argument("--sim_csv")
parser.add_argument("--output_dir")
args = parser.parse_args()

# -------------------------------
# Helper functions
# -------------------------------
def _read_cfg(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)


# -------------------------------
# Main Controller
# -------------------------------
def main():
    cfg_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else "evaluation_master.yaml"
    master_cfg = _read_cfg(cfg_path)

    # ✅ 覆盖配置（命令行参数优先）
    if args.real_csv:
        master_cfg["real_csv"] = args.real_csv
    if args.sim_csv:
        master_cfg["sim_csv"] = args.sim_csv
    if args.output_dir:
        master_cfg["output_dir"] = args.output_dir

    # ✅ 打印确认覆盖生效
    print("\n📂 [Runtime Overrides]")
    print(f"  real_csv   = {master_cfg.get('real_csv')}")
    print(f"  sim_csv    = {master_cfg.get('sim_csv')}")
    print(f"  output_dir = {master_cfg.get('output_dir')}")
    type_cfg_dir = Path(master_cfg.get("type_config_dir", "./evaluation/config"))
    type_code_dir = Path(master_cfg.get("type_code_dir", "./evaluation/code_by_type"))
    out_root = Path(master_cfg.get("output_dir", "./evaluation_results"))
    out_root.mkdir(parents=True, exist_ok=True)

    types = master_cfg.get("types", ["type1", "type2", "type3", "type4", "type5"])
    results_summary = []

    for type_name in types:
        module_path = type_code_dir / f"{type_name}.py"
        config_path = type_cfg_dir / f"{type_name}.yaml"

        if not module_path.exists():
            print(f"[WARN] {module_path} not found, skipping {type_name}")
            continue
        if not config_path.exists():
            print(f"[WARN] {config_path} not found, skipping {type_name}")
            continue

        # 动态 import 模块（例如 code_by_type.type1）
        module_name = f"code_by_type.{type_name}"
        module = importlib.import_module(module_name)

        # 找到函数名（约定格式 run_type1_eval / run_type2_eval / …）
        func_name = f"run_{type_name}_eval"
        if not hasattr(module, func_name):
            print(f"[WARN] {module_name} has no {func_name}, skipping.")
            continue
        func = getattr(module, func_name)

        print(f"\n🚀 Running {func_name} ...")

        # === 读取 type 级配置文件 ===
        type_cfg = _read_cfg(config_path)

        # === 参数覆盖逻辑 ===
        B = master_cfg.get("bootstrap_B", type_cfg.get("bootstrap_B"))
        alpha = master_cfg.get("alpha", type_cfg.get("alpha"))
        sample_n = master_cfg.get("bootstrap_sample_n", type_cfg.get("bootstrap_sample_n"))
        print(f"   → bootstrap_B={B}, alpha={alpha}, sample_n={sample_n}")

        # === 调用对应函数 ===
        res = func(
            config=str(config_path),
            real_csv=master_cfg.get("real_csv"),
            sim_csv=master_cfg.get("sim_csv"),
            out_dir=out_root,
            bootstrap_B=B,
            alpha=alpha,
            bootstrap_sample_n=sample_n,     # ✅ 新增：允许覆盖
            verbose=True
        )

        # === 记录结果 ===
        score = res.get("avg_insignificant_rate", None)
        results_summary.append({
            "type": type_name,
            "avg_insignificant_rate": score,
            "summary_path": res.get("summary_path")
        })

        if score is not None:
            print(f"✅ {type_name.upper()} finished — Insignificant Rate = {score:.3f}")
        else:
            print(f"[WARN] {type_name.upper()} produced no valid score.")

    # 汇总平均分
    df = pd.DataFrame(results_summary)
    df["avg_insignificant_rate"] = pd.to_numeric(df["avg_insignificant_rate"], errors="coerce")

    # 计算平均分
    avg_rate = df["avg_insignificant_rate"].mean(skipna=True)

    print("\n==================== FINAL SUMMARY ====================")
    print(df.to_string(index=False))
    print(f"\nOverall Average Insignificant Rate = {avg_rate:.3f}")
    print("=================================================")

    # === 输出 CSV ===
    out_csv = out_root / "overall_insignificant_summary.csv"

    # 添加一行“总体平均分”
    summary_with_avg = pd.concat([
        df,
        pd.DataFrame([{
            "type": "Overall Average",
            "avg_insignificant_rate": avg_rate
        }])
    ], ignore_index=True)

    summary_with_avg.to_csv(out_csv, index=False)

    print(f"📄 Saved detailed summary (with overall average) to {out_csv}\n")


if __name__ == "__main__":
    main()
