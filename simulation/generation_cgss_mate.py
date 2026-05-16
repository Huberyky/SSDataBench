#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

X2_OPTIONS = [
    "您收入的50%（一半）", "您收入的60%（六成）", "您收入的70%（七成）", "您收入的80%（八成）",
    "您收入的90%（九成）", "和您收入差不多", "您收入的150%（1.5倍）", "您收入的200%（2倍）", "您收入的300%（3倍）",
]
X3_OPTIONS = ["父母在农村", "父母在城市"]
X4_OPTIONS = ["名下没有房产", "名下有房"]
X5_OPTIONS = ["高中", "本科", "研究生"]
X6_MALE = ["有点丑", "一般", "比较漂亮"]
X6_FEMALE = ["有点丑", "一般", "比较帅"]


def rand_x1(gender, rng):
    if int(gender) == 2:
        n = rng.integers(-5, 16)
    else:
        n = rng.integers(-15, 6)
    if n < 0:
        return f"小您{abs(int(n))}岁"
    if n > 0:
        return f"大您{int(n)}岁"
    return "和您同样大"


def parse_int_1_7(v):
    try:
        x = int(v)
    except Exception:
        return None
    return x if 1 <= x <= 7 else None


def build_round(prefix, gender, rng):
    return {
        f"{prefix}1": rand_x1(gender, rng),
        f"{prefix}2": rng.choice(X2_OPTIONS),
        f"{prefix}3": rng.choice(X3_OPTIONS),
        f"{prefix}4": rng.choice(X4_OPTIONS),
        f"{prefix}5": rng.choice(X5_OPTIONS),
        f"{prefix}6": rng.choice(X6_FEMALE if int(gender) == 2 else X6_MALE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--params-config", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="./simulated_data/cgss2021_mate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pcfg = yaml.safe_load(Path(args.params_config).read_text(encoding="utf-8"))
    model = pcfg.get("llm_parameters", {}).get("model", "openai/gpt-4o-mini")
    temp = pcfg.get("llm_parameters", {}).get("temperature", 0.8)
    top_p = pcfg.get("llm_parameters", {}).get("top_p", 1.0)
    max_tokens = pcfg.get("llm_parameters", {}).get("max_completion_tokens", 600)

    dta_path = cfg["dataset"]["dta_path"]
    df = pd.read_stata(dta_path, convert_categoricals=False)
    df = df[df["A2"].isin([1, 2])].copy().reset_index(drop=True)
    sample_n = min(args.n, len(df))
    sampled = df.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = f"sim_profiles_{model.replace('/', '-')}_cgss_mate"
    run_dir = outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in sampled.iterrows():
        rec = {"profile_id": i, "A2": int(row["A2"]) }
        rec.update(build_round("XA", rec["A2"], rng))
        rec.update(build_round("XB", rec["A2"], rng))
        rec.update(build_round("XC", rec["A2"], rng))

        prompt = cfg["prompt_template"].format(**rec)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格JSON生成器，只输出合法JSON对象。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        js = json.loads(content)
        for k in ["B101_1", "B102_1", "B103_1"]:
            rec[k] = parse_int_1_7(js.get(k))
        records.append(rec)

    sim_df = pd.DataFrame(records)
    sim_csv = run_dir / f"sim_profiles_{model.replace('/', '-')}_cgss_mate.csv"
    sim_df.to_csv(sim_csv, index=False)

    real_cols = ["A2", "XA1","XA2","XA3","XA4","XA5","XA6","XB1","XB2","XB3","XB4","XB5","XB6","XC1","XC2","XC3","XC4","XC5","XC6","B101_1","B102_1","B103_1"]
    real_df = sampled[real_cols].copy().reset_index(drop=True)
    real_df.insert(0, "profile_id", np.arange(len(real_df)))
    real_csv = run_dir / f"sampled_inputs_{model.replace('/', '-')}_cgss_mate.csv"
    real_df.to_csv(real_csv, index=False)
    print(f"saved real={real_csv}\nsaved sim={sim_csv}")


if __name__ == "__main__":
    main()
