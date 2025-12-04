#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_profiles_from_yaml.py (life_trajectory version, with detailed English comments)
-----------------------------------------------------------------------------------------
Main functions:
1. Load content and parameter YAML configurations
2. Optionally sample conditioning inputs from the real dataset
3. Build structured prompts for GPT to generate synthetic individuals
   - All sequential variables are merged into one field `life_trajectory`
     with format: {15: {"edu": ..., "occ": ...}, 16: {...}, ...}
4. Use multithreading to generate multiple individuals concurrently
5. Validate and flatten sequential variables into age-specific columns
6. Save all results into JSONL and CSV formats
7. Optionally run postprocess modules defined in `postprocess_utils.py`
Usage:
  export OPENAI_API_KEY=...
  python generation.py --config ./content_config.yaml --params-config ./params_config.yaml --n 100 --outdir ./Simulated --max-workers 4
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time, random, importlib
from dotenv import load_dotenv
load_dotenv()

# =====================================================
# CLI setup
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml", help="Path to content YAML config file")
parser.add_argument("--params-config", type=str, default="./params_config.yaml", help="Path to parameters YAML config file")
parser.add_argument("--n", type=int, help="Number of synthetic individuals to generate (overrides config)")
parser.add_argument("--outdir", type=str, help="Output directory (overrides config)")
parser.add_argument("--max-workers", type=int, help="Maximum number of concurrent threads (overrides config)")
args = parser.parse_args()

def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_model_name(name: str) -> str:
    """Map friendly model aliases to provider-supported identifiers."""
    alias_map = {
        "google/gemini-1.5-flash": "google/gemini-flash-1.5",
        "google/gemini-1.5-flash-latest": "google/gemini-flash-1.5",
        "google/gemini-flash-2.5": "google/gemini-flash-1.5",
    }
    normalized = alias_map.get(name, name)
    if normalized != name:
        print(f"[Info] Remapping model '{name}' to '{normalized}' for provider compatibility.")
    return normalized


def load_configs(content_config_path: str, params_config_path: str):
    cfg = _load_yaml(content_config_path)
    params_cfg = _load_yaml(params_config_path)
    return cfg, params_cfg


def derive_run_settings(args, params_cfg):
    generation_params = params_cfg.get("generation", {})
    threading_params = params_cfg.get("threading", {})
    num_profiles = args.n if args.n else generation_params.get("num_profiles", 10)
    output_dir = args.outdir if args.outdir else generation_params.get("output_directory", "./data/simulated")
    max_workers = args.max_workers if args.max_workers else threading_params.get("default_max_workers", 4)
    return num_profiles, output_dir, max_workers


def prepare_output_paths(output_dir: str, model: str, temperature: float, top_p: float):
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = model.replace("/", "-").replace(" ", "")
    base_name = f"sim_profiles_{model_clean}_temp{temperature}_top{top_p}"
    run_dir = base_dir / base_name
    if run_dir.exists():
        candidate = base_dir / f"{base_name}_{ts}"
        counter = 1
        while candidate.exists():
            candidate = base_dir / f"{base_name}_{ts}_{counter}"
            counter += 1
        run_dir = candidate
    run_dir.mkdir(parents=True, exist_ok=False)
    out_jsonl = run_dir / f"{base_name}.jsonl"
    out_csv = run_dir / f"{base_name}.csv"
    return base_dir, run_dir, out_jsonl, out_csv, base_name, model_clean


def init_llm_client(model: str):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found. Please set it for simulation.")
    client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    print(f"🌐 Using OpenRouter provider for model {model}")
    return client


def prepare_real_data(original_data_path, requested_n, run_dir, model_clean, temperature, top_p):
    df_real = None
    if original_data_path and Path(original_data_path).exists():
        try:
            df_real = pd.read_csv(original_data_path)
            print(f"✅ Loaded original data for sampling: {original_data_path}")
        except Exception as e:
            print(f"[Warning] Could not load original data: {e}")
    else:
        print("[Info] No original dataset found; input sampling skipped.")

    sampled_inputs_df = pd.DataFrame()
    if df_real is not None and not df_real.empty:
        sample_size = min(requested_n or 10, len(df_real))
        sampled_inputs_df = df_real.sample(n=sample_size, replace=False, random_state=42).reset_index(drop=True).copy()
        sampled_inputs_df["profile_id"] = np.arange(len(sampled_inputs_df))
        out_test = run_dir / f"sampled_inputs_{model_clean}_temp{temperature}_top{top_p}.csv"
        sampled_inputs_df.to_csv(out_test, index=False)
        print(f"📄 Test samples saved to: {out_test}")

    return df_real, sampled_inputs_df


# =====================================================
# Load configs & global settings
# =====================================================
cfg, params_cfg = load_configs(args.config, args.params_config)
context = cfg.get("context", "")
input_vars = cfg.get("input_variables", {})
output_vars = cfg.get("output_variables", {})
post_modules = cfg.get("postprocess_modules", [])
original_data_path = cfg.get("original_data", {}).get("file", None)
generation_params = params_cfg.get("generation", {})
MAX_RETRIES = max(0, int(generation_params.get("max_retries", 0) or 0))
RETRY_DELAY = max(0.0, float(generation_params.get("retry_delay", 0.0) or 0.0))

llm_params = params_cfg.get("llm_parameters", {})
model = _normalize_model_name(llm_params.get("model", "gpt-4o"))
temperature = llm_params.get("temperature", 1.0)
top_p = llm_params.get("top_p", 1.0)
max_completion_tokens = llm_params.get("max_completion_tokens", 5000)

num_profiles, output_dir, max_workers = derive_run_settings(args, params_cfg)
OUT_DIR, run_dir, OUT_JSONL, OUT_CSV, base_name, model_clean = prepare_output_paths(
    output_dir, model, temperature, top_p
)
FAILED_RESPONSES_DIR = run_dir / "failed_responses"
FAILED_RESPONSES_DIR.mkdir(exist_ok=True)
client = init_llm_client(model)
df_real, sampled_inputs_df = prepare_real_data(original_data_path, args.n, run_dir, model_clean, temperature, top_p)

# =====================================================
# Variable metadata
# =====================================================
FIELDS = list(output_vars.keys())
DESC_MAP = {k: v.get("description", "") for k, v in output_vars.items()}
ALLOWED = {k: v.get("allowed", None) for k, v in output_vars.items()}
TYPES = {k: v.get("type", "static") for k, v in output_vars.items()}
sequential_fields = [k for k, v in TYPES.items() if v == "sequential"]

# =====================================================
# Helpers
# =====================================================
def _supports_json_mode(model_name: str) -> bool:
    name = model_name.lower()
    supported = [
        "gpt-4o",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-5",
        "o4",
        "o3",
        "o1",
        "gemini",
        "qwen",
        "grok",
    ]
    return any(tok in name for tok in supported)


def _extract_message_content(message):
    """Normalize message content across providers (string or segment list)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for segment in content:
            if isinstance(segment, dict):
                if "text" in segment and segment["text"]:
                    parts.append(segment["text"])
                elif "content" in segment and segment["content"]:
                    parts.append(segment["content"])
                elif "json" in segment and segment["json"] is not None:
                    parts.append(json.dumps(segment["json"]))
                elif "object" in segment and segment["object"] is not None:
                    parts.append(json.dumps(segment["object"]))
                elif "data" in segment and segment["data"] is not None:
                    parts.append(json.dumps(segment["data"]))
            else:
                parts.append(str(segment))
        return "".join(parts).strip()
    return str(content or "").strip()


def _load_json_safely(raw_text: str):
    """Try to load JSON, trimming markdown fences if needed."""
    if raw_text is None:
        raise json.JSONDecodeError("Empty content", "", 0)
    text = raw_text.strip()
    if not text:
        raise json.JSONDecodeError("Empty content", raw_text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise


def _is_empty_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _record_has_empty(record: dict):
    for key, value in record.items():
        if key == "profile_id" or key in input_vars:
            continue
        if _is_empty_value(value):
            return True
    return False

def _stringify_allowed(spec):
    if isinstance(spec, list):
        return "[" + ", ".join([f'"{v}"' for v in spec]) + "]"
    if isinstance(spec, dict) and spec.get("type") == "numeric":
        rng = f"[{spec.get('min')}, {spec.get('max')}]" if spec.get("min") is not None else "any"
        special = spec.get("special", [])
        if special:
            return f"number in {rng} OR one of {special}"
        return f"number in {rng}"
    return str(spec)

def _validate_cat(value, allowed_list):
    if value is None:
        return None
    s = str(value).strip()
    if s in allowed_list:
        return s
    lowered = {str(a).lower(): a for a in allowed_list}
    return lowered.get(s.lower(), None)

def _validate_numeric(value, spec):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        lowmap = {str(x).lower(): x for x in spec.get("special", [])}
        if s.lower() in lowmap:
            return lowmap[s.lower()]
    try:
        v = float(value)
    except Exception:
        return None
    v = max(v, spec.get("min", v))
    v = min(v, spec.get("max", v))
    return int(round(v))

def sample_input_variables(profile_id):
    if sampled_inputs_df is None or sampled_inputs_df.empty:
        return {}
    row = sampled_inputs_df.iloc[profile_id]
    sampled = {}
    for k in input_vars.keys():
        sampled[k] = row[k]
    return sampled

# =====================================================
# Build prompt
# =====================================================
def build_prompt(sampled_inputs):
    desc_lines, allowed_lines = [], []
    for k in FIELDS:
        desc = DESC_MAP[k]
        allowed = ALLOWED[k]
        desc_lines.append(f"- {k} ({TYPES[k]}): {desc}")
        allowed_lines.append(f"- {k}: {_stringify_allowed(allowed)}")
    desc_block = "\n".join(desc_lines)
    allowed_block = "\n".join(allowed_lines)

    pop_context = cfg.get("population_context", {})
    age_range = pop_context.get("age_range", [14, 60])
    age_min, age_max = int(age_range[0]), int(age_range[-1])

    seq_instruction = ""
    if sequential_fields:
        seq_vars_list = ", ".join(sequential_fields)

        # Build JSON-style example string safely
        def make_example(age: int) -> str:
            inner = ", ".join([f'"{v}": "..."' for v in sequential_fields])
            return f"    {age}: {{ {inner} }}"

        example_lines = [
            make_example(age_min),
            make_example(age_min + 1),
            "    ...",
            make_example(age_max)
        ]
        seq_example = "{\n  \"life_trajectory\": {\n" + ",\n".join(example_lines) + "\n  }\n}"

        seq_instruction = (
            f"For sequential variables ({seq_vars_list}), combine them into a single field named 'life_trajectory'. "
            f"'life_trajectory' must be a JSON object mapping each age from {age_min} to {age_max} "
            f"to a sub-object that specifies all sequential variables at that age.\n"
            f"For example:\n{seq_example}"
            f"Ensure life trajectory is aligned with static variables."
        )

    input_context = ""
    if sampled_inputs:
        ctx_parts = [f"{k}: {v}" for k, v in sampled_inputs.items()]
        input_context = "Conditioned on the following input attributes:\n" + ", ".join(ctx_parts) + "\n"

    static_fields = [f for f in FIELDS if TYPES[f] != "sequential"]

    return f"""
You are simulating one *randomly selected* synthetic individual sampled from {context}
{input_context}
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
one key 'life_trajectory' plus all static variables {", ".join(static_fields)} .

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.
3) {seq_instruction}

Descriptions:
{desc_block}

Allowed values:
{allowed_block}
""".strip()

# =====================================================
# Generation
# =====================================================
def generate_single_profile(profile_id):
    sampled_inputs = sample_input_variables(profile_id)
    # import pdb; pdb.set_trace()
    prompt = build_prompt(sampled_inputs)
    # print(prompt)
    last_record = None

    for attempt in range(MAX_RETRIES + 1):
        print(
            f"[Thread {threading.current_thread().name}] Generating profile {profile_id} "
            f"(attempt {attempt + 1}/{MAX_RETRIES + 1})"
        )
        try:
            messages_payload = [
                {"role": "system", "content": "You are a strict JSON generator. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            req = dict(
                model=model,
                messages=messages_payload,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
            )
            if _supports_json_mode(model):
                req["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**req)
            content = _extract_message_content(resp.choices[0].message)
            # print(content)
            try:
                js = _load_json_safely(content)
            except json.JSONDecodeError as jde:
                fail_path = FAILED_RESPONSES_DIR / f"profile_{profile_id}.txt"
                try:
                    fail_path.write_text(content or "", encoding="utf-8")
                    print(f"[Debug] Saved raw response for profile {profile_id} to {fail_path}")
                except Exception as log_err:
                    print(f"[Debug] Failed to log raw response for profile {profile_id}: {log_err}")
                raise jde
        except Exception as e:
            print(f"[Warning] GPT generation failed for profile {profile_id}: {e}")
            js = {}

        record = dict(sampled_inputs)
        record["profile_id"] = profile_id

        pop_context = cfg.get("population_context", {})
        age_range = pop_context.get("age_range", [14, 60])
        age_min, age_max = int(age_range[0]), int(age_range[-1])

        # ---- Static variables ----
        for k in FIELDS:
            if TYPES[k] != "sequential":
                val = js.get(k, None)
                spec = ALLOWED[k]
                if isinstance(spec, list):
                    record[k] = _validate_cat(val, spec)
                elif isinstance(spec, dict) and spec.get("type") == "numeric":
                    record[k] = _validate_numeric(val, spec)
                else:
                    record[k] = val

        # ---- life_trajectory ----
        life_traj = js.get("life_trajectory", {})
        if isinstance(life_traj, dict):
            for age_str, subobj in life_traj.items():
                try:
                    age = int(age_str)
                except:
                    continue
                if not isinstance(subobj, dict):
                    continue
                for k in sequential_fields:
                    if k not in subobj:
                        continue
                    val = subobj[k]
                    spec = ALLOWED[k]
                    col = f"{k}_{age}"
                    if isinstance(spec, list):
                        record[col] = _validate_cat(val, spec)
                    elif isinstance(spec, dict) and spec.get("type") == "numeric":
                        record[col] = _validate_numeric(val, spec)
                    else:
                        record[col] = val

        if not _record_has_empty(record):
            return record

        last_record = record
        if attempt < MAX_RETRIES:
            print(
                f"[Retry] Profile {profile_id} contained empty values; retrying "
                f"({attempt + 1}/{MAX_RETRIES})..."
            )
            if RETRY_DELAY > 0:
                time.sleep(RETRY_DELAY)

    print(f"[Retry] Profile {profile_id} exhausted retries; returning last incomplete record.")
    return last_record if last_record is not None else record

# =====================================================
# Multithreaded generation
# =====================================================
print(f"🚀 Starting generation of {num_profiles} profiles using {max_workers} threads...")
print(f"🗂️ Output subfolder: {run_dir}")
rows = []
start = time.time()
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    future_to_id = {ex.submit(generate_single_profile, i): i for i in range(num_profiles)}
    for future in as_completed(future_to_id):
        pid = future_to_id[future]
        try:
            record = future.result()
            rows.append(record)
            print(f"✅ Completed profile {pid}")
        except Exception as e:
            print(f"❌ Profile {pid} failed: {e}")
end = time.time()
print(f"⏱️ Total generation time: {end - start:.2f}s")
print(f"📊 Generated {len(rows)} profiles successfully")
# DEBUG: find non-serializable fields

def np_to_py(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
# for r in rows:
#     for k, v in r.items():
#         try:
#             json.dumps(v)
#         except Exception as e:
#             print(f"[JSON ERROR] key='{k}', value='{v}', type={type(v)}")
#             raise e
# =====================================================
# Save intermediate JSONL
# =====================================================
with open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for r in rows:
        fout.write(json.dumps(r, default=np_to_py, ensure_ascii=False) + "\n")

df = pd.DataFrame(rows)
print(f"✅ Generated {len(df)} synthetic individuals")

# =====================================================
# Postprocessing
# =====================================================
def run_postprocess_modules(df, module_names):
    """
    Execut/e postprocessing functions defined in postprocess_utils.py
    according to the names listed in the config (e.g., age_completed_education).

    Each function must take a DataFrame and return a Series or a modified DataFrame.
    """
    import postprocess_utils

    if not module_names:
        print("[Postprocess] No postprocess modules specified in config.")
        return df

    for name in module_names:
        try:
            print(f"[Postprocess] Running module: {name}")

            # Check if the function exists in postprocess_utils
            func = getattr(postprocess_utils, f"compute_{name}", None)
            if func is None:
                print(f"[Postprocess] ⚠️ No function 'compute_{name}' found in postprocess_utils. Skipping.")
                continue

            # Execute the postprocess function
            result = func(df)

            # If the function returns a Series, append it as a new column
            if isinstance(result, pd.Series):
                df[name] = result
                print(f"[Postprocess] ✅ Added column '{name}'.")
            elif isinstance(result, pd.DataFrame):
                df = result
                print(f"[Postprocess] ✅ Updated DataFrame from '{name}'.")
            else:
                print(f"[Postprocess] ⚠️ Unsupported return type from '{name}'. Skipping.")

        except Exception as e:
            print(f"[Postprocess] ❌ Failed to run '{name}': {e}")

    print("[Postprocess] All modules completed.\n")
    return df


# Run postprocessing modules
df = run_postprocess_modules(df, post_modules)

# =====================================================
# Save final CSV
# =====================================================
df.to_csv(OUT_CSV, index=False)
print(f"📄 JSONL saved to: {OUT_JSONL}")
print(f"📊 CSV saved to:   {OUT_CSV}")
