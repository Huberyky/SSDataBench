#!/usr/bin/env python3

"""Shared utilities for batch evaluation scripts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_EVAL_SCRIPT = Path("./evaluation/run_all_types.py")


def add_batch_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_sim_root: Path,
    default_output_base: Path,
    default_config: Path,
    default_sampled_prefix: str = "sampled_",
    default_sim_prefix: str = "sim_",
    default_eval_script: Path = DEFAULT_EVAL_SCRIPT,
) -> argparse.ArgumentParser:
    """Attach common CLI flags to a dataset-specific parser."""
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=default_sim_root,
        help="Directory containing simulation subfolders.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=default_output_base,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to evaluation master config.",
    )
    parser.add_argument(
        "--sampled-prefix",
        default=default_sampled_prefix,
        help="Filename prefix for sampled (real) CSVs.",
    )
    parser.add_argument(
        "--sim-prefix",
        default=default_sim_prefix,
        help="Filename prefix for simulated CSVs.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=default_eval_script,
        help="Path to evaluation/run_all_types.py (override if relocated).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Treat sim-root as a single run folder containing sampled/sim CSVs (no subfolders).",
    )
    return parser


def _pick_csv(subdir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(subdir.glob(f"{prefix}*.csv"))
    return candidates[0] if candidates else None


def _locate_pair(subdir: Path, sampled_prefix: str, sim_prefix: str) -> Optional[Tuple[Path, Path]]:
    sampled = _pick_csv(subdir, sampled_prefix)
    sim = _pick_csv(subdir, sim_prefix)
    if sampled and sim:
        return sampled, sim
    return None


def _run_eval(real_csv: Path, sim_csv: Path, output_dir: Path, config: Path, eval_script: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(eval_script),
        "--config",
        str(config),
        "--real_csv",
        str(real_csv),
        "--sim_csv",
        str(sim_csv),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def run_batch(
    *,
    sim_root: Path,
    output_base: Path,
    config: Path,
    sampled_prefix: str = "sampled_",
    sim_prefix: str = "sim_",
    eval_script: Path = DEFAULT_EVAL_SCRIPT,
    single: bool = False,
):
    """Scan model folders (or a single folder) for CSV pairs and run evaluation for each."""
    sim_root = sim_root.expanduser()
    output_base = output_base.expanduser()
    config = config.expanduser()
    eval_script = eval_script.expanduser()

    if not sim_root.exists():
        raise FileNotFoundError(f"Simulation root not found: {sim_root}")

    print(f"[BatchEval] scanning {sim_root}")
    processed = 0
    targets = []
    if single:
        targets.append(sim_root)
    else:
        for subdir in sorted(sim_root.iterdir()):
            if subdir.is_dir():
                targets.append(subdir)

    for subdir in targets:
        pair = _locate_pair(subdir, sampled_prefix, sim_prefix)
        if not pair:
            print(f"[Skip] missing CSV pair in {subdir}")
            continue
        real_csv, sim_csv = pair
        model_name = subdir.name.replace("sim_profiles_", "") if not single else subdir.name
        out_dir = output_base if single else output_base / model_name
        print(f"\n▶ Evaluating {model_name}")
        print(f"   real_csv = {real_csv}")
        print(f"   sim_csv  = {sim_csv}")
        print(f"   out_dir  = {out_dir}")
        try:
            _run_eval(real_csv, sim_csv, out_dir, config, eval_script)
            processed += 1
        except subprocess.CalledProcessError as exc:
            print(f"[Error] evaluation failed for {model_name}: {exc}")

    print(f"\n[BatchEval] completed {processed} model(s).")
    return processed
