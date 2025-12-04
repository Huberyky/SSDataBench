#!/usr/bin/env python3

"""Batch evaluation helper for CPS 1980 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_eval import add_batch_arguments, run_batch

DEFAULT_SIM_ROOT = Path("/home/yx5888/Workspace/AI4SS-Evaluation/simulated_data_1117/cps_1980")
DEFAULT_OUTPUT_BASE = Path("./evaluation_results/cps_1980")
DEFAULT_MASTER_CFG = Path("./evaluation/config/cps_1980/evaluation_master.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run CPS 1980 evaluation across multiple model folders.")
    add_batch_arguments(
        parser,
        default_sim_root=DEFAULT_SIM_ROOT,
        default_output_base=DEFAULT_OUTPUT_BASE,
        default_config=DEFAULT_MASTER_CFG,
    )
    args = parser.parse_args()

    run_batch(
        sim_root=args.sim_root,
        output_base=args.output_base,
        config=args.config,
        sampled_prefix=args.sampled_prefix,
        sim_prefix=args.sim_prefix,
        eval_script=args.eval_script,
        single=args.single,
    )


if __name__ == "__main__":
    main()
