#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from batch_eval import add_batch_arguments, run_batch

DEFAULT_SIM_ROOT = Path("./simulated_data/cgss2021_mate")
DEFAULT_OUTPUT_BASE = Path("./evaluation_results/cgss2021_mate")
DEFAULT_MASTER_CFG = Path("./evaluation/config/cgss2021_mate/evaluation_master.yaml")

def main():
    parser = argparse.ArgumentParser(description="Run CGSS2021 mate-preference evaluation across model folders.")
    add_batch_arguments(parser, default_sim_root=DEFAULT_SIM_ROOT, default_output_base=DEFAULT_OUTPUT_BASE, default_config=DEFAULT_MASTER_CFG)
    args = parser.parse_args()
    run_batch(sim_root=args.sim_root, output_base=args.output_base, config=args.config, sampled_prefix=args.sampled_prefix, sim_prefix=args.sim_prefix, eval_script=args.eval_script, single=args.single)

if __name__ == "__main__":
    main()
