#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
python "$PROJECT_ROOT/simulation/generation_cgss_mate.py" \
  --config "${SIM_CONTENT_CONFIG:-$PROJECT_ROOT/simulation/configs/content/cgss2021_mate.yaml}" \
  --params-config "${SIM_PARAMS_CONFIG:-$PROJECT_ROOT/simulation/configs/param/gpt5.yaml}" \
  --n "${NUM_PROFILES:-200}" \
  --outdir "${SIM_OUTPUT_DIR:-$PROJECT_ROOT/simulated_data/cgss2021_mate}"
