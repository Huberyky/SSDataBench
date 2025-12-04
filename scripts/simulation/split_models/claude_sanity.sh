#!/bin/bash
# Sanity run for simulation using Claude via OpenRouter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

SIMULATION_SCRIPT="$PROJECT_ROOT/simulation/generation.py"
CONTENT_CONFIG="${SIM_CONTENT_CONFIG:-$PROJECT_ROOT/simulation/configs/content/nlsy79.yaml}"
PARAMS_CONFIG="${SIM_PARAMS_CONFIG:-$PROJECT_ROOT/simulation/configs/param/claude.yaml}"
OUTPUT_DIR="${SIM_OUTPUT_DIR:-$PROJECT_ROOT/simulated_data/nlsy}"
NUM_PROFILES="${NUM_PROFILES:-100}"
MAX_WORKERS="${MAX_WORKERS:-20}"

echo "Running Claude sanity simulation (${NUM_PROFILES} profiles)..."
echo "  script:    $SIMULATION_SCRIPT"
echo "  content:   $CONTENT_CONFIG"
echo "  params:    $PARAMS_CONFIG"
echo "  outdir:    $OUTPUT_DIR"

python "$SIMULATION_SCRIPT" \
    --config "$CONTENT_CONFIG" \
    --params-config "$PARAMS_CONFIG" \
    --n "$NUM_PROFILES" \
    --outdir "$OUTPUT_DIR" \
    --max-workers "$MAX_WORKERS"

echo "Done. Results saved to $OUTPUT_DIR"
