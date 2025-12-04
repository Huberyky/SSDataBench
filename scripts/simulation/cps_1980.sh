#!/bin/bash
# Unified sanity run for simulation across all models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

SIMULATION_SCRIPT="$PROJECT_ROOT/simulation/generation_cs.py"
CONTENT_CONFIG="${SIM_CONTENT_CONFIG:-$PROJECT_ROOT/simulation/configs/content/cps_1980.yaml}"
PARAMS_DIR="$PROJECT_ROOT/simulation/configs/param"
OUTPUT_BASE="${SIM_OUTPUT_DIR:-$PROJECT_ROOT/simulated_data/cps_1980}"
NUM_PROFILES="${NUM_PROFILES:-5}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# list of models (corresponding to YAML filenames)
MODELS=(
  "claude"
  "deepseek"
  "gemini"
  "gpt4"
  "gpt4o"
  "gpt5"
  "gpt35"
  "grok"
  "qwen"
)

echo "=== Running sanity simulations for all models ==="
echo "Models: ${MODELS[*]}"
echo "Output base: $OUTPUT_BASE"
echo

for model in "${MODELS[@]}"; do
    PARAMS_CONFIG="$PARAMS_DIR/${model}.yaml"
    OUTPUT_DIR="${OUTPUT_BASE}"

    echo "------------------------------------------------------------"
    echo "▶ Running sanity simulation for model: ${model}"
    echo "  script:    $SIMULATION_SCRIPT"
    echo "  content:   $CONTENT_CONFIG"
    echo "  params:    $PARAMS_CONFIG"
    echo "  outdir:    $OUTPUT_DIR"
    echo "------------------------------------------------------------"

    python "$SIMULATION_SCRIPT" \
        --config "$CONTENT_CONFIG" \
        --params-config "$PARAMS_CONFIG" \
        --n "$NUM_PROFILES" \
        --outdir "$OUTPUT_DIR" \
        --max-workers "$MAX_WORKERS"

    echo "✅ Done: ${model}"
    echo
done

echo "=== All model sanity simulations completed ==="
