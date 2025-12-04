#!/bin/bash
# Plot strength comparisons for types 2/3/5 (structure disabled).
# Optionally set VIS_ROOT_DIR to point at your evaluation results root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
# Allow optional root as first positional; otherwise default.
ROOT_DIR="evaluation_results"
if [[ $# -gt 0 && $1 != --* ]]; then
  ROOT_DIR="$1"
  shift
fi

python "$PROJECT_ROOT/visualization/strength.py" --root "$ROOT_DIR" "$@"
