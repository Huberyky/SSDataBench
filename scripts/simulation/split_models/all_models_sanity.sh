#!/bin/bash
# Run simulation sanity checks for all supported models sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SANITY_SCRIPTS=(
  "gpt35_sanity.sh"
  "gemini_sanity.sh"
  "qwen_sanity.sh"
  "claude_sanity.sh"
  "deepseek_sanity.sh"
  "grok_sanity.sh"
)

for script in "${SANITY_SCRIPTS[@]}"; do
    echo ""
    echo "=============================="
    echo "Running $script"
    echo "=============================="
    bash "$SCRIPT_DIR/$script"
done

echo ""
echo "All simulation sanity checks completed."
