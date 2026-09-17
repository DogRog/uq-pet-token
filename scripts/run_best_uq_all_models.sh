#!/usr/bin/env bash
# Run all saved winners with all UQ metrics. Pass --dry-run to preview without training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for config in configs/best_uq/*.json; do
  printf '\nBest-config UQ comparison: %s\n' "$config"
  uv run scripts/run_uq_metrics.py --config "$config" "$@"
done
