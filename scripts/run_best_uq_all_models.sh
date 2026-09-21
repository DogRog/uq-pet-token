#!/usr/bin/env bash
# Run one saved winner per model with shared metric training and concurrent seeds.
# Results are grouped by model and metric; completed comparisons resume automatically.
# Pass --dry-run to preview without training or --seed-workers N to set concurrency.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for config in configs/best_uq/*.json; do
  printf '\nBest-config UQ comparison: %s\n' "$config"
  uv run scripts/run_uq_metrics.py --config "$config" "$@"
done
