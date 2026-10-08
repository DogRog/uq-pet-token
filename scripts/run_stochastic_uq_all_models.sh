#!/usr/bin/env bash
# Run Gumbel noise and word form cap variants of every saved winner (ADR 0009).
# Results are grouped by model and variant; completed comparisons resume automatically.
# Pass --dry-run to preview without training or --seed-workers N to set concurrency.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

variants=(
  "--gumbel-noise"
  "--max-per-word-form 1"
  "--max-per-word-form 2"
  "--gumbel-noise --max-per-word-form 1"
)

for config in configs/best_uq/*.json; do
  for variant in "${variants[@]}"; do
    printf '\nStochastic UQ comparison: %s %s\n' "$config" "$variant"
    # shellcheck disable=SC2086  # each variant is a list of flags
    uv run scripts/run_uq_metrics.py --config "$config" $variant \
      --sweeps-dir results/best_uq_stochastic "$@"
  done
done
