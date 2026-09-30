#!/usr/bin/env bash
# Tune and evaluate the fully supervised baseline sequentially. Pass --dry-run to print plans.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

configs=(
  configs/supervised/distilbert.json
  configs/supervised/bert_base.json
  configs/supervised/roberta_base.json
  configs/supervised/deberta_v3_base.json
  configs/supervised/modernbert_base.json
)

for config in "${configs[@]}"; do
  printf '\nSupervised baseline: %s\n' "$config"
  uv run scripts/run_supervised.py --config "$config" "$@"
done
