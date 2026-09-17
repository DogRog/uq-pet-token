#!/usr/bin/env bash
# Run random-only tuning sequentially. Pass --dry-run to print plans without training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

configs=(
  configs/tune_random/distilbert_tune_random_100.json
  configs/tune_random/bert_base_tune_random_100.json
  configs/tune_random/roberta_base_tune_random_100.json
  configs/tune_random/deberta_v3_base_tune_random_100.json
  configs/tune_random/modernbert_base_tune_random_100.json
)

for config in "${configs[@]}"; do
  printf '\nRandom-only tuning: %s\n' "$config"
  uv run scripts/bert_token_uq_search.py --config "$config" "$@"
done
