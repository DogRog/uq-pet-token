#!/usr/bin/env bash
# Run DistilBERT random-only tuning on Quishpi, then MedicalProcessInstruks. Medical is
# gated: accept its conditions on the Hub and run `uv run hf auth login` first. Extra
# arguments pass through, e.g. --dry-run or --seed-workers 5.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

configs=(
  configs/tune_random/distilbert_quishpi_tune_random_100.json
  configs/tune_random/distilbert_medical_tune_random_100.json
)

for config in "${configs[@]}"; do
  printf '\nRandom-only tuning: %s\n' "$config"
  uv run scripts/bert_token_uq_search.py --config "$config" "$@"
done
