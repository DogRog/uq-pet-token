#!/usr/bin/env bash
# Run random-only tuning for every dataset and checkpoint in sequence. MedicalProcessInstruks
# is gated: accept its conditions on the Hub and run `uv run hf auth login` first.
# Set DATASETS to a subset, e.g. DATASETS="quishpi medical". Extra arguments pass
# through, e.g. --dry-run or --seed-workers 5.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/models.sh

for dataset in ${DATASETS:-pet quishpi medical}; do
  for model in "${models[@]}"; do
    config=configs/$dataset/tune_random/$model.json
    printf '\nRandom-only tuning: %s\n' "$config"
    uv run scripts/run_search.py --config "$config" "$@"
  done
done
