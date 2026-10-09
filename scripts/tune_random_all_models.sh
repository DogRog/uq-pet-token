#!/usr/bin/env bash
# Run random-only tuning for every dataset and checkpoint in sequence. MedicalProcessInstruks
# is gated: accept its conditions on the Hub and run `uv run hf auth login` first.
# Set DATASETS to a subset, e.g. DATASETS="quishpi medical". Extra arguments pass
# through, e.g. --dry-run or --seed-workers 5.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

models=(distilbert bert_base roberta_base deberta_v3_base modernbert_base)

for dataset in ${DATASETS:-pet quishpi medical}; do
  suffix=$([[ $dataset == pet ]] && echo "" || echo "_$dataset")
  for model in "${models[@]}"; do
    config=configs/tune_random/${model}${suffix}_tune_random_100.json
    printf '\nRandom-only tuning: %s\n' "$config"
    uv run scripts/bert_token_uq_search.py --config "$config" "$@"
  done
done
