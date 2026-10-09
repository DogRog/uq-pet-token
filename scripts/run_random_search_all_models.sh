#!/usr/bin/env bash
# Run the exploratory 30-configuration UQ-versus-random sweep for every dataset and
# checkpoint in sequence. Set DATASETS to a subset, e.g. DATASETS="quishpi medical".
# Pass --dry-run to print plans without training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

models=(distilbert bert_base roberta_base deberta_v3_base modernbert_base)

for dataset in ${DATASETS:-pet quishpi medical}; do
  for model in "${models[@]}"; do
    config=configs/$dataset/random_search/$model.json
    printf '\nRandom sweep: %s\n' "$config"
    uv run scripts/bert_token_uq_search.py --config "$config" "$@"
  done
done
