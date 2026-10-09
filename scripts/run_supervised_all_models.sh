#!/usr/bin/env bash
# Tune and evaluate the fully supervised baseline for every dataset and checkpoint in
# sequence. Set DATASETS to a subset, e.g. DATASETS="quishpi medical". Pass --dry-run to
# print plans.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

models=(distilbert bert_base roberta_base deberta_v3_base modernbert_base)

for dataset in ${DATASETS:-pet quishpi medical}; do
  for model in "${models[@]}"; do
    config=configs/$dataset/supervised/$model.json
    printf '\nSupervised baseline: %s\n' "$config"
    uv run scripts/run_supervised.py --config "$config" "$@"
  done
done
