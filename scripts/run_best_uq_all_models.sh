#!/usr/bin/env bash
# Freeze each validation winner into configs/best_uq/, then compare all UQ metrics with
# top-K and with Gumbel noise (ADR 0009). Models whose tuning has no winner yet are
# skipped. Set DATASETS to a subset, e.g. DATASETS="quishpi medical". Completed
# comparisons resume automatically. Pass --dry-run to preview without training or
# --seed-workers N to set concurrency.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

models=(distilbert bert_base roberta_base deberta_v3_base modernbert_base)

for dataset in ${DATASETS:-pet quishpi medical}; do
  suffix=$([[ $dataset == pet ]] && echo "" || echo "_$dataset")
  for model in "${models[@]}"; do
    frozen=$(uv run scripts/freeze_best_uq.py "configs/tune_random/${model}${suffix}_tune_random_100.json")
    [[ -n $frozen ]] || continue
    printf '\nBest-config UQ comparison: %s\n' "$frozen"
    uv run scripts/run_uq_metrics.py --config "$frozen" "$@"
    printf '\nBest-config UQ comparison with Gumbel noise: %s\n' "$frozen"
    uv run scripts/run_uq_metrics.py --config "$frozen" --gumbel-noise "$@"
  done
done
