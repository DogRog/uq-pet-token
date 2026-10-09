#!/usr/bin/env bash
# Run word form cap variants of every validation winner (ADR 0009); Gumbel noise alone
# runs in run_best_uq_all_models.sh. Winners are frozen into configs/best_uq/ first, and
# models whose tuning has no winner yet are skipped. Set DATASETS to a subset, e.g.
# DATASETS=pet. Completed comparisons resume automatically. Pass --dry-run to preview
# without training or --seed-workers N to set concurrency.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

models=(distilbert bert_base roberta_base deberta_v3_base modernbert_base)
variants=(
  "--max-per-word-form 1"
  "--max-per-word-form 2"
  "--gumbel-noise --max-per-word-form 1"
)

for dataset in ${DATASETS:-pet quishpi medical}; do
  suffix=$([[ $dataset == pet ]] && echo "" || echo "_$dataset")
  for model in "${models[@]}"; do
    frozen=$(uv run scripts/freeze_best_uq.py "configs/tune_random/${model}${suffix}_tune_random_100.json")
    [[ -n $frozen ]] || continue
    for variant in "${variants[@]}"; do
      printf '\nStochastic UQ comparison: %s %s\n' "$frozen" "$variant"
      # shellcheck disable=SC2086  # each variant is a list of flags
      uv run scripts/run_uq_metrics.py --config "$frozen" $variant \
        --sweeps-dir results/best_uq_stochastic "$@"
    done
  done
done
