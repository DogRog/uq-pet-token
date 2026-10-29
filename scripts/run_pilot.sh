#!/usr/bin/env bash
# Run the training-recipe pilot (ADR 0012, criteria in configs/README.md): random-only
# validation tuning of the first 30 configurations, with the bootstrap learning rate
# split off and clipping, then also with averaged models. One chain per dataset, the
# chains in parallel on one GPU; each logs to logs/pilot/<dataset>.log. Rerun to resume.
# DATASETS and MODELS pick a subset (default PET and Quishpi, BERT and ModernBERT).
# Extra arguments pass through, e.g. --dry-run. scripts/pilot_report.py reads the results.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

datasets=(${DATASETS:-pet quishpi})
models=(${MODELS:-bert_base modernbert_base})
variants=(split split_avg)
logs=logs/pilot
mkdir -p "$logs"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

chain() {
  local dataset=$1
  shift
  for variant in "${variants[@]}"; do
    for model in "${models[@]}"; do
      config=configs/$dataset/pilot/${model}_$variant.json
      printf '\nPilot: %s\n' "$config"
      uv run scripts/run_search.py --config "$config" "$@"
    done
  done
}

pids=()
for dataset in "${datasets[@]}"; do
  chain "$dataset" "$@" >"$logs/$dataset.log" 2>&1 &
  pids+=($!)
  printf 'Started %s (pid %d): %s\n' "$dataset" "$!" "$logs/$dataset.log"
done

failed=()
for index in "${!datasets[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'Finished %s\n' "${datasets[$index]}"
  else
    failed+=("${datasets[$index]}")
    printf 'FAILED %s; last lines of %s:\n' "${datasets[$index]}" "$logs/${datasets[$index]}.log"
    tail -n 20 "$logs/${datasets[$index]}.log"
  fi
done

if ((${#failed[@]})); then
  printf '\nFailed chains: %s. Rerun to resume them.\n' "${failed[*]}"
  exit 1
fi
printf '\nAll chains finished: uv run scripts/pilot_report.py\n'
