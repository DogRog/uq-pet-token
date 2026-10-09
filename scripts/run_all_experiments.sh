#!/usr/bin/env bash
# Run every paper experiment: one chain per dataset, the chains in parallel on one GPU.
# Each chain runs random-baseline tuning, best UQ (top-K and Gumbel noise) at the frozen
# winners, the supervised baseline, and the random sweep in order. The appendix's
# test-tuned random oracle (run_oracle_random_all_models.sh) and word form caps
# (run_stochastic_uq_all_models.sh) are not included.
#
# DATASETS picks the chains (default "pet quishpi medical"). SEED_WORKERS sets
# concurrent seeds per chain (default 2, about six seed processes in all). Output goes to
# logs/run_all/<dataset>.log. A failed chain stops alone; rerun to resume completed work.
# Extra arguments pass through to every stage, e.g. --dry-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

datasets=(${DATASETS:-pet quishpi medical})
workers=(--seed-workers "${SEED_WORKERS:-2}")
logs=logs/run_all
mkdir -p "$logs"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ " $* " != *" --dry-run "* ]]; then
  # Download every dataset once, before the chains race for it; fails early if the gated
  # medical dataset is not accessible.
  uv run python -c 'import sys; from uq_pet.data_prep import load_splits
for dataset in sys.argv[1:]: load_splits(dataset)' "${datasets[@]}"
fi

chain() {
  export DATASETS=$1
  shift
  bash scripts/tune_random_all_models.sh "$@"
  bash scripts/run_best_uq_all_models.sh "$@"
  bash scripts/run_supervised_all_models.sh "$@"
  bash scripts/run_random_search_all_models.sh "$@"
}

pids=()
for dataset in "${datasets[@]}"; do
  chain "$dataset" "${workers[@]}" "$@" >"$logs/$dataset.log" 2>&1 &
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
printf '\nAll chains finished.\n'
