#!/usr/bin/env bash
# Run every paper experiment as one job queue on one GPU (scripts/run_queue.py): random-
# baseline tuning, best UQ (top-K and Gumbel noise) at each frozen winner once its tuning
# finishes, the supervised baseline, and the random sweep, for every dataset and model.
# The appendix's test-tuned random oracle (run_oracle_random_all_models.sh) and word form
# caps (run_stochastic_uq_all_models.sh) are not included.
#
# DATASETS and MODELS pick a subset (default all three datasets and every model). SLOTS
# sets the jobs running at once (default 3) and SEED_WORKERS the concurrent seeds per job
# (default 2), about six seed processes in all. Each job logs to logs/run_all/. A failed
# job skips only the comparisons that need it; rerun to resume completed work. Extra
# arguments pass through to every job, e.g. --dry-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/models.sh
datasets=(${DATASETS:-pet quishpi medical})

if [[ " $* " != *" --dry-run "* ]]; then
  # Download every dataset once, before the jobs race for it; fails early if the gated
  # medical dataset is not accessible.
  uv run python -c 'import sys; from uq_pet.data_prep import load_splits
for dataset in sys.argv[1:]: load_splits(dataset)' "${datasets[@]}"
fi

exec uv run scripts/run_queue.py \
  --stages tune_random best_uq supervised random_search \
  --datasets "${datasets[@]}" --models "${models[@]}" \
  --slots "${SLOTS:-3}" --logs logs/run_all \
  --seed-workers "${SEED_WORKERS:-2}" "$@"
