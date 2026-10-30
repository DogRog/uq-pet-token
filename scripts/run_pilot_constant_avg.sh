#!/usr/bin/env bash
# Run the averaged-model follow-up pilot (criteria in configs/README.md) as a job queue on
# one GPU (scripts/run_queue.py): random-only validation tuning of the first 30
# configurations with averaged models at the constant rate, then both arms on the same
# holdout at each winner, with and without averaging. Each job logs to logs/pilot/; rerun
# to resume. DATASETS and MODELS pick a subset (default PET and Quishpi, BERT and
# ModernBERT); SLOTS sets the jobs running at once (default 2). Extra arguments pass
# through, e.g. --dry-run. scripts/pilot_report.py reads the results.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run scripts/run_queue.py \
  --stages pilot_constant_avg pilot_gap \
  --datasets ${DATASETS:-pet quishpi} --models ${MODELS:-bert_base modernbert_base} \
  --slots "${SLOTS:-2}" --logs logs/pilot "$@"
printf 'Report: uv run scripts/pilot_report.py\n'
