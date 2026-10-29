#!/usr/bin/env bash
# Run the training-recipe pilot (ADR 0012, criteria in configs/README.md) as a job queue on
# one GPU (scripts/run_queue.py): random-only validation tuning of the first 30
# configurations, with the bootstrap learning rate split off and clipping, then also with
# averaged models. Each job logs to logs/pilot/; rerun to resume.
# DATASETS and MODELS pick a subset (default PET and Quishpi, BERT and ModernBERT); SLOTS
# sets the jobs running at once (default 2). Extra arguments pass through, e.g. --dry-run.
# scripts/pilot_report.py reads the results.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run scripts/run_queue.py \
  --stages pilot \
  --datasets ${DATASETS:-pet quishpi} --models ${MODELS:-bert_base modernbert_base} \
  --slots "${SLOTS:-2}" --logs logs/pilot "$@"
printf 'Report: uv run scripts/pilot_report.py\n'
