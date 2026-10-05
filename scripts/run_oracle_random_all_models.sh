#!/usr/bin/env bash
# Appendix only (ADR 0007): the test-tuned random oracle for each checkpoint, sequentially.
# Stage 1 tunes random only on the test split over the 100 tuning configurations and
# freezes oracle_config.json. Stage 2 compares all UQ metrics against random at that
# configuration. Completed work resumes. Pass --dry-run to print plans without training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

configs=(
  configs/oracle_random/distilbert_oracle_random_100.json
  configs/oracle_random/bert_base_oracle_random_100.json
  configs/oracle_random/roberta_base_oracle_random_100.json
  configs/oracle_random/deberta_v3_base_oracle_random_100.json
  configs/oracle_random/modernbert_base_oracle_random_100.json
)

for config in "${configs[@]}"; do
  printf '\nStage 1, test-split random tuning: %s\n' "$config"
  uv run scripts/bert_token_uq_search.py --config "$config" "$@"

  # Stage 2 reuses the frozen oracle with stage 1's W&B and concurrency settings.
  sweep=$(uv run python -c 'import json, sys; c = json.load(open(sys.argv[1])); print(c["sweeps_dir"] + "/" + c["sweep_name"])' "$config")
  if [[ ! -f "$sweep/oracle_config.json" ]]; then
    printf 'Stage 2 waits for %s/oracle_config.json from a completed stage 1.\n' "$sweep"
    continue
  fi
  settings=$(uv run python -c '
import json, sys
stage_1 = json.load(open(sys.argv[1]))
oracle = json.load(open(sys.argv[2]))
keep = ("wandb_enabled", "wandb_project", "seed_workers")
print(json.dumps({**oracle, **{key: stage_1[key] for key in keep if key in stage_1}}))
' "$config" "$sweep/oracle_config.json")
  printf '\nStage 2, UQ metrics at the oracle: %s\n' "$sweep/oracle_config.json"
  uv run scripts/run_uq_metrics.py --config-json "$settings" \
    --sweeps-dir results/oracle_random/uq --sweep-name "$(basename "$sweep")-uq" "$@"
done
