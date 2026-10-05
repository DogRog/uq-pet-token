#!/usr/bin/env bash
# Run the test-tuned random oracle sweeps (appendix only; see ADR 0007) sequentially.
# Each sweep runs both arms and all three UQ metrics on the test split for the same 100
# configurations as random-baseline tuning. Pass --dry-run to print plans without training.
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
  printf '\nTest-tuned random oracle sweep: %s\n' "$config"
  uv run scripts/bert_token_uq_search.py --config "$config" "$@"
done
