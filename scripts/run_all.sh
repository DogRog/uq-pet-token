#!/usr/bin/env bash
# Run BERT-UQ configs sequentially. No gateway, API key, or score cache is involved.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(configs/bert_*.yaml)
fi

if [ "${SKIP_DONE:-0}" = "1" ]; then
    KEPT=()
    for cfg in "${CONFIGS[@]}"; do
        name="$(basename "$cfg" .yaml)"
        if compgen -G "results/${name}_*/metrics.json" >/dev/null; then
            echo "skip (already complete): $cfg"
        else
            KEPT+=("$cfg")
        fi
    done
    CONFIGS=("${KEPT[@]:-}")
fi

[ -z "${CONFIGS[*]:-}" ] && { echo "Nothing to run."; exit 0; }

# Validate every YAML file and uncertainty arm before loading any model weights.
uv run python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

from uq_pet.config import load_config
from uq_pet.uncertainty import validate_arm

for path in (Path(value) for value in sys.argv[1:]):
    cfg = load_config(path)
    for arm in cfg.arms:
        validate_arm(arm.strategy, arm.params)
    cells = len(cfg.budget_pct) * len(cfg.arms) * len(cfg.model_seeds)
    print(
        f"{path.name:24s} {cells:4d} cells  checkpoint={cfg.train.checkpoint} "
        f"bootstrap_epochs={cfg.uq.bootstrap_epochs}"
    )
PY
[ $? -ne 0 ] && { echo "Preflight failed; nothing was run."; exit 1; }

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: preflight passed."
    exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/_sweeps/$STAMP"
mkdir -p "$LOG_DIR"
failures=0

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"
    log="$LOG_DIR/$name.log"
    echo "==> $cfg"
    uv run python -m uq_pet.main --config "$cfg" >"$log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        echo "    ok -> $log"
    else
        echo "    FAILED ($code) -> $log"
        failures=$((failures + 1))
    fi
done

exit "$failures"
