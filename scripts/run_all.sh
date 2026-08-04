#!/usr/bin/env bash
#
# Run every config in configs/ back to back, unattended.
#
#   scripts/run_all.sh                        # all configs except smoke.yaml
#   scripts/run_all.sh configs/a.yaml ...     # just these, in this order
#   SKIP_DONE=1 scripts/run_all.sh            # skip configs that already have results/
#   DRY_RUN=1   scripts/run_all.sh            # print the plan and stop
#
# A failing config does not stop the sweep — the next one starts and the failure is
# reported in the closing summary. Exit status is the number of configs that failed.
#
# The preflight below is the point of the script: it loads and validates every config,
# checks the API keys, and refuses to start when two configs would share one score
# cache. Those are the failures that otherwise show up at 3am after the first run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=()
    for cfg in configs/*.yaml; do
        [ "$(basename "$cfg")" = "smoke.yaml" ] && continue
        CONFIGS+=("$cfg")
    done
fi

if [ "${SKIP_DONE:-0}" = "1" ]; then
    KEPT=()
    for cfg in "${CONFIGS[@]}"; do
        name="$(basename "$cfg" .yaml)"
        # Run dirs are results/<stem>_<timestamp>/; a metrics.json means one finished.
        if compgen -G "results/${name}_*/metrics.json" >/dev/null; then
            echo "skip (already has results): $cfg"
        else
            KEPT+=("$cfg")
        fi
    done
    CONFIGS=("${KEPT[@]:-}")
fi

[ -z "${CONFIGS[*]:-}" ] && { echo "Nothing to run."; exit 0; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/_sweeps/$STAMP"
mkdir -p "$LOG_DIR"

echo "Sweep $STAMP — ${#CONFIGS[@]} config(s), logs in $LOG_DIR"
printf '  %s\n' "${CONFIGS[@]}"
echo

# ---------------------------------------------------------------- preflight ----
# Reuses the package's own loader and arm validation, so a typo'd metric name or an
# unknown YAML key fails here rather than after hours of training.
uv run python - "${CONFIGS[@]}" <<'PY'
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from uq_pet.config import PROJECT_ROOT, load_config
from uq_pet.uncertainty import RANDOM, validate_arm

load_dotenv(PROJECT_ROOT / ".env")


def is_downloaded(checkpoint: str) -> bool:
    """True if the checkpoint is already in the HF cache, so the run needs no network."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        return True
    except Exception:
        return False


problems = []
caches = {}
checkpoints = {}
for path in (Path(p) for p in sys.argv[1:]):
    try:
        cfg = load_config(path)
    except Exception as e:  # unknown key, bad YAML, missing file
        problems.append(f"{path}: {type(e).__name__}: {e}")
        continue

    for arm in cfg.arms:
        try:
            validate_arm(arm.strategy, arm.params)
        except ValueError as e:
            problems.append(f"{path}: {e}")

    checkpoints.setdefault(cfg.train.checkpoint, []).append(path.name)
    cells = len(cfg.budget_pct) * len(cfg.arms) * len(cfg.train_seeds)

    needs_llm = any(arm.strategy != RANDOM for arm in cfg.arms)
    if not needs_llm:
        print(f"{path.name:34s} {cells:4d} cells  all-{RANDOM}: no cache, no API key")
        continue

    if not os.environ.get(cfg.llm.api_key_env):
        problems.append(f"{path}: {cfg.llm.api_key_env} is not set (env or .env)")

    cache = cfg.llm.cache_path("pool", tag=cfg.few_shot_tag())
    # Identity, not filename: two configs may legitimately share a cache, but only if
    # the records one writes are records the other would have written.
    recipe = (cfg.llm.model, tuple(sorted(cfg.llm.sampling_params().items())))
    caches.setdefault(cache, []).append((path, recipe))
    n = sum(1 for _ in cache.open()) if cache.exists() else 0
    print(f"{path.name:34s} {cells:4d} cells  {cache.name} ({n} records)")

for cache, users in caches.items():
    groups = {}
    for path, identity in users:
        groups.setdefault(identity, []).append(path.name)
    if len(groups) > 1:
        recipes = " vs. ".join("{" + ", ".join(names) + "}" for names in groups.values())
        problems.append(
            f"{cache.name} is claimed by {len(groups)} different sampling recipes: "
            f"{recipes} — each run evicts the other's records and re-scores the whole "
            f"pool. Give one of them its own llm.cache_prefix / llm.cache_suffix."
        )
    elif len(users) > 1:
        print(f"{'':34s}           ^ shared by {len(users)} configs, same recipe: scored once")

missing = [c for c in checkpoints if not is_downloaded(c)]
for checkpoint in missing:
    print(f"note: {checkpoint} is not in the HF cache yet — it downloads on first use")

# One 1-token generation per distinct model. A wrong model name is a 404 on every
# sentence of an uncached config, which is hours of the night spent failing.
if os.environ.get("SKIP_PROBE") != "1":
    import openai

    probed = {}
    for path in (Path(p) for p in sys.argv[1:]):
        try:
            cfg = load_config(path)
        except Exception:
            continue
        if all(arm.strategy == RANDOM for arm in cfg.arms):
            continue
        target = (cfg.llm.base_url, cfg.llm.model, cfg.llm.api_key_env)
        if target in probed:
            continue
        client = openai.OpenAI(
            base_url=cfg.llm.base_url,
            api_key=os.environ.get(cfg.llm.api_key_env, ""),
            timeout=30.0,
            max_retries=0,
        )
        try:
            client.chat.completions.create(
                model=cfg.llm.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            probed[target] = None
            print(f"probe ok  {cfg.llm.model}")
        except openai.APIStatusError as e:
            probed[target] = e
            problems.append(f"{cfg.llm.model} is unreachable: {e.status_code} {e.message[:120]}")
        except Exception as e:  # timeout, DNS, connection reset
            probed[target] = e
            print(f"probe WARN {cfg.llm.model}: {type(e).__name__} — the gateway may be flaky")

if problems:
    sys.stdout.flush()
    print("\nPreflight failed:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
PY
[ $? -ne 0 ] && { echo "Aborting: nothing was run."; exit 1; }

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1: plan checks out, stopping here."
    exit 0
fi

# ------------------------------------------------------------------- sweep ----
# caffeinate -i keeps macOS awake for the duration without touching the display.
RUNNER=()
command -v caffeinate >/dev/null && RUNNER=(caffeinate -i)

STATUSES=()
SWEEP_START=$SECONDS

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"
    log="$LOG_DIR/$name.log"
    start=$SECONDS
    echo "==> $(date '+%F %T')  $cfg  -> $log"

    "${RUNNER[@]}" uv run python -m uq_pet.main --config "$cfg" -v >"$log" 2>&1
    code=$?

    mins=$(( (SECONDS - start) / 60 ))
    if [ $code -eq 0 ]; then
        echo "    ok (${mins}m)"
        STATUSES+=("ok       ${mins}m  $name")
    else
        echo "    FAILED exit $code (${mins}m) — tail of $log:"
        tail -n 15 "$log" | sed 's/^/      /'
        STATUSES+=("FAIL($code) ${mins}m  $name")
    fi
done

# ----------------------------------------------------------------- summary ----
{
    echo
    echo "Sweep $STAMP finished at $(date '+%F %T') after $(( (SECONDS - SWEEP_START) / 60 ))m"
    printf '  %s\n' "${STATUSES[@]}"
} | tee "$LOG_DIR/summary.txt"

failed=$(printf '%s\n' "${STATUSES[@]}" | grep -c '^FAIL')
[ "$failed" -gt 0 ] && echo "  ($failed failed — logs in $LOG_DIR)"
exit "$failed"
