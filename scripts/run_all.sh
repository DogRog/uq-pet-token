#!/usr/bin/env bash
#
# Run every config in configs/ back to back, unattended.
#
#   scripts/run_all.sh                        # all configs except smoke.yaml
#   scripts/run_all.sh configs/a.yaml ...     # just these, in this order
#   SKIP_DONE=1 scripts/run_all.sh            # skip configs that already have results/
#   DRY_RUN=1   scripts/run_all.sh            # print the plan and stop
#   JOBS=4      scripts/run_all.sh            # up to 4 configs at once
#
# A failing config does not stop the sweep — the next one starts and the failure is
# reported in the closing summary. Nor does a config the preflight finds unrunnable
# (an unloadable checkpoint, say): it is dropped, named, and the others still run.
# Exit status is the number of configs that failed plus the number skipped.
#
# JOBS parallelises *across score caches, never within one*. Configs that share a cache
# file run in sequence with each other however high JOBS goes: an unscored cache would
# otherwise be filled by several runs at once, which is N times the API bill and, since
# a record is ~85KB and appends that large interleave, a corrupt file at the end of it.
# So the unit of parallelism is a chain of configs sharing one cache, not a config.
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
CACHE_GROUPS="$LOG_DIR/cache_groups.tsv"
uv run python - "$CACHE_GROUPS" "${CONFIGS[@]}" <<'PY'
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from uq_pet.config import PROJECT_ROOT, load_config
from uq_pet.uncertainty import RANDOM, validate_arm

load_dotenv(PROJECT_ROOT / ".env")


def checkpoint_status(checkpoint: str) -> tuple[bool, Exception | None]:
    """(already in the HF cache, why the tokenizer cannot be built).

    Building the tokenizer is the part that fails for reasons a config file cannot
    show: DeBERTa-v3's needs `tiktoken` to convert its vocabulary, and a missing
    converter raises here rather than at `from_pretrained` of the config. A few
    seconds of downloading now beats finding out an hour into a sweep.
    """
    from transformers import AutoConfig, AutoTokenizer

    try:
        AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        cached = True
    except Exception:
        cached = False
    try:
        AutoTokenizer.from_pretrained(checkpoint)
    except Exception as e:
        return cached, e
    return cached, None


groups_path = Path(sys.argv[1])
skipped_path = groups_path.with_name("skipped.tsv")
problems = []
# Reasons to drop one config but run the rest. A sweep is unattended, so a fault in
# one config is worth reporting and stepping over, not worth spending the night on.
unrunnable = {}
caches = {}
checkpoints = {}
# config -> the cache file it writes to, which is what may not be written twice at
# once. A config with no cache is its own group and parallelises freely.
group_of = {}
for path in (Path(p) for p in sys.argv[2:]):
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

    checkpoints.setdefault(cfg.train.checkpoint, []).append(path)
    cells = len(cfg.budget_pct) * len(cfg.arms) * len(cfg.train_seeds)

    needs_llm = any(arm.strategy != RANDOM for arm in cfg.arms)
    if not needs_llm:
        group_of[path] = str(path)
        print(f"{path.name:34s} {cells:4d} cells  all-{RANDOM}: no cache, no API key")
        continue

    if not os.environ.get(cfg.llm.api_key_env):
        problems.append(f"{path}: {cfg.llm.api_key_env} is not set (env or .env)")

    cache = cfg.llm.cache_path("pool", tag=cfg.few_shot_tag())
    # Identity, not filename: two configs may legitimately share a cache, but only if
    # the records one writes are records the other would have written.
    recipe = (cfg.llm.model, tuple(sorted(cfg.llm.sampling_params().items())))
    caches.setdefault(cache, []).append((path, recipe))
    group_of[path] = str(cache)
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

for checkpoint, users in checkpoints.items():
    cached, error = checkpoint_status(checkpoint)
    if error is not None:
        reason = f"{checkpoint} tokenizer will not load — {type(error).__name__}: {str(error)[:200]}"
        for path in users:
            unrunnable[path] = reason
        print(f"SKIP {', '.join(p.name for p in users)}: {reason}")
    elif not cached:
        print(f"note: {checkpoint} is not in the HF cache yet — it downloads on first use")

# One 1-token generation per distinct model. A wrong model name is a 404 on every
# sentence of an uncached config, which is hours of the night spent failing.
if os.environ.get("SKIP_PROBE") != "1":
    import openai

    probed = {}
    for path in (Path(p) for p in sys.argv[2:]):
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

skipped_path.write_text("".join(f"{path}\t{why}\n" for path, why in unrunnable.items()))

# In argv order, so the sweep runs the configs in the order it was asked to.
groups_path.write_text(
    "".join(
        f"{path}\t{group_of.get(path, path)}\n"
        for path in (Path(p) for p in sys.argv[2:])
        if path not in unrunnable
    )
)
PY
[ $? -ne 0 ] && { echo "Aborting: nothing was run."; exit 1; }

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1: plan checks out, stopping here."
    exit 0
fi

# ------------------------------------------------------------------- sweep ----
# caffeinate -i keeps macOS awake for the duration without touching the display. It
# does not exist on the Linux boxes this also runs on, hence the guard.
RUNNER=()
command -v caffeinate >/dev/null && RUNNER=(caffeinate -i)

JOBS="${JOBS:-1}"
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"
SWEEP_START=$SECONDS

if [ "$JOBS" -gt 1 ]; then
    # Each config is a process that would otherwise take every core for itself, and
    # JOBS of them thrashing is slower than one. Torch's default is the core count.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export TOKENIZERS_PARALLELISM=false

    # Warm the HF cache serially first: parallel chains would otherwise race to
    # download the same checkpoint, and a name typo is better found now.
    echo "Prefetching checkpoints..."
    # Only the configs that survived the preflight: the ones it dropped are dropped
    # for reasons that would just be re-raised here, noisily.
    RUNNABLE=()
    while IFS= read -r line; do RUNNABLE+=("$line"); done < <(cut -f1 "$CACHE_GROUPS")
    uv run python - "${RUNNABLE[@]:-}" <<'PY'
import sys
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

from uq_pet.config import load_config
from uq_pet.model_training import configure_hf_logging

# Every checkpoint is loaded with a fresh classifier head, so the missing-weights
# report is expected noise here exactly as it is during a run.
configure_hf_logging(quiet=True)

for checkpoint in dict.fromkeys(load_config(Path(p)).train.checkpoint for p in sys.argv[1:]):
    AutoTokenizer.from_pretrained(checkpoint)
    AutoModelForTokenClassification.from_pretrained(checkpoint)
    print(f"  {checkpoint}")
PY
    # Not fatal: one unfetchable checkpoint costs its own configs, and taking the
    # whole night down with it would be the more expensive failure. The preflight
    # already refused the checkpoints that are broken rather than merely missing.
    [ $? -ne 0 ] && echo "  WARN: a checkpoint could not be prefetched — its configs will fail"
fi

# Kill the whole process group, not just the config in front: with JOBS>1 there are
# several children, and a half-stopped sweep is worse than either outcome.
trap 'echo; echo "Interrupted — stopping the sweep. Logs in $LOG_DIR"; kill 0; exit 130' INT TERM

run_one() {
    local cfg="$1" name log start code mins
    name="$(basename "$cfg" .yaml)"
    log="$LOG_DIR/$name.log"
    start=$SECONDS
    echo "==> $(date '+%F %T')  $cfg  -> $log"

    "${RUNNER[@]}" uv run python -m uq_pet.main --config "$cfg" -v >"$log" 2>&1
    code=$?

    mins=$(( (SECONDS - start) / 60 ))
    if [ $code -eq 0 ]; then
        echo "    ok (${mins}m)  $name"
        printf 'ok       %sm  %s\n' "$mins" "$name" >"$STATUS_DIR/$name"
    else
        echo "    FAILED exit $code (${mins}m) — tail of $log:"
        tail -n 15 "$log" | sed 's/^/      /'
        printf 'FAIL(%s) %sm  %s\n' "$code" "$mins" "$name" >"$STATUS_DIR/$name"
    fi
    return 0
}

# One chain per cache file, in the order the configs were given. Configs inside a
# chain run one after another; chains run against each other, JOBS at a time.
CHAINS=()
while IFS= read -r chain; do
    [ -n "$chain" ] && CHAINS+=("$chain")
done < <(awk -F'\t' '
    { if (!($2 in seen)) { seen[$2] = ++n; order[n] = $2 }
      chain[$2] = chain[$2] " " $1 }
    END { for (i = 1; i <= n; i++) print substr(chain[order[i]], 2) }
' "$CACHE_GROUPS")

if [ "$JOBS" -gt 1 ]; then
    echo "Running ${#CHAINS[@]} chain(s), $JOBS at a time"
    echo
fi

for chain in "${CHAINS[@]:-}"; do
    [ -z "$chain" ] && continue
    if [ "$JOBS" -le 1 ]; then
        for cfg in $chain; do run_one "$cfg"; done
        continue
    fi
    # `jobs -rp` counts only the still-running children, and works in bash 3.2 —
    # `wait -n` would be tidier but is bash 4.3+.
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    ( for cfg in $chain; do run_one "$cfg"; done ) &
done
wait

# ----------------------------------------------------------------- summary ----
SKIPPED="$LOG_DIR/skipped.tsv"
STATUSES=()
for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"
    if [ -f "$STATUS_DIR/$name" ]; then
        STATUSES+=("$(cat "$STATUS_DIR/$name")")
    elif reason="$(grep -m1 -F "$cfg	" "$SKIPPED" 2>/dev/null | cut -f2-)" && [ -n "$reason" ]; then
        STATUSES+=("SKIPPED     $name — $reason")
    else
        STATUSES+=("NORUN       $name")
    fi
done

{
    echo
    echo "Sweep $STAMP finished at $(date '+%F %T') after $(( (SECONDS - SWEEP_START) / 60 ))m"
    printf '  %s\n' "${STATUSES[@]}"
} | tee "$LOG_DIR/summary.txt"

failed=$(printf '%s\n' "${STATUSES[@]}" | grep -c '^FAIL')
skipped=$(printf '%s\n' "${STATUSES[@]}" | grep -c '^SKIPPED')
[ "$failed" -gt 0 ] && echo "  ($failed failed — logs in $LOG_DIR)"
[ "$skipped" -gt 0 ] && echo "  ($skipped skipped by the preflight and never started)"
exit $(( failed + skipped ))
