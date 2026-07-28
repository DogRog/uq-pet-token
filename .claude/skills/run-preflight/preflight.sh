#!/usr/bin/env bash
# Pre-push gate for uq-pet: ruff lint + format, offline tests, push-safety scan.
#
#   .claude/skills/run-preflight/preflight.sh [--fix] [--no-tests] [--strict]
#
# --fix       autofix lint + reformat PKG_PATHS first, then re-check
# --no-tests  skip pytest (lint/format/safety only)
# --strict    notebook lint/format failures become blocking too
#
# Exit 0 = safe to push. Exit 1 = at least one blocking check failed.

set -uo pipefail

# Run from the repo root no matter where the caller is.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 1

PKG_PATHS=(src tests)          # blocking scope
NB_PATHS=(notebooks)           # advisory scope (see --strict)

FIX=0; RUN_TESTS=1; STRICT=0
for arg in "$@"; do
  case "$arg" in
    --fix)      FIX=1 ;;
    --no-tests) RUN_TESTS=0 ;;
    --strict)   STRICT=1 ;;
    -h|--help)  sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

FAILED=()
WARNED=()

hdr() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()  { printf '\033[32mPASS\033[0m  %s\n' "$1"; }
bad() { printf '\033[31mFAIL\033[0m  %s\n' "$1"; FAILED+=("$1"); }
warn(){ printf '\033[33mWARN\033[0m  %s\n' "$1"; WARNED+=("$1"); }

# ---------------------------------------------------------------- 0. autofix
if [ "$FIX" = 1 ]; then
  hdr "autofix (${PKG_PATHS[*]})"
  uv run ruff check --fix "${PKG_PATHS[@]}"
  uv run ruff format "${PKG_PATHS[@]}"
fi

# ------------------------------------------------------------ 1. lint/format
hdr "ruff lint — ${PKG_PATHS[*]}"
if uv run ruff check --output-format concise "${PKG_PATHS[@]}"; then
  ok "ruff check ${PKG_PATHS[*]}"
else
  bad "ruff check ${PKG_PATHS[*]}  (rerun with --fix, or fix by hand)"
fi

hdr "ruff format --check — ${PKG_PATHS[*]}"
if uv run ruff format --check "${PKG_PATHS[@]}"; then
  ok "ruff format ${PKG_PATHS[*]}"
else
  bad "ruff format ${PKG_PATHS[*]}  (rerun with --fix)"
fi

# Notebooks: ruff lints .ipynb by default. Cell-level E402 is normal in
# exploratory notebooks, so this is advisory unless --strict.
hdr "ruff — ${NB_PATHS[*]} (advisory)"
nb_bad=0
uv run ruff check --output-format concise "${NB_PATHS[@]}" || nb_bad=1
uv run ruff format --check "${NB_PATHS[@]}" || nb_bad=1
if [ "$nb_bad" = 0 ]; then
  ok "ruff ${NB_PATHS[*]}"
elif [ "$STRICT" = 1 ]; then
  bad "ruff ${NB_PATHS[*]} (--strict)"
else
  warn "ruff ${NB_PATHS[*]} — not blocking; see SKILL.md > Gotchas"
fi

# ------------------------------------------------------------------ 2. tests
if [ "$RUN_TESTS" = 1 ]; then
  hdr "pytest (offline)"
  # --tb=line keeps a failing gate readable: one line per failure, not 200.
  if uv run pytest -q --tb=line; then
    ok "pytest"
  else
    bad "pytest"
  fi
fi

# ---------------------------------------------------- 3. push-safety scan
hdr "push safety"

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  bad ".env is TRACKED — git rm --cached .env before pushing"
else
  ok ".env not tracked"
fi

# Live-key shapes for the providers this repo talks to (OpenRouter, OpenAI, HF).
KEYPAT='sk-or-v1-[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,}'
hits=$(git ls-files -z | xargs -0 grep -lE "$KEYPAT" 2>/dev/null)
if [ -n "$hits" ]; then
  bad "possible API key in tracked files:"; printf '        %s\n' $hits
else
  ok "no API-key patterns in tracked files"
fi

# Anything tracked out of data/ or results/ is a run artifact that should not
# be in the repo (both are gitignored, but `git add -f` bypasses that).
artifacts=$(git ls-files -- data results)
if [ -n "$artifacts" ]; then
  bad "run artifacts are tracked:"; printf '        %s\n' $artifacts
else
  ok "no data/ or results/ artifacts tracked"
fi

big=$(git ls-files -z | xargs -0 du -k 2>/dev/null | awk '$1 > 1024 {print $1"K\t"$2}')
if [ -n "$big" ]; then
  warn "tracked files over 1 MB:"; printf '        %s\n' "$big"
else
  ok "no tracked file over 1 MB"
fi

# The prompt file is fingerprinted into every LLM score cache; if it is missing
# from the tree, score-pool rejects the caches and 6 tests fail.
if [ -f prompts/ner_v1.txt ]; then
  ok "prompts/ner_v1.txt present"
else
  bad "prompts/ner_v1.txt MISSING — see SKILL.md > Troubleshooting"
fi

# ----------------------------------------------------------------- 4. verdict
hdr "verdict"
for w in "${WARNED[@]:-}"; do [ -n "$w" ] && printf '\033[33m  warn:\033[0m %s\n' "$w"; done
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mready to push\033[0m\n'
  exit 0
fi
for f in "${FAILED[@]}"; do printf '\033[31m  fail:\033[0m %s\n' "$f"; done
printf '\033[31mdo NOT push yet\033[0m\n'
exit 1
