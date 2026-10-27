#!/usr/bin/env bash
# Run every paper experiment for ModernBERT only, e.g. after the byte-level BPE word-start
# fix (6e9c128). Same chains, logs, and resume behaviour as run_all_experiments.sh; extra
# arguments pass through, e.g. --dry-run.
set -euo pipefail

MODELS=modernbert_base exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all_experiments.sh" "$@"
