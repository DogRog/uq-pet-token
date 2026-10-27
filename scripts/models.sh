# Sourced by the run_*_all_models.sh launchers: every checkpoint they run, in order, named
# as the config files under configs/<dataset>/<stage>/. Set MODELS to a subset, e.g.
# MODELS="modernbert_base".
# shellcheck disable=SC2034  # used by the scripts that source this file
models=(${MODELS:-distilbert bert_base roberta_base deberta_v3_base modernbert_base})
