# Fixed configurations for UQ comparisons

Each JSON file is the validation winner of one random-baseline tuning sweep: the
`best_config.json` of the matching `configs/tune_random/<name>_tune_random_100.json`
search under `results/random_baseline_search/`, written by `scripts/freeze_best_uq.py`
with that sweep's W&B setting, a `<model>[-<dataset>]-best-uq` W&B project, and
`seed_workers: 5`. Winners maximize the seed-mean validation entity-F1 AUC
(`random_validation_entity_f1_auc`); test results never select them, and the test-tuned
oracle is never frozen here (ADR 0007). The winning trial and its validation score are
in the sweep's `selection.json`.

From the project root:

```bash
bash scripts/run_best_uq_all_models.sh --dry-run
bash scripts/run_best_uq_all_models.sh
```

The launcher re-freezes every winner before running, so a file here is rewritten
whenever its tuning sweep picks a new winner, and models whose tuning has no winner yet
are skipped. Commit the files after tuning so the frozen settings are recorded.

The checked-in PET files predate the learning-rate range of [1e-5, 5e-4]: they were
tuned over [1e-6, 1e-4] and are replaced once the rerun tuning completes.

Each frozen config runs entropy, least confidence, and margin with top-K selection and
again with Gumbel noise, keeping every saved setting fixed. Bootstrap and random-baseline
training are shared across metrics. Results are grouped under
`results/best_uq/<checkpoint>[-<dataset>]-best-uq[-gumbel]/`, with metric outputs in
`runs/config_0000/<metric>/`. Re-running skips completed comparisons. Use
`--seed-workers N` to change concurrency. Set `WANDB_API_KEY` in the environment or the
project's `.env` file before running. Completed comparisons are skipped and are not
uploaded retroactively.
