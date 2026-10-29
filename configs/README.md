# Experiment configurations

Configs are grouped by dataset, then by stage, with one file per model:
`configs/<dataset>/<stage>/<model>.json`. Datasets are `pet`, `quishpi`, and `medical`
(MedicalProcessInstruks); models are `distilbert`, `bert_base`, `roberta_base`,
`deberta_v3_base`, and `modernbert_base`.

| Stage | Contents | Launcher |
| --- | --- | --- |
| `tune_random/` | random-baseline tuning on the validation holdout | `scripts/tune_random_all_models.sh` |
| `best_uq/` | frozen validation winners for the UQ comparisons | `scripts/run_best_uq_all_models.sh` |
| `supervised/` | the supervised baseline | `scripts/run_supervised_all_models.sh` |
| `random_search/` | the exploratory 30-configuration UQ-versus-random sweep | `scripts/run_random_search_all_models.sh` |
| `oracle_random/` | PET only: the appendix's test-tuned random oracle ([README](pet/oracle_random/README.md)) | `scripts/run_oracle_random_all_models.sh` |
| `pilot/` | PET and Quishpi, BERT and ModernBERT: the training-recipe pilot ([below](#training-recipe-pilot-in-pilot)) | `scripts/run_pilot.sh` |

`scripts/run_all_experiments.sh` runs every stage except the oracle for all three
datasets. Each dataset's `tune_random/` and `supervised/` files share one validation
holdout (66 sentences on PET, 31 on Quishpi, 51 on MedicalProcessInstruks).

## Frozen winners in `best_uq/`

Each `best_uq/<model>.json` is the validation winner of `tune_random/<model>.json` in the
same dataset folder: its sweep's `best_config.json` under
`results/random_baseline_search/`, written by `scripts/freeze_best_uq.py` with that
sweep's W&B setting, a `<model>[-<dataset>]-best-uq` W&B project, and `seed_workers: 5`.
Winners maximize the seed-mean validation entity-F1 AUC (`random_validation_entity_f1_auc`);
test results never select them, and the test-tuned oracle is never frozen here
(ADR 0007). The winning trial and its validation score are in the sweep's
`selection.json`.

The best-UQ launcher re-freezes every winner before running, so a file is rewritten
whenever its tuning sweep picks a new winner, and models whose tuning has no winner yet
are skipped. Commit the files after tuning so the frozen settings are recorded.

The checked-in `pet/best_uq/` files predate the learning-rate range of [1e-5, 5e-4]: they
were tuned over [1e-6, 1e-4] and are replaced once the rerun tuning completes. Quishpi
and MedicalProcessInstruks get their `best_uq/` folders from their first completed
tuning.

Each frozen config runs entropy, least confidence, and margin with top-K selection and
again with Gumbel noise, keeping every saved setting fixed. Results are grouped under
`results/best_uq/<checkpoint>[-<dataset>]-best-uq[-gumbel]/`, with metric outputs in
`runs/config_0000/<metric>/`. Re-running skips completed comparisons.

## Training-recipe pilot in `pilot/`

The pilot decides whether the paper rerun uses the training settings of ADR 0012. Each
`pilot/<model>_<variant>.json` tunes random selection on the validation holdout again,
over the first 30 configurations of `tune_random/<model>.json`: the same sampler seed,
holdout, and seeds, so each configuration pairs with a finished constant-rate trial.
Both variants clip gradients at norm 1 and sample a bootstrap learning rate per
configuration (`tune_bootstrap_learning_rate`); `split_avg` also turns on averaged
models. The test split is never read. Outputs go to `results/pilot/`, W&B stays off, and
`scripts/pilot_report.py` refuses any pilot sweep that does not pair with its tuning
sweep.

These criteria were fixed before any pilot run. For each setting and recipe, over all
150 runs (30 configurations, five seeds):

- **Drop share**: the share of runs whose entity F1 falls 0.10 or more below their best so
  far, from round 1 on.
- **Median jitter**: the median over runs of each round's mean distance from the
  midpoint of its two neighbours.
- **Failed bootstraps**: the share of runs with round-0 entity F1 below 0.05.
- **Winner**: the configuration with the best seed-mean validation entity-F1 AUC (ties to
  the lowest ID), its AUC, and the standard deviation of its final F1 across seeds.

The split improves on the constant rate in a setting when its drop share is lower and its
winner's AUC is at most 0.01 below the constant rate's. Averaging improves on the split in
the same way, compared with `split`. Once all four settings are complete:

1. Adopt the split bootstrap and clipping if the split improves in at least 3 of the 4
   settings; otherwise keep the constant learning rate.
2. Also adopt averaged models only if the split was adopted and averaging improves in at
   least 3 of the 4 settings.

Drop share decides because median jitter falls under averaging by construction; jitter,
failed bootstraps, and the winner's seed spread are reported to explain the result.
Quishpi's 31-sentence holdout makes its numbers noisy, which is why no single setting
decides.
