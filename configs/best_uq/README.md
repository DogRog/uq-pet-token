# Fixed configurations for UQ comparisons

Each JSON file preserves the winning experiment settings from `best_config.json` in the
corresponding completed search under `results/random_baseline_search/`, with
`seed_workers: 5` added to run all five seeds concurrently. All five searches completed 100 trials
and selected the winner by seed-mean validation entity-F1 AUC
(`random_validation_entity_f1_auc`). Test results were not used to select these settings.

| Configuration | Source search directory | Winning trial | Validation AUC |
| --- | --- | --- | --- |
| `bert_base.json` | `bert-base-random-baseline-100` | `config_0039` | 0.6652780764690579 |
| `deberta_v3_base.json` | `deberta-v3-base-random-baseline-100` | `config_0066` | 0.7178028042145446 |
| `distilbert.json` | `distilbert-random-baseline-100` | `config_0039` | 0.6554448758993312 |
| `modernbert_base.json` | `modernbert-base-random-baseline-100` | `config_0033` | 0.550755580155163 |
| `roberta_base.json` | `roberta-base-random-baseline-100` | `config_0039` | 0.7153730393814678 |

From the project root:

```bash
bash scripts/run_best_uq_all_models.sh --dry-run
bash scripts/run_best_uq_all_models.sh
```

The runner supplies entropy, least confidence, and margin, keeping every saved setting
fixed. Each comparison uses the original 5 seed / 328 pool / 84 test sentence split
and all five model seeds. The saved configs run five seeds concurrently; bootstrap and
random-baseline training are shared across metrics. Results and summaries are grouped
under `results/best_uq/<checkpoint>-best-uq/`, with metric outputs in
`runs/config_0000/<metric>/`. Re-running skips completed comparisons. Use
`--seed-workers N` to change concurrency or `--sweep-name NAME` for a fresh result group.
W&B logging remains disabled by default, as the saved winners contain no W&B settings.
