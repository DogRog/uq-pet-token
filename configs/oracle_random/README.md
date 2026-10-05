# Test-tuned random oracle sweeps (appendix only)

These configurations feed an appendix sensitivity check, never a headline result
([ADR 0007](../../docs/adr/0007-random-is-tuned-on-validation-and-test-tuned-oracle-is-appendix-only.md)).
Never copy an oracle configuration into `configs/best_uq/`.

Each file is a `tune-random` sweep with `"tuning_split": "test"`. With `sampler_seed: 0`
and `num_configs: 100` it samples the same 100 configurations as the matching
`configs/tune_random/*_tune_random_100.json`, but trains random only on the full pool and
scores each configuration by seed-mean test entity-F1 AUC (`random_test_entity_f1_auc`).
Five model seeds run concurrently, and W&B logs to one `<model>-oracle-random` project
per checkpoint.

The launcher runs two stages per checkpoint:

1. **Test-split random tuning** writes `oracle_config.json` (never `best_config.json`)
   and a `selection.json` with `"selection_split": "test"` under
   `results/oracle_random/<model>-oracle-random-100/`.
2. **UQ at the oracle** runs entropy, least confidence, and margin against random with
   those settings via `scripts/run_uq_metrics.py`, under
   `results/oracle_random/uq/<model>-oracle-random-100-uq/`.

From the project root:

```bash
bash scripts/run_oracle_random_all_models.sh --dry-run
bash scripts/run_oracle_random_all_models.sh
```

Set `WANDB_API_KEY` in the environment or the project's `.env` first. Re-running resumes
completed trials and comparisons. Expect roughly 28 hours for all five checkpoints on
the GPU that ran the tuning sweeps. `notebooks/oracle_random_analysis.py` reports the
results.
