# Test-tuned random oracle sweeps (appendix only)

These configurations feed an appendix sensitivity check, never a headline result
([ADR 0007](../../docs/adr/0007-random-is-tuned-on-validation-and-test-tuned-oracle-is-appendix-only.md)).
Do not copy a configuration selected from these results into `configs/best_uq/`.

Each file runs the existing `compare` mode with `sampler_seed: 0` and `num_configs: 100`,
which samples the same 100 configurations as the matching
`configs/tune_random/*_tune_random_100.json`. Unlike tuning, every configuration trains
on the full pool and evaluates both arms and all three UQ metrics on the test split,
with five model seeds run concurrently. W&B logging is off.

`notebooks/oracle_random_analysis.py` then picks the configuration with the highest
seed-mean random test entity-F1 AUC (the test-tuned random oracle) and reports every
UQ metric against random at that configuration. Because the selection uses the test
split, the oracle random arm is optimistically biased.

From the project root:

```bash
bash scripts/run_oracle_random_all_models.sh --dry-run
bash scripts/run_oracle_random_all_models.sh
```

Results go to `results/oracle_random/<model>-oracle-random-100/`. Re-running resumes
completed comparisons. Expect roughly 65–90 hours for all five checkpoints on the
machine that ran the tuning sweeps.
