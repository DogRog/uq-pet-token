# The bootstrap has its own learning rate, and arms report averaged models

One learning rate used to serve two jobs. The bootstrap is five sentences, so at batch size 32 or 64 its 10 epochs are about 10 updates; in the 2026-10 rerun's tuning sweeps (21,900 runs), round-0 entity F1 stayed below 0.05 in 82% of runs below 3e-5, against 14% at 2e-4 and above. Tuning therefore picks high rates, and the rounds then run at them: a 10-point drop from a run's best F1 so far occurred in 35% of runs below 3e-5 and in 82–84% above 1e-4. Most of those dips were not divergence. Most one-round drops of more than 5 points recovered within three rounds, and only 13% of drops of more than 15 points came with a training loss twice its recent median.

Three optional settings address this, each off by default so that earlier plans, run names, and results are unchanged:

- `bootstrap_learning_rate` gives the bootstrap its own peak rate, warmed up over the first tenth of its updates and decayed linearly to zero. Each arm's optimizer keeps the bootstrap's AdamW history but restarts at the constant round `learning_rate`. Sweeps with `tune_bootstrap_learning_rate` sample it log-uniformly over the same range as `learning_rate`, from a separate random stream, so every other sampled setting matches the untuned sweep with the same sampler seed.
- `weight_averaging` keeps an averaged copy of each arm, starting from the bootstrap weights and updated after every round with half of its weight kept (`WEIGHT_AVERAGE_DECAY = 0.5`, about the three-round recovery span). Updates continue on the live weights, while uncertainty scoring, evaluation, and model snapshots read the averaged model, so the reported model is also the one that selects.
- `max_grad_norm` clips each update's gradient norm, for the rare loss spikes. It applies to the supervised baseline too, which keeps its recipe matched to active learning (ADR 0006).

The decay is fixed before any run uses it, like Gumbel's beta (ADR 0009): choosing it from test results would tune on the test split (ADR 0007).

## Considered Options

A learning rate decayed over all rounds was rejected. Nothing pointed to instability late in a run, and a run-long decay makes later rounds learn less by design, which flattens both arms' late curves for reasons unrelated to acquisition. Averaging after every update instead of every round was rejected because rounds take 1 to 24 updates across sampled configurations, so a per-update average would span very different numbers of rounds. Averaging only for evaluation was rejected because the reported model would then differ from the model that chose its words.

## Consequences

Both arms still start from one identical state (ADR 0003): the averaged copies start from the cloned bootstrap weights. Averaging keeps one more model per arm on the device. Results that use any of these settings need fresh random-baseline tuning, best-UQ, supervised, and sweep runs; they are not comparable with the 2026-10 rerun's results, which keep the constant rate.
