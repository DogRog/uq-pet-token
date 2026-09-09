# uq-pet: token-level active learning

Does a PET token classifier learn faster when each online update uses its most
uncertain unlabelled words instead of the same number of random words?

This branch, `bert-token-uq`, implements the experiment as a small marimo workflow.
The earlier sentence-level BERT experiment remains on `bert-uq`.

## Protocol

The stable PET split remains 5 labelled seed sentences, 328 pool sentences, and 84
held-out test sentences. For every model seed:

1. Fine-tune a fresh 15-tag token classifier on the five seed sentences.
2. Clone the exact fitted weights and optimizer state into uncertainty and random arms.
3. Evaluate both arms before acquisition (round 0).
4. At each round, acquire `K` previously unseen pool words, or the remaining budget
   in the final round:
   - uncertainty selects the largest score from the configured UQ metric;
   - random draws uniformly from its remaining word pool.
5. Reveal only the selected labels.
6. Continue each arm from its current weights and optimizer state using the new
   token-centred items plus limited replay.
7. Evaluate on the held-out test set and repeat.

The full sentence remains model input, but each online training item labels exactly
one word. Only its first subword contributes to loss; every other position is `-100`.
By default, each round replays up to one older labelled token per new word, including
seed tokens. Replay scales down with a smaller final round. A 100% pool budget acquires
every scoreable word; lower percentage budgets round down only to a whole number of words.

The notebook exposes three token-level UQ metrics, all using the first subword's class
probabilities and all ranked with larger values meaning more uncertain:

- `entropy`: normalized predictive entropy.
- `least_confidence`: one minus the largest class probability.
- `margin`: one minus the gap between the two largest class probabilities.

## Run it

```bash
uv sync
uv run marimo edit notebooks/bert_token_uq.py
```

Choose the settings and press **Run experiment**. A completed run writes:

```text
results/bert_token_uq_<timestamp>/
├── config.json
├── results.csv
└── selections.json
```

The entity F1 and token-accuracy charts appear after bootstrap evaluation and update
live after both arms finish every round. The notebook also shows selected label counts,
the run description, paired gap against random, and the token-level selection log.

For a read-only presentation view, use:

```bash
uv run marimo run notebooks/bert_token_uq.py
```

## Batch runs

Interactive and batch runs both execute the marimo notebook with the same validated
`ExperimentConfig`. Passing one JSON object starts a real non-interactive run; running the
notebook without arguments remains a synthetic no-download validation:

```bash
uv run notebooks/bert_token_uq.py --config-json \
  '{"checkpoint":"microsoft/deberta-v3-base","model_seeds":[0],"update_passes":2,"learning_rate":0.00003}'
```

To share a large GPU across independent seeds, set **Concurrent seeds** in the notebook
or pass `seed_workers` in JSON (default `1`):

```bash
uv run notebooks/bert_token_uq.py --config-json \
  '{"model_seeds":[0,1,2,3,4],"seed_workers":2,"score_batch_size":128}'
```

Workers use separate spawned processes on the selected device, each owning both arms
and their optimizer and RNG states. Start with two workers, then increase if measured
throughput improves and peak GPU memory permits. The worker count is capped at the
number of seeds. Each seed's rounds remain sequential; live updates append only complete
two-arm rounds as they arrive, and saved records retain the configured seed order.
Errors or interruption stop the remaining workers. `seed_workers` also applies to the
seeds within each sweep configuration. Within a seed, learner groups share
one bootstrap and random trajectory. Seeds can be working on different metric groups at the
same time. Worker count can change when resuming without changing the saved plan.

Pool and evaluation tokenization and first-subword positions are cached once per seed.
Uncertainty is computed in batches on the model device, transferring only one score per
remaining word to the CPU. Increasing `score_batch_size` can improve inference throughput;
training batch size and `K` remain separate experimental choices. Device reductions can
produce small floating-point differences from the earlier CPU scoring implementation,
which can affect acquisition order for nearly tied scores.

`score_batch_size` defaults to **256 sentences** for both pool scoring and test evaluation;
it does not change the number of words acquired or the training batch size.

`precision` defaults to `"auto"`: native BF16 autocasting on supported CUDA GPUs (including
the RTX 4090), and FP32 on other devices. Bootstrap, online training, pool scoring, and
test evaluation all use the same resolved precision. Parameters and AdamW state stay
FP32, and scoring converts first-subword logits to FP32 before softmax and uncertainty
reduction. BF16 uses no gradient scaler. Set `"precision":"fp32"` for a full-precision
comparison, or `"precision":"bf16"` to require native CUDA BF16 support and fail early
otherwise. The notebook exposes the same selector. Exports record the requested
`precision` and `effective_precision`; a sweep's plan also locks the latter across resumes.
BF16 can change predictions and acquisition order, so compare speed and learning curves
under a new sweep name rather than mixing precision within an existing sweep.

`model_batch_size` defaults to **2 learners per seed**. PyTorch `vmap` applies the model
to stacked parameter tensors, using batched matrix operations for different learners.
With all three UQ metrics, entropy and random train/evaluate together, followed by
least confidence and margin together. Pool scoring batches the UQ learners in each group;
the random learner does not score the pool. Set `model_batch_size` to `1` for the previous
sequential path, or `3`/`4` to test larger groups. This is separate from sentence batch
size, training batch size, and annotation `k`: each learner still gets the same number
of newly labelled words, replay items, and optimizer updates.

Every learner has its own FP32 parameter slice and AdamW moment slices, cloned from
bootstrap. Per-model mean losses are summed before backward so gradients are not divided
by the number of learners. The common AdamW step counter is valid because every learner
in a group has the same number of updates. Each learner retains its original seeded
shuffle, selection, and replay; vectorized dropout uses distinct draws per learner from
an isolated seed/round RNG stream. Grouping and shared padding change dropout draws, so
batched runs are reproducible but are not bit-for-bit continuations of sequential runs.
Changing group size requires a new sweep name.

The batched path uses eager attention for `vmap` compatibility across the five supported
architectures. Pool/test attention masks are prepared once per group outside `vmap`.
In BF16 mode, differentiable BF16 execution copies prevent FP32 biases from promoting
vectorized linear outputs back to FP32; master parameters and optimizer moments stay FP32.
Larger groups increase activation memory and may not be faster than sequential execution
with fused attention. Compare complete-run time with the same checkpoint, hyperparameters,
seeds, precision, and worker count; `model_batch_size=1` versus `2` is the first comparison.

## Random hyperparameter sweep

`scripts/bert_token_uq_search.py` samples a fixed set of distinct configurations
uniformly without replacement, using a local seeded RNG. It saves the entire plan
before loading data or training. Scores never change the plan, run order, or budget.
Every configuration runs entropy, least confidence, and margin against a matched
random-acquisition arm on the original 5 seed / 328 pool / 84 test sentence split.
Test metrics are evaluated after bootstrap (round 0) and each acquisition round.
There is no validation holdout, optimization objective, pruning, or best-gap selection.

The five files in `configs/*_random_5_seeds.json` retain the model checkpoints and
five seeds. Each budgets **30 hyperparameter configurations × 3 UQ metrics = 90
paired comparisons**, with five concurrent seeds and 100% pool acquisition. This is a larger
workload than the old 30-trial search, which sampled only one UQ metric per trial.
For each configuration and seed, bootstrap runs once. Its fitted weights and optimizer
state are held in CPU memory and cloned into each arm. The first UQ metric trains beside
random; later metrics train their own UQ arms and reuse the random evaluation rows
and selected-token records. With the default group size, at most two learner states
reside on the GPU per seed (plus transient BF16 execution copies and activations).
For three metrics this removes two bootstrap runs and two random trajectories per seed;
all three UQ trajectories retain separate model and optimizer histories.
Each metric retains a complete paired export and live chart. The shared random baselines
are not independent observations to pool across metrics.

```bash
uv run scripts/bert_token_uq_search.py --config configs/distilbert_random_5_seeds.json
```

The unchanged hyperparameter ranges are:

| Setting | Values |
| --- | --- |
| `k` | 8, 16, 32 |
| `bootstrap_epochs` | 10 |
| `update_passes` | 1, 2, 4 |
| `learning_rate` | 0.00002, 0.00005 |
| `batch_size` | 8, 16 |
| `replay_ratio` | 0, 1, 2 |
| `weight_decay` | 0, 0.01 |

There are 216 distinct combinations. All supported UQ metrics are applied to each;
`uq_metric` is not sampled. Values for sampled fields in the input configuration
are overwritten by the saved plan. Other experiment settings remain fixed.
Identical sampler seeds and budgets give identical sampled configurations across checkpoints.
W&B is disabled for sweep runs.

Use `--config` or `--config-json`, with optional overrides `--num-configs`,
`--sweep-name`, `--sweeps-dir`, and `--sampler-seed`. `num_configs` is required and
is the **total fixed budget**, not an additional-run count. A valid command starts
real training. For example:

```bash
uv run scripts/bert_token_uq_search.py \
  --num-configs 10 --sweep-name distilbert-random-10 \
  --config-json '{"checkpoint":"distilbert-base-cased","model_seeds":[0,1,2,3,4]}'
```

Running the same command resumes unfinished comparisons, retaining completed outputs.
A configuration trains its unfinished metrics together and saves paired exports after
all its seed workers finish. If interrupted during training, its unfinished comparisons
restart together from bootstrap; model and optimizer checkpoints are not saved. Exports
already marked complete are retained even if saving a later metric fails. Only
`seed_workers` may change without changing the scientific plan.
Changing the budget, sampler seed, checkpoint, model seeds, search ranges, or other
scientific settings requires a new sweep name. Run only one process per sweep.
The model-batching implementation uses plan version 3 and requires a new sweep name
for historical version 1/2 sweeps; their existing outputs remain readable and untouched.
Choose the search space and budget before inspecting test curves; changing them in
response to favorable test gaps would make the resulting assessment exploratory.

Outputs under `results/random_search/<sweep-name>/`:

- `plan.json`: immutable sampled configurations, all UQ metrics, seeds, and fixed settings.
- `search_config.json`: settings of the latest accepted invocation.
- `runs/<config-id>/<metric>/progress.csv`: atomically updated test rows after each
  completed round pair, including round 0.
- Each completed comparison saves `config.json`, `results.csv`, and `selections.json`
  in a timestamped run directory, referenced by `completed.json`.
- `summary.json`: every planned comparison, its pending/failed/complete status, and
  per-metric mean test F1 gap AUC, mean final F1 gap, wins, ties, losses, and win fraction.
  Failed comparisons retain their error and are retried on resume.

The AUC integrates UQ minus random entity F1 against acquired-pool percentage,
normalizes by the observed acquisition interval, and averages across seeds. Each
configuration has equal weight in per-metric summaries; wins use its seed-mean AUC
gap, with absolute gaps at most 1e-12 treated as ties. Incomplete sweeps are explicitly
marked, with completed and planned counts. No configuration is ranked or selected.

The previous Optuna dependency and tuning workflow have been removed. Existing
`results/optuna` artifacts remain untouched and contain validation evaluations;
they are excluded from test analysis.

## Checks

```bash
uv run pytest
uv run ruff check src tests notebooks scripts
uv run ruff format --check src tests notebooks scripts
uv run marimo check notebooks/bert_token_uq.py
uv run notebooks/bert_token_uq.py
uv run scripts/bert_token_uq_search.py --help
```

Running the notebook as a plain script uses small synthetic display data. It validates
the reactive notebook without downloading weights or starting an experiment.

## Agent skills

This project uses the [Vercel skills CLI](https://github.com/vercel-labs/skills) to
install skills for Codex. The marimo skills come from
[marimo-team/skills](https://github.com/marimo-team/skills):

```bash
npx skills add marimo-team/skills --agent codex
```

The installed files live under the ignored `.agents/skills/` directory, while
`skills-lock.json` records their sources and hashes. Useful maintenance commands are:

```bash
npx skills list --agent codex
npx skills update --project --yes
npx skills add owner/repository --agent codex
```

Use `--list` on an `add` command to inspect a repository before installing it, or
`--skill <name>` to select particular skills.

## Scientific invariants

- The test set is evaluation-only.
- Pool inputs passed to uncertainty scoring contain no labels.
- A label is read from the private pool lookup only after its word is selected.
- Acquisition is without replacement and counts newly labelled words.
- First subwords are used consistently for scoring, online loss, and evaluation.
- Continuation subwords, padding, and special tokens are ignored.
- Truncated pool words are not selectable; truncated test words are an error.
- A word is truncated if any of its subwords are missing, including at the left boundary.
- Both arms begin from the same fitted state and receive the same update budget.
- The uncertainty and random arms keep separate weights and optimizer histories.
- Selection and replay use deterministic local random generators.
- Training uses the seed for each arm and round for dropout as well as shuffling,
  restoring the surrounding RNG states even if an update fails.

## Layout

| Path | Purpose |
| --- | --- |
| `src/uq_pet/pet_data.py` | PET identity, download, stable split, and private label lookup |
| `src/uq_pet/token_model.py` | masking, training, UQ metrics, inference, and evaluation |
| `src/uq_pet/utils/truncation.py` | word alignment and truncation checks |
| `src/uq_pet/active_learning.py` | acquisition rounds, replay, orchestration, and outputs |
| `src/uq_pet/experiment.py` | shared configuration, run execution, summaries, and W&B records |
| `notebooks/bert_token_uq.py` | controls, experiment run, tables, and plots |
| `notebooks/fixed_all_metrics_analysis.py` | read-only analysis of saved historical sweeps |
| `notebooks/charts.py` | shared chart builders and W&B comparison media |
| `scripts/bert_token_uq_search.py` | fixed random sweep, paired test evaluation, resume, and summaries |
| `notebooks/test_analysis.py` | individual test curves and sweep-wide paired gaps |
| `tests/test_token_uq.py` | focused offline invariant tests |
| `skills-lock.json` | project skill sources and content hashes |

### Benchmark seed concurrency

Training reuses label-free sentence tokenization in RAM within each seed. Targets
are rebuilt per batch; no model states are cached or written to disk. Prepared pool
and evaluation inputs stay on the selected device across rounds.

Compare 1, 2, and 5 workers on the server (stop other experiment runs first):

```bash
uv run scripts/benchmark_seed_workers.py --config configs/distilbert_random_5_seeds.json > worker_timings.json
```

This runs all configured seeds for each worker count, twice, using fixed experiment
settings from the JSON (or their defaults), rather than sampling configurations.
It uses the original test split, caps acquisition at 5%, and prints timings plus
the fastest worker count when finished. It does not write a sweep or change the config.
Optional JSON fields are `worker_counts`, `repeats`, and `benchmark_pool_percent`.
Set `seed_workers` in the search config after comparing the results. Timings include
model loading and bootstrap; confirm the winner on a longer workload if close.

### Plot test performance

```bash
uv run marimo edit notebooks/test_analysis.py
```

This read-only notebook plots held-out test F1 and token accuracy after bootstrap
(round 0) and every acquisition round, with seed means and SD bands. Select a sweep
or regular experiments, then a checkpoint, UQ metric, and saved run. Runs with
different settings remain separate. It also shows cumulative acquired NER-tag coverage.

For random sweeps, it shows all configuration-level AUC gaps around zero, a per-metric
summary table, and pending/failed comparisons. Sweep summaries always cover all completed
configurations in that sweep, independently of the individual-run selectors below.

It reads regular exports directly under `results/bert_token_uq_*` and completed sweeps
under `results/random_search/*` (the default output directory). It never downloads or
trains a model. Legacy top-level regular exports without an evaluation split marker
are supported; sweep exports must explicitly identify the test split.
