# uq-pet: token-level active learning

Does a PET token classifier learn faster when each online update uses its most
uncertain unlabelled words instead of the same number of random words?

The project provides an interactive marimo experiment, random-baseline hyperparameter
tuning, fixed comparisons of all three uncertainty metrics, and resumable random
searches. The supplied configurations cover DistilBERT, BERT, RoBERTa, DeBERTa-v3,
and ModernBERT.

## Quick start

Use Python 3.12 or newer and `uv`. From the project root:

```bash
uv sync
uv run notebooks/bert_token_uq.py
uv run marimo edit notebooks/bert_token_uq.py
```

The plain script command validates the notebook with synthetic display data and does
not download data or train. In the editor, choose settings and press **Run experiment**
to start a real run. Real runs download PET and model weights when needed; PET is
cached at `data/raw/PETv1.1-entities.jsonl`.

| Task | Entry point |
| --- | --- |
| Run one UQ metric against random | `notebooks/bert_token_uq.py` |
| Compare all UQ metrics with saved validation winners | `scripts/run_best_uq_all_models.sh` |
| Tune random acquisition on a validation holdout | `scripts/tune_random_all_models.sh` |
| Sample hyperparameters and compare UQ against random | `scripts/bert_token_uq_search.py` |
| Inspect tuning completeness and winners | `notebooks/random_baseline_analysis.py` |
| Inspect random-search test curves and selections | `notebooks/random_search_analysis.py` |
| Compare best-config UQ results and learning curves | `notebooks/best_uq_analysis.py` |

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

## Interactive experiment

The notebook uses the validated defaults in `src/uq_pet/config.py`:

| Setting | Default |
| --- | --- |
| Checkpoint / model seeds | `distilbert-base-cased` / `0,1,2,3,4` |
| UQ metric / new words per round | `entropy` / `32` |
| Pool acquisition budget | `100%` of scoreable words |
| Bootstrap epochs / online update passes | `20` / `1` |
| Replay ratio | `1` older labelled token per new word |
| Learning rate / weight decay | `5e-5` / `0.01` |
| Training / scoring batch size | `8` items / `256` sentences |
| Maximum tokenizer length | `256` |
| Concurrent seeds / precision | `1` / `auto` |
| W&B logging | Disabled |

Saved sweep and winner configurations override these defaults. A completed run writes:

```text
results/bert_token_uq_<timestamp>/
├── config.json
├── results.csv
└── selections.json
```

The entity F1, macro entity F1, and token-accuracy charts appear after bootstrap
evaluation and update live after both arms finish every round. The notebook also shows selected label counts,
the run description, paired gap against random, cumulative NER-tag coverage, and the
token-level selection log.

For the app view without the code editor, use:

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

To run all three UQ metrics with fixed settings, use:

```bash
uv run scripts/run_uq_metrics.py --config configs/best_uq/bert_base.json
```

This compares entropy, least confidence, and margin using exactly one supplied
configuration, sharing bootstrap training and the matched random baseline across
metrics. The saved best-UQ configs run all five seeds concurrently; use `--seed-workers N` to
override this. Configs without `seed_workers` default to two concurrent seeds.
No hyperparameters are sampled.

Results are grouped under `results/best_uq/<checkpoint>-best-uq/`, with `plan.json`,
`search_config.json`, `summary.json`, and per-metric outputs under
`runs/config_0000/<metric>/`. Each metric retains settings, evaluation rows, and
selected tokens. Re-running skips completed comparisons; an interrupted comparison
restarts. Changed scientific settings require a new `--sweep-name`.

Use `--uq-metrics entropy margin` to select a subset and `--dry-run` to print the
validated plan without downloading data or training. `--config-json` accepts inline
JSON instead of a file; `--sweeps-dir` changes the output parent directory.

Configuration files are grouped by purpose:

- `configs/best_uq/`: saved winners from the five completed random-baseline searches, with five concurrent seeds.
- `configs/tune_random/`: random-baseline hyperparameter tuning settings.
- `configs/random_search/`: sampled UQ-versus-random comparison settings.

Run every saved winner with all three UQ metrics (five models × three metrics × five
seeds, with matched random baselines):

```bash
bash scripts/run_best_uq_all_models.sh
```

Pass `--dry-run` to preview the five fixed plans (three metrics each) without training,
or `--uq-metrics entropy margin` to run a subset. Models run sequentially, with parallel
seeds within each model; the script stops on the first failure. Winner provenance is in `configs/best_uq/README.md`.

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
seeds within each sweep configuration. Within a seed, metrics run sequentially and share
one bootstrap and random trajectory. Different seeds may be working on different metrics
at the same time. Worker count can change when resuming without changing the saved plan.

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

Learners execute sequentially within each seed, each retaining its own model and
optimizer state cloned from the shared bootstrap.

## Tune the random baseline

Use `tune-random` to search hyperparameters with **random selection only**, save the
validation winner, and stop. It never launches UQ acquisition or evaluates the test
split. Run your UQ experiments separately using the saved settings when ready.

```bash
uv run scripts/bert_token_uq_search.py --config configs/tune_random/distilbert_tune_random_100.json --dry-run
uv run scripts/bert_token_uq_search.py --config configs/tune_random/distilbert_tune_random_100.json
```

To tune all five models (DistilBERT, BERT, RoBERTa, DeBERTa-v3, and ModernBERT)
sequentially, use:

```bash
bash scripts/tune_random_all_models.sh --dry-run
bash scripts/tune_random_all_models.sh
```

Each `configs/tune_random/*_tune_random_100.json` uses 100 configurations, five model seeds,
two seed workers, the same validation split and objective, and a separate output
folder. The launcher stops on the first failure; rerun it to resume saved searches.
It runs random-only tuning and saves each model's winning settings without launching UQ.

The supplied configuration samples 100 configurations for DistilBERT with five model
seeds and two seed workers. Dry run prints the plan without loading data or models.
`--mode compare` remains a separate workflow that compares UQ methods against random;
`--mode tune-random` only tunes the random baseline. The JSON config selects the mode.

1. Keep the original five seed sentences and 84 test sentences. Hold out 66 of the
   328 pool sentences for validation with a fixed local RNG, leaving 262 acquisition
   sentences. Validation sentences cannot be acquired or replayed.
2. Train only random selection for each sampled configuration. Maximize the seed-mean
   normalized validation entity-F1 AUC against acquired-pool percentage, including
   round 0. To optimize the endpoint instead, set `objective` to
   `random_validation_final_entity_f1` before starting. Ties use ascending config ID.
3. Save the winning hyperparameters and validation score, then exit. No model is
   trained after selecting the winner and no final test comparison is scheduled.

The search uses `k` 32/64, bootstrap epochs 10, update passes 1/2/4, batch size 32/64,
replay ratio 0/1/2, weight decay 0/0.01, and learning rates sampled log-uniformly from
1e-6 to 1e-4. Other experiment settings remain fixed. The inherited `uq_metric` field
is unused for acquisition in this mode and is omitted from new trial metadata and
winning settings. Random-only tuning trains a single random learner. W&B logs validation random metrics only when enabled.

The five tuning configs enable W&B and use separate projects:
`distilbert-random-baseline`, `bert-base-random-baseline`, `roberta-base-random-baseline`,
`deberta-v3-base-random-baseline`, and `modernbert-base-random-baseline`.
Online runs create a real W&B sweep and a saved workspace chart automatically.
The local seeded plan still schedules the search; no W&B agents or UQ runs are launched.

The sweep contains **one summary run per completed configuration**, averaged over all
model seeds. Its objective is `random_validation_entity_f1_auc` (or the configured
endpoint objective). The saved parallel-coordinates chart explicitly includes learning
rate (log scale), batch size, `k`, update passes, replay ratio, weight decay, final
validation F1, and validation F1 AUC, with the objective as its last/color axis.
Constant checkpoint/bootstrap settings are excluded from its axes. Per-seed live logs
remain separate and are excluded from this saved chart.

Sweep IDs, chart links, and published configuration IDs are saved in `wandb_sweeps.json`.
Resuming reuses the sweep and publishes any completed configurations not yet uploaded.
A completed search can be published without loading data or training:

```bash
uv run scripts/bert_token_uq_search.py --config configs/tune_random/distilbert_tune_random_100.json --publish-wandb-only
```

This requires an existing local plan and online W&B credentials. Offline mode keeps
per-seed local logs; create the online sweep later with the publication command.

Outputs under `results/random_baseline_search/<sweep-name>/`:

- `plan.json` and `split.json`: fixed search settings and validation split provenance.
- `tuning/<config-id>/`: validation settings, results, selections, progress, and completion score.
- `best_config.json`: winning experiment settings, ready for a separate experiment.
- `selection.json`: winner ID, objective, and validation score.
- `summary.json`: every trial's status and score; complete after both winner files exist.

Check trial completeness and view the best configuration for each model:

```bash
uv run marimo edit notebooks/random_baseline_analysis.py
```

The notebook reads local saved validation scores, reports trial completeness, and
shows the highest-scoring configuration in each completed sweep. Incomplete sweeps
are excluded from the winner table. To compare a newly tuned winner across UQ metrics:

```bash
uv run scripts/run_uq_metrics.py \
  --config results/random_baseline_search/distilbert-random-baseline-100/best_config.json
```

The checked-in winners and their provenance are documented in
[configs/best_uq/README.md](configs/best_uq/README.md).

Repeat the tuning command to resume. Completed trials are reused and interrupted trials restart
from bootstrap. Older two-stage sweeps are accepted when their tuning settings match:
`plan.v2.json` preserves the original plan, and historical `final/` artifacts are left
untouched. Resuming never starts or resumes those historical UQ comparisons. Exact
saved learning rates are retained despite insignificant sampling roundoff on comparison.
Changing scientific settings requires a new sweep name; workers and W&B preferences
may change on resume.

The winner is the best sampled configuration on this validation split, not a guaranteed
global optimum. Keep its settings frozen when assessing UQ methods separately; do not
choose new hyperparameters from test results.

## Random hyperparameter sweep

In its default `--mode compare`, `scripts/bert_token_uq_search.py` samples a fixed set of configurations
using independent uniform categorical draws and log-uniform learning-rate draws
with a local seeded RNG. It saves the entire plan
before loading data or training. Scores never change the plan, run order, or budget.
Every configuration runs entropy, least confidence, and margin against a matched
random-acquisition arm on the original 5 seed / 328 pool / 84 test sentence split.
Test metrics are evaluated after bootstrap (round 0) and each acquisition round.
There is no validation holdout, optimization objective, pruning, or best-gap selection.

The five files in `configs/random_search/*_random_5_seeds.json` retain the model checkpoints and
five seeds. Each budgets **30 hyperparameter configurations × 3 UQ metrics = 90
paired comparisons**, with five concurrent seeds and 100% pool acquisition. This is a larger
workload than the old 30-trial search, which sampled only one UQ metric per trial.
For each configuration and seed, bootstrap runs once. Its fitted weights and optimizer
state are held in CPU memory and cloned into each arm. The first UQ metric trains beside
random; later metrics train their own UQ arms and reuse the random evaluation rows
and selected-token records. At most two learner states
reside on the GPU per seed (plus transient BF16 execution copies and activations).
For three metrics this removes two bootstrap runs and two random trajectories per seed;
all three UQ trajectories retain separate model and optimizer histories.
Each metric retains a complete paired export and live chart. The shared random baselines
are not independent observations to pool across metrics.

```bash
uv run scripts/bert_token_uq_search.py --config configs/random_search/distilbert_random_5_seeds.json
```

The hyperparameter ranges are:

| Setting | Values |
| --- | --- |
| `k` | 32, 64 |
| `bootstrap_epochs` | 10 |
| `update_passes` | 1, 2, 4 |
| `learning_rate` | Log-uniform from 0.000001 to 0.0001 |
| `batch_size` | 32, 64 |
| `replay_ratio` | 0, 1, 2 |
| `weight_decay` | 0, 0.01 |

The 72 categorical combinations can repeat with different learning rates. There is
no finite grid limit on the budget. Each configuration uses the same sampled learning
rate across all model seeds and acquisition arms. All supported UQ metrics are applied to each;
`uq_metric` is not sampled. Values for sampled fields in the input configuration
are overwritten by the saved plan. Other experiment settings remain fixed.
Identical sampler seeds and budgets give identical sampled configurations across checkpoints.
Set `"wandb_enabled": true` and `"wandb_project": "your-project"` to log sweep runs.
Online logging requires `WANDB_API_KEY` in the process environment or the project's
`.env` file and validates it before loading data or training. Exported environment
variables take precedence over `.env` values. `WANDB_MODE=offline` writes local W&B records without
credentials; offline mode has no online project link.

The CLI prints the project and run URLs when round 0 arrives. Each configuration,
UQ metric, and seed has its own W&B run, grouped under the sweep name, with both UQ
and random curves against acquired-pool percentage. Logging stays in the parent process
and publishes only completed comparison pairs. Runs close on completion or failure.
Console capture is disabled in sweep and notebook runs, so terminal progress redraws
are not saved or uploaded as `output.log`. Metrics are still logged normally.
The shared random curves across metrics remain repeated observations of one baseline.

W&B settings can change when resuming an existing sweep without changing its scientific
plan. Completed comparisons are skipped and are not uploaded retroactively. An unfinished
configuration restarts from bootstrap and gets fresh W&B runs so its new trajectory is
not appended to an interrupted one. Updating the code or config does not alter an already
running process.

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
`seed_workers` and W&B logging settings may change without changing the scientific plan.
Changing the budget, sampler seed, checkpoint, model seeds, search ranges, or other
scientific settings requires a new sweep name. Run only one process per sweep.
Log-uniform sampling uses comparison plan version 4 and tuning plan version 3.
Fixed comparisons use plan version 1.
Sweeps created with the previous sampler require a new sweep name; their existing outputs remain readable and untouched.
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
uv run scripts/run_uq_metrics.py --help
bash scripts/run_best_uq_all_models.sh --dry-run
bash scripts/tune_random_all_models.sh --dry-run
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
| `src/uq_pet/config.py` | shared validated experiment and search configuration |
| `src/uq_pet/experiment.py` | run execution and label coverage summaries |
| `src/uq_pet/utils/wandb_logging.py` | W&B credentials, evaluation records, and comparison run lifecycles |
| `notebooks/bert_token_uq.py` | controls, experiment run, tables, and plots |
| `notebooks/fixed_all_metrics_analysis.py` | read-only analysis of saved historical sweeps |
| `notebooks/utils/` | shared notebook helpers, including chart builders and W&B comparison media |
| `src/uq_pet/search.py` | random sweep or baseline tuning, shared execution, resume, and summaries |
| `scripts/bert_token_uq_search.py` | CLI entry point for search and tuning |
| `notebooks/random_search_analysis.py` | random-sweep summaries and per-configuration drill-downs |
| `notebooks/random_baseline_analysis.py` | trial completeness and best saved validation configurations |
| `scripts/run_uq_metrics.py` | fixed multi-metric comparisons from one experiment configuration |
| `scripts/run_best_uq_all_models.sh` | fixed comparisons for all five saved winners |
| `scripts/tune_random_all_models.sh` | random-only tuning for all five checkpoints |
| `src/uq_pet/utils/wandb_tuning.py` | tuning sweep publication and workspace charts |
| `configs/` | saved winners, tuning configs, and random-search configs |
| `tests/` | offline protocol, model-boundary, CLI, concurrency, resume, and logging checks |
| `skills-lock.json` | project skill sources and content hashes |

## Analyze saved results

```bash
uv run marimo edit notebooks/random_baseline_analysis.py
uv run marimo edit notebooks/random_search_analysis.py
uv run marimo edit notebooks/best_uq_analysis.py
```

The baseline notebook reads `results/random_baseline_search/` and displays trial
completeness and validation winners. The random-search notebook reads test sweep
summaries under `results/random_search/`, with aggregate metric results, pending or
failed comparisons, and a per-configuration drill-down. It plots seed means and
standard-deviation bands for entity F1, macro entity F1 (when available), and token
accuracy, plus acquired NER-tag coverage and selected words with sentence context.
Sentence context requires the local `data/raw/PETv1.1-entities.jsonl` file.

These notebooks read saved files and do not train models. The random-search viewer
does not currently discover `results/best_uq/`; fixed comparisons save the same
per-metric exports and summary files there for inspection.

`notebooks/fixed_all_metrics_analysis.py` is a historical analysis notebook tied to
`results/sweeps/bert_token_uq_20260825_090329_945236_fixed-all-metrics/`. It requires
that sweep's `combined_results.csv` and selection files; it is not a general viewer
for new fixed comparisons.
