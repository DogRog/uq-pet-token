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

The local sweep launcher samples acquisition and training configurations and crosses each one with the
requested checkpoints and model seeds. It is a dry run unless `--launch` is present:

```bash
# Four configurations × five checkpoints × five seeds = 100 printed commands.
uv run scripts/bert_token_uq_grid.py

# A small real smoke sweep: one configuration × two checkpoints × one seed.
uv run scripts/bert_token_uq_grid.py \
  --count 1 \
  --checkpoints distilbert-base-cased,bert-base-cased \
  --seeds 0 \
  --launch
```

To compare every UQ metric on every default checkpoint and model seed with one shared
hyperparameter configuration, use the fixed mode. It uses the validated
`ExperimentConfig` defaults and plans 3 metrics × 5 checkpoints × 5 seeds = 75 runs:

```bash
# Inspect all 75 commands without training.
uv run scripts/bert_token_uq_grid.py --fixed-all-metrics

# Execute the same plan sequentially.
uv run scripts/bert_token_uq_grid.py --fixed-all-metrics --launch
```

In fixed mode, `--count` and `--seed` are ignored. `--checkpoints`, `--seeds`,
`--max-pool-percent`, and the W&B options still apply uniformly to every run.

## Optuna tuning

The Optuna runner tunes the same discrete acquisition and training search space as the
grid runner. It makes a deterministic validation subset from the pool and removes those
sentences from acquisition during tuning. The held-out test split is never passed to a
trial, so it remains available for one final evaluation of the selected configuration.

The number of trials is required because a default full study would be expensive:

```bash
uv run scripts/bert_token_uq_optuna.py \
  --trials 20 \
  --study-name distilbert-validation \
  --config-json \
  '{"checkpoint":"distilbert-base-cased","model_seeds":[0,1],"max_pool_percent":50}'
```

The objective is the mean, across model seeds, of the normalized acquisition-curve area
for `uncertainty entity F1 - random entity F1`. The study uses persistent SQLite storage.
Running the same command and study name resumes it and adds the requested number of
trials. Changing fixed configuration, validation settings, or the acquisition schedule
requires a new study name. Studies using the older full-round-only schedule cannot be
resumed with the smaller-final-round protocol.

Each study writes `study.db`, `trials.json`, `summary.json`, `best_config.json`, and the
complete per-trial experiment records below `results/optuna/<study-name>/`. After tuning,
pass the contents of `best_config.json` to the notebook's `--config-json` option to run
the selected configuration once on the original pool and untouched test split.

Launched sweeps write a manifest, per-run records, and `combined_results.csv` below
`results/sweeps/`. Add `--wandb` to enable Weights & Biases for every job; it is off by
default. Set `WANDB_API_KEY` in the environment or `.env`, and optionally choose a project:

```bash
uv run scripts/bert_token_uq_grid.py \
  --count 1 \
  --wandb \
  --wandb-project uq-pet-token-uq \
  --launch
```

For online W&B launches, the launcher checks `WANDB_API_KEY` before creating the sweep
or starting its first job. `WANDB_MODE=offline` intentionally bypasses that key check.

The first real run downloads `distilbert-base-cased` if it is not already cached. It
does not require an API key.

## Checks

```bash
uv run pytest
uv run ruff check src tests notebooks scripts
uv run ruff format --check src tests notebooks scripts
uv run marimo check notebooks/bert_token_uq.py
uv run notebooks/bert_token_uq.py
uv run scripts/bert_token_uq_optuna.py --help
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
| `src/uq_pet/experiment.py` | shared configuration, run execution, charts, and W&B records |
| `notebooks/bert_token_uq.py` | controls, experiment run, tables, and plots |
| `scripts/bert_token_uq_grid.py` | dry-run-first local sweep generation and execution |
| `scripts/bert_token_uq_optuna.py` | persistent validation-only Optuna tuning |
| `src/uq_pet/tuning.py` | shared search space, nested validation split, and objective |
| `tests/test_token_uq.py` | focused offline invariant tests |
| `tests/test_batch_grid.py` | deterministic sweep and aggregation tests |
| `skills-lock.json` | project skill sources and content hashes |
