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
seeds within each search trial; search trials themselves remain sequential. It can be
changed when resuming a study without changing that study's identity.

Pool and evaluation tokenization and first-subword positions are cached once per seed.
Uncertainty is computed in batches on the model device, transferring only one score per
remaining word to the CPU. Increasing `score_batch_size` can improve inference throughput;
training batch size and `K` remain separate experimental choices. Device reductions can
produce small floating-point differences from the earlier CPU scoring implementation,
which can affect acquisition order for nearly tied scores.

## Hyperparameter search

All search logic lives in `scripts/bert_token_uq_search.py`. Choose Optuna's sampler
in the JSON configuration: `"sampler":"tpe"` (default), `"sampler":"grid"`, or
`"sampler":"random"`. Each uses the same discrete `SEARCH_SPACE`, validation split,
objective, persistent SQLite study, and trial outputs.

All launch settings can live alongside experiment settings in one JSON file. The
included DistilBERT configuration runs up to 100 additional TPE trials, each with
five parallel seeds and a 100% acquisition budget over the tuning pool:

```bash
uv run scripts/bert_token_uq_search.py --config configs/distilbert_tpe_5_seeds.json
```

The file includes `trials`, `study_name`, `studies_dir`, `validation_fraction`,
`validation_seed`, `sampler_seed`, and `timeout` (`null` means no time limit), plus
`sampler` and experiment settings such as `checkpoint`, `model_seeds`, and
`seed_workers`. Relative paths resolve from the working directory. `trials` is
required; other omitted fields retain their defaults. Settings in `SEARCH_SPACE`
are still chosen by Optuna per trial, overriding any fixed values for those fields.

`--config-json` accepts the same configuration inline. Choose either `--config`
or `--config-json`; explicit launch flags override values from either JSON source.
For example, `--config configs/distilbert_tpe_5_seeds.json --trials 20` runs up to
20 additional trials. Existing commands remain supported:

```bash
uv run scripts/bert_token_uq_search.py \
  --trials 20 \
  --study-name distilbert-tpe \
  --config-json \
  '{"sampler":"tpe","checkpoint":"distilbert-base-cased","model_seeds":[0,1],"max_pool_percent":50}'

uv run scripts/bert_token_uq_search.py \
  --trials 20 --study-name distilbert-grid \
  --config-json '{"sampler":"grid"}'

uv run scripts/bert_token_uq_search.py \
  --trials 20 --study-name distilbert-random \
  --config-json '{"sampler":"random"}'
```

TPE adapts suggestions using previous trial results. Random search samples independently
and can repeat configurations. Grid search enumerates combinations in a seeded shuffled
order, stopping at `trials`, the optional `timeout`, or grid exhaustion. The full
current grid contains 5,832 combinations; a smaller trial count explores only part of it.
`sampler_seed` controls the sampler seed. Trial count is always required, and a valid
command starts real training immediately.

During a search, the terminal shows one updating progress line with the current
trial, latest seed and round, and overall progress. Round tables and routine Optuna
messages are suppressed. The study summary and best configuration path print once
the invocation finishes; each completed trial still saves its full evaluation and
selection records as it finishes.

Search withholds a deterministic validation subset from the pool and removes those
sentences from acquisition. The held-out test split is never passed to a trial. The
objective is the mean across model seeds of the normalized acquisition-curve area for
`uncertainty entity F1 - random entity F1`. Each trial uses one checkpoint and the
configured model seeds. Search trials disable W&B logging.

Running the same command and study name resumes the study for up to `trials` additional
trials. Completed trial history is retained, but interrupted trials do not resume:
model and optimizer states are not saved. An exhausted grid exits without loading
data or training. Changing the sampler,
sampler seed, fixed experiment settings, search space, validation settings, or acquisition
schedule requires a new study name. Compatible older TPE studies are recognized as TPE.
The SQLite database retains trial history, but restarting a process reinitializes the
sampler RNG; a resumed TPE/random sequence need not match one uninterrupted run.

Run only one search process per study; stop the old process and its workers before restarting.

Each study writes `study.db`, `search_config.json`, `trials.json`, `summary.json`, `best_config.json`, and
per-trial settings, evaluations, and selected tokens below
`results/optuna/<study-name>/`. The summary and trial records include the search context.
`search_config.json` records the resolved settings from the latest accepted invocation,
including CLI overrides.
After selecting a configuration, pass the contents of `best_config.json` to the notebook's
`--config-json` option for evaluation on the original pool and untouched test split.
The exported best configuration contains experiment settings only.

This replaces the separate grid/Optuna scripts and the earlier `sweep`/`tune` commands.
The old balanced sweep, fixed-all-metrics, and sweep W&B flags are removed; use the
notebook for individual final evaluations and W&B logging.

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
| `scripts/bert_token_uq_search.py` | Optuna TPE/grid/random search, validation split, objective, and outputs |
| `tests/test_token_uq.py` | focused offline invariant tests |
| `skills-lock.json` | project skill sources and content hashes |
