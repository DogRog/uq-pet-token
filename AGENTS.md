# AGENTS.md

## Project purpose

`uq-pet-token` compares sequential acquisition of uncertain NER words (PET by default,
CoNLL-2003 optionally) with the same number of random words. The implementation is
intentionally small: three focused experiment modules, one shared experiment entry
layer, and one marimo notebook. Sentence-level LLM selection lives in the separate
`DogRog/uq-pet` repository; do not port code between the two.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests notebooks scripts
uv run ruff format --check src tests notebooks scripts
uv run marimo check notebooks/bert_token_uq.py
uv run notebooks/bert_token_uq.py
uv run marimo edit notebooks/bert_token_uq.py
```

Plain notebook script mode uses synthetic display records and must not download or
train a model unless a validated `--config-json` batch configuration is supplied.
Real runs start from the notebook's **Run experiment** button or that batch mode.

## Experiment invariants

- Keep the default 5 seed / 328 pool / 84 test split and `datasets==2.19.2` pin.
- `dataset` defaults to PET. CoNLL-2003 draws 5 seed sentences and the pool from its
  train split and evaluates on its full test split, at the pinned `CONLL_REVISION`.
  `dataset_percent` keeps a nested prefix of the pool; seed and test never change.
  Leave the full PET pool implicit in saved plans (`omit_default_dataset`) so earlier
  sweeps still resume.
- The test split is evaluation-only.
- Scoring accepts label-free pool inputs. Read a pool label only after selection.
- A candidate is `(pool_sentence_index, word_index)` and cannot be selected twice.
- Score, supervise, and predict from the first subword only.
- Mask every other training position with `-100`; do not replace input words by
  `[MASK]`.
- Exclude truncated pool words and reject truncated evaluation sentences.
- Clone one bootstrap-trained model and optimizer state into both arms. Thereafter
  each arm keeps its own state across rounds.
- Each round selects `K` new tokens per arm, or the remaining token budget in the
  final round. Both arms use the same replay ratio, update passes, batch size, and
  learning rate. Scale replay to the actual number of new tokens in that round.
- Replay is sampled at token level from labels available before the current round.
- Selection and replay use local seeded RNGs. Global seeding belongs only in
  `token_model.set_seed`.
- UQ metrics consume first-subword class probabilities and return larger values for
  greater uncertainty. Keep the notebook selector and saved `uq_metric` synchronized.
- Live progress snapshots are emitted after round 0 and after both arms complete each
  round. Do not publish a half-finished round to the comparison chart.
- Outputs must record settings, every evaluation row, and every selected token.

## YAGNI boundary

Do not add YAML configuration, a CLI framework, metric registries, callbacks, model
checkpoint management, or generalized experiment abstractions without a concrete
need. Keep validated experiment and search configuration in `config.py`, UI controls
in the marimo notebook, and shared W&B logging in `utils/wandb_logging.py`. Keep data,
model operations, and active-learning orchestration in their focused modules, and
search planning, resume handling, and search summaries together in `search.py`.
Keep the fully supervised baseline's sampled search, tuning, test runs, and resume
logic in `supervised.py`. It tunes only on the validation holdout, and its test runs
read the test split once per trained model.

## Agent skills

The marimo skills from [marimo-team/skills](https://github.com/marimo-team/skills) are
installed for Codex with the [Vercel skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add marimo-team/skills --agent codex
```

Installed files live under `.agents/skills/`, and `skills-lock.json` records their
sources and hashes. Both are gitignored, so each checkout installs its own. Useful
maintenance commands:

```bash
npx skills list --agent codex
npx skills update --project --yes
npx skills add owner/repository --agent codex
```

Use `--list` on an `add` command to inspect a repository before installing it, or
`--skill <name>` to select particular skills.
