# AGENTS.md

`uq-pet-token` compares sequential acquisition of uncertain NER words with random
acquisition of the same number of words, on PET (default) or CoNLL-2003. Sentence-level
LLM selection lives in the separate `DogRog/uq-pet` repository; never port code between
the two.

`llm-annotation-eval/` holds the LLM baseline for the same paper: a separate uv project with
its own environment, `pyproject.toml`, and `CLAUDE.md`; run its commands from that folder. It
reads this project's splits (through `scripts/uq-pet-token-splits.py`, run with
`uv run --project ..`) and `results/supervised/`, so changes to `data_prep` splits or that
results layout must keep its `--check` passing.

- Terms: `GLOSSARY.md`. Use its words in code, docs, and conversation.
- Decisions and their reasons: `docs/adr/`. Read the ADR before changing what it covers.
- Protocol and its invariants: the Protocol section of `README.md`.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests notebooks scripts
uv run ruff format --check src tests notebooks scripts
uv run marimo check notebooks/bert_token_uq.py
```

`uv run notebooks/bert_token_uq.py` renders synthetic records; it downloads and trains
only with a validated `--config-json`.

A project hook runs `marimo check` after every notebook edit. For a live notebook, the
user starts `MARIMO_ENABLE_HOST=1 uv run marimo edit notebooks/ --watch --no-token`;
use `uv run marimo pair` against it to read errors, outputs, and variables.
`pyproject.toml` makes it re-import edited `notebooks/analysis` modules on its own.

## Rules

- Scoring takes label-free pool inputs; read a pool label only after its candidate is
  selected. The test split is evaluation-only.
- Selection and replay use local seeded RNGs; global seeding happens only in
  `token_model.set_seed`.
- Every UQ metric returns larger values for more uncertainty; keep the notebook's metric
  selector and the saved `uq_metric` in sync.
- Publish progress after round 0 and after both arms finish a round, never mid-round.
- Outputs record the settings, every evaluation row with its predicted tags, and every
  selected candidate.
- Exact-match entity F1 is the tuning objective and headline gap; partial match is
  reported only (ADR 0010).

## Module boundaries

Keep modules small and focused: overcomplicating slows a fast-changing codebase. Add
YAML configs, a CLI framework, registries, callbacks, checkpoint management, or generic
experiment abstractions only for a concrete need.

- `config.py`: validated experiment and search settings; UI controls stay in the notebook
- `paths.py`: the project root and results directory
- `data_prep.py`, `token_model.py`, `active_learning.py`: data, model operations, rounds
- `experiment.py`: one interactive-notebook run, persisted
- `search.py`: search planning, resume handling, and summaries
- `sweeps.py`: the learning-rate range and saved sweep files that search and supervised share
- `supervised.py`: the supervised baseline's search, tuning, test runs, and resume
- `utils/wandb_logging.py`: shared W&B logging
- `utils/wandb_tuning.py`: W&B tuning sweeps and their workspace charts
- `utils/truncation.py`: which words survive subword truncation
