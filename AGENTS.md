# AGENTS.md

## Branch purpose

`bert-token-uq` compares sequential acquisition of uncertain PET NER words with the
same number of random words. The implementation is intentionally small: three focused
modules and one marimo notebook.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests notebooks
uv run ruff format --check src tests notebooks
uv run marimo check notebooks/bert_token_uq.py
uv run notebooks/bert_token_uq.py
uv run marimo edit notebooks/bert_token_uq.py
```

Plain script mode uses synthetic display records and must not download or train a
model. A real run starts only from the notebook's **Run experiment** button.

## Experiment invariants

- Keep the default 5 seed / 328 pool / 84 test split and `datasets==2.19.2` pin.
- The test split is evaluation-only.
- Scoring accepts label-free pool inputs. Read a pool label only after selection.
- A candidate is `(pool_sentence_index, word_index)` and cannot be selected twice.
- Score, supervise, and predict from the first subword only.
- Mask every other training position with `-100`; do not replace input words by
  `[MASK]`.
- Exclude truncated pool words and reject truncated evaluation sentences.
- Clone one bootstrap-trained model and optimizer state into both arms. Thereafter
  each arm keeps its own state across rounds.
- Each round selects exactly `K` new tokens per arm and uses the same replay ratio,
  update passes, batch size, and learning rate.
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
need. Keep experimental controls in the marimo notebook; keep data, model operations,
and active-learning orchestration in their existing focused modules.
