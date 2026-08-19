# AGENTS.md

## What this branch is

This is the `bert-uq` experiment branch. It tests whether uncertainty from the same
PET token-classification model that is subsequently trained selects better labelled
data than random choice. The earlier LLM grader and its caches are preserved on the
`llm-uq` branch and intentionally do not exist here.

## Commands

```bash
uv sync
uv run pytest
uv run pytest tests/test_uncertainty.py
uv run pytest -m slow
uv run ruff check src tests
uv run ruff format --check src tests

uv run python -m uq_pet.main --config configs/smoke.yaml
uv run python -m uq_pet.main --config configs/bert_uq.yaml
DRY_RUN=1 scripts/run_all.sh
scripts/run_all.sh
```

The smoke and real runs need Hugging Face checkpoint weights but no API key. The CLI
has no subcommands; `main.py` is the single entry point.

## Experiment protocol

The split is 5 labelled seed / 328 pool / 84 held-out test by default.

For each `model_seed`:

1. Load `train.checkpoint` with a fresh 15-tag PET classification head.
2. Fit it on the labelled seed set for `uq.bootstrap_epochs`.
3. Score the pool without reading its labels, using first-subword probabilities.
4. Select one set per arm and budget.
5. Clone the exact seed-trained parameter state independently for every cell.
6. Continue each clone on `seed + selected` examples.
7. Evaluate on the test set.

Thus scorer and learner share the fitted state, not merely the encoder name. Never
train cells sequentially from one another: every arm and budget must start at the same
seed snapshot or selection is no longer the only cell-level difference.

`uq.bootstrap_epochs: 0` is a supported untrained-head ablation. Do not make it the
default or describe its entropy as learned PET NER uncertainty.

## Module order

1. `config.py`: constants, YAML-backed dataclasses, and paths. It imports no package
   module and must remain that way.
2. `dataset.py`: PET loading and the seed/pool/test split.
3. `model_training.py`: token alignment, model loading, fitting, state snapshots,
   continuation training, prediction, and evaluation.
4. `bert_uq.py`: label-free pool inference into per-word probability records.
5. `uncertainty.py`: metric registry, controls, ranking, and top-n selection.
6. `plotting.py`: presentation only.
7. `main.py`: orchestration and CLI. Nothing imports from `main.py`.

All modules must be import-side-effect-free: no downloads, model loads, file I/O, or
logging reconfiguration at import time.

## Invariants

- The test split is evaluation-only.
- Pool `ner-tags` must not be read before selection. `bert_uq.score_pool` accepts
  examples without that key, and its test pins this property.
- Only the first subword per word contributes to labels, predictions, and UQ.
- Padding and special tokens never contribute to uncertainty.
- Truncated sentences remain selectable, but truncation counts must be reported.
- One checkpoint field (`train.checkpoint`) is the source for both scoring and
  continuation training.
- A budget counts newly selected pool sentences. Outputs separately report `n_seed`,
  `n_selected`, and `n_train`.
- Selection is nested across budgets because every arm ranks once and takes top-n.
- Unless explicitly configured, the random control follows `model_seed`; its variance
  is therefore comparable with seed-dependent UQ selection.
- Every run writes a unique timestamped directory and snapshots the resolved config
  before model work.
- Selection output is keyed by model seed, then budget, then arm.
- Use local RNG objects for selection. `model_training.set_seed` is the only global RNG
  seeding entry point.

## Adding an uncertainty metric

Add one function to `uncertainty.py` and decorate it with `@register`. It receives
records shaped like:

```python
{
    "idx": 0,
    "n_tokens": 12,
    "n_scored_tokens": 12,
    "truncated": False,
    "probabilities": [[... 15 class probabilities ...], ...],
}
```

Return `{pool_index: score}` with larger values meaning more uncertain. Keyword-only
arguments automatically become YAML arm parameters through signature inspection.

## Data comparability

`SEED`, `TEST_SIZE`, `N_SEED_EXAMPLES`, `SEED_SPLIT_SEED`, and the exact
`datasets==2.19.2` pin define the default split. The dataset canary test locks its size
and first pool key so this branch stays comparable with `llm-uq`.

The default five seed sentences omit four BIO tags. This is an experimental limitation,
not permission to inspect pool labels and construct an oracle-stratified seed set.
