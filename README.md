# uq-pet: BERT uncertainty

Does uncertainty from the same token-classification model that will be trained provide
a better PET NER training set than random selection?

The earlier LLM-scored experiment is preserved on the `llm-uq` branch. This branch,
`bert-uq`, uses a seed-trained BERT-family model for both grading and training.

## Experiment design

```text
PET NER dataset (417 sentences)
├── 5 labelled seed sentences
│     └── fit one PET token classifier per model seed
├── 328 experiment-pool sentences
│     └── seed-trained classifier -> word probabilities -> sentence uncertainty
│           ├── top N% by uncertainty
│           ├── most-confident / length controls
│           └── random N%
│
│     For every arm and budget:
│       clone the same seed-trained weights
│       -> continue training on seed + selected sentences
└── 84 held-out test sentences -> entity-level NER evaluation only
```

The central invariant is that grading and training share not just a checkpoint name,
but the exact fitted parameter state. For one model seed, the seed set is trained once;
every arm and budget starts from an independent clone of that state. The pool's gold
labels are not read during scoring, and the test split is used only for evaluation.

Selection depends on the model seed, so the grid is:

```text
model_seeds x budgets x arms
```

The reported spread therefore includes both seed-dependent selection and continuation
training, rather than training noise alone.

## Uncertainty arms

All scores are sentence-level and increase with uncertainty:

- `mean_token_entropy`: mean normalized predictive entropy over words.
- `max_token_entropy`: entropy of the least-certain word.
- `least_confident`: mean `1 - max(class probability)` over words.
- `margin`: mean inverse gap between the two most likely tag classes.
- `confident`: reverse another metric, defaulting to `mean_token_entropy`.
- `length`: sentence-length control.
- `random`: uniform random-selection control. Unless its `seed` parameter is explicit,
  it follows the model seed.

Only the first subword represents each word, matching training-label alignment. Padding
and special tokens never enter uncertainty. Truncated word counts are recorded in
`metrics.json`.

## Seed training and the untrained ablation

`uq.bootstrap_epochs` controls how long the PET classifier is fitted on the labelled
seed set before it scores the pool. The default is 20. Setting it to zero is supported
as an explicit untrained-head ablation:

```yaml
uq:
  bootstrap_epochs: 0
```

That run is technically valid, but its entropy primarily describes the randomly
initialized PET classification head rather than learned PET NER uncertainty.

The default five seed sentences contain 11 of the 15 BIO tags. They omit `I-Activity`,
`I-XOR Gateway`, `B-AND Gateway`, and `I-AND Gateway`. The split is retained so results
remain directly comparable with `llm-uq`; seed-size experiments should report label
coverage as a limitation.

## Running

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .

# One seed, one bootstrap epoch, one continuation epoch.
uv run python -m uq_pet.main --config configs/smoke.yaml

# Primary experiment.
uv run python -m uq_pet.main --config configs/bert_uq.yaml

# Validate the unattended sweep without running it.
DRY_RUN=1 scripts/run_all.sh
scripts/run_all.sh
```

The first run downloads `train.checkpoint` if it is not already in the Hugging Face
cache. There are no LLM requests, API keys, prompts, or generation caches.

`--dry-run` still fits the seed model and scores the pool, then prints selections and
stops before continuation training. `scripts/run_all.sh`'s `DRY_RUN=1` is cheaper: it
only validates configs and arms.

## Configuration

```yaml
budget_pct: [5, 10, 25]
arms:
  - random
  - mean_token_entropy
  - max_token_entropy
  - strategy: confident
    metric: mean_token_entropy
model_seeds: [0, 1, 2]
n_seed: 5
seed_split_seed: 42

uq:
  bootstrap_epochs: 20
  score_batch_size: 32

train:
  checkpoint: distilbert-base-cased
  epochs: 20
  batch_size: 8
  learning_rate: 5.0e-5
  max_length: 256
```

There is intentionally only one checkpoint field. Both the uncertainty scorer and all
continuation models read `train.checkpoint`.

## Outputs

Each run writes a new timestamped directory under `results/` containing:

- `config.yaml`: fully resolved configuration.
- `selection.json`: selected sentence keys by model seed, budget, and arm.
- `results.csv`: one evaluation row per model seed, budget, and arm, including
  `n_seed`, `n_selected`, and `n_train`.
- `summary.csv`, `per_type_f1.csv`, and `metrics.json`.
- `figures/arm_f1.png` and `run.log`.

Entity-level micro F1 is the primary result. `gap_vs_random` in `metrics.json` compares
each arm's mean across model seeds with random at the same budget.

## Layout

| Path | Purpose |
| --- | --- |
| `src/uq_pet/config.py` | experiment, UQ, and training configuration |
| `src/uq_pet/dataset.py` | PET loading and seed/pool/test split |
| `src/uq_pet/model_training.py` | load, seed-fit, clone, continue, and evaluate models |
| `src/uq_pet/bert_uq.py` | word-level predictive probability records |
| `src/uq_pet/uncertainty.py` | BERT UQ metrics, controls, ranking, and selection |
| `src/uq_pet/plotting.py` | result figures |
| `src/uq_pet/main.py` | end-to-end CLI pipeline |
| `configs/` | primary and smoke experiment definitions |
| `tests/` | offline unit tests plus opt-in slow model tests |

Every package module remains import-side-effect-free. Network access and model loading
happen only after entering `main()` or explicitly calling a training function.
