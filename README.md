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
4. At each round, acquire `K` previously unseen pool words:
   - uncertainty selects the largest score from the configured UQ metric;
   - random draws uniformly from its remaining word pool.
5. Reveal only the selected labels.
6. Continue each arm from its current weights and optimizer state using the new
   token-centred items plus limited replay.
7. Evaluate on the held-out test set and repeat.

The full sentence remains model input, but each online training item labels exactly
one word. Only its first subword contributes to loss; every other position is `-100`.
By default, each round replays up to `K` older labelled tokens, including seed tokens.

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

The first real run downloads `distilbert-base-cased` if it is not already cached. It
does not require an API key.

## Checks

```bash
uv run pytest
uv run ruff check src tests notebooks
uv run ruff format --check src tests notebooks
uv run marimo check notebooks/bert_token_uq.py
uv run notebooks/bert_token_uq.py
```

Running the notebook as a plain script uses small synthetic display data. It validates
the reactive notebook without downloading weights or starting an experiment.

## Scientific invariants

- The test set is evaluation-only.
- Pool inputs passed to uncertainty scoring contain no labels.
- A label is read from the private pool lookup only after its word is selected.
- Acquisition is without replacement and counts newly labelled words.
- First subwords are used consistently for scoring, online loss, and evaluation.
- Continuation subwords, padding, and special tokens are ignored.
- Truncated pool words are not selectable; truncated test words are an error.
- Both arms begin from the same fitted state and receive the same update budget.
- The uncertainty and random arms keep separate weights and optimizer histories.
- Selection and replay use deterministic local random generators.

## Layout

| Path | Purpose |
| --- | --- |
| `src/uq_pet/pet_data.py` | PET identity, download, stable split, and private label lookup |
| `src/uq_pet/token_model.py` | masking, training, UQ metrics, inference, and evaluation |
| `src/uq_pet/active_learning.py` | acquisition rounds, replay, orchestration, and outputs |
| `notebooks/bert_token_uq.py` | controls, experiment run, tables, and plots |
| `tests/test_token_uq.py` | focused offline invariant tests |
