# uq-pet

Is **LLM uncertainty quantification a good criterion for choosing training data**, compared to random choice? This repo runs that experiment on the [PET dataset](https://github.com/patriziobellan86/PETv1.1) (process-extraction NER, 417 sentences, 15 BIO tags).

## Experiment design

```text
PET NER dataset (417 sentences)
├── 5 few-shot examples (`n_few_shot`, used in every prompt)
├── 328 experiment pool
│     │  LLM repeated sampling (K=5, temp 1.0, with token logprobs)
│     │  → one uncertainty score per sentence, per metric
│     │    (from the logprobs, or from how much the K outputs disagree)
│     ├── top-N% by metric A  → fine-tune distilbert → evaluate
│     ├── top-N% by metric B  → fine-tune distilbert → evaluate
│     └── random N%           → fine-tune distilbert → evaluate
└── 84 held-out test ──────────────────────────────→ used for every evaluation
```

At a given budget every arm gets **exactly the same number of sentences**, the same
recipe and the same seeds — *which* sentences were selected is the only variable. The
grid is `budgets × arms × train_seeds`, because a ~32-sentence fine-tune is noisy
enough that a single-seed gap between arms would not be a result.

- **Arms** are declared in the config as a list of metric names — `random`, the control,
  is one of them:

  ```yaml
  budget_pct: [5, 10, 25]

  arms:
    - random
    - avg_neg_logprob_filtered
    - avg_neg_logprob_pure
  ```

  An arm is labelled by its metric name unless you set an explicit `label:`, which needs
  the mapping form (`- strategy: avg_neg_logprob_pure` plus `label:`) — the same form
  passes a metric parameter, e.g. `random`'s `seed`.
- **Metrics** live in `src/uq_pet/uncertainty.py`, one registry entry per variant. They
  read different halves of a cached sample; the control is registered alongside them, so
  selection has exactly one rule — take the top n of a metric's ranking — and no special
  case:
  - `random` — a uniform random score per sentence, whose `seed` parameter (default 42)
    picks the draw. Ranking by iid uniform scores and taking the top n *is* a uniform
    random sample of n. It scores the whole pool rather than the cache, so the control
    never inherits the LLM's blind spots.
  - `avg_neg_logprob_filtered` — the mean over the K samples of the average negative
    logprob of the tag-ID tokens, higher meaning less confident. Restricting to tag IDs
    keeps brackets and commas from diluting the signal.
  - `avg_neg_logprob_pure` — the same over every token in the response.
  - `vote_entropy` — **outputs only, no logprobs**: parse the K sampled tag arrays and
    average, over token positions, the Shannon entropy of the K votes over log K (so
    0–1). The samples are a committee; the more they vary, the less settled the model
    is. Samples that disagree about the token count count as disagreeing (a sample that
    has ended votes a distinct "missing" tag), so a truncated response reads as
    uncertain rather than being dropped. Because it needs no logprobs, it also works
    against a gateway that doesn't return them.
  - `disagreement` — the same committee reading, scored as the fraction of samples off
    the plurality tag instead of the entropy.
- **Few-shot prompt**: `n_few_shot` (default 5) and `few_shot_seed` (default 42) choose
  the demonstrations shown to the LLM. They also decide which sentences are held out of
  the pool, so changing either changes both the prompt and the size of the pool
  (`n_few_shot: 10` gives 10 / 323 / 84). A non-default pair scores into its own cache
  file — `nhr_gemma_pool_fs10s42_dist.jsonl` — so it costs a fresh pass over the pool
  and cannot corrupt the default one.
- **Trained model**: `distilbert-base-cased` token classifier, manual torch loop.
- **Evaluation**: entity-level micro F1 (seqeval), per-type F1, token accuracy.

Scores are recomputed from the cached samples on every run and never stored, so
metrics, their parameters, the budgets and the seeds can all be swept **without new
LLM calls**.

### Adding an uncertainty metric

Write one function in `uncertainty.py` and decorate it. Nothing else changes — the
config, the CLI and the reporting pick it up from the registry, and its keyword-only
parameters automatically become the parameters its arm accepts in YAML:

```python
@register("sequence_variance")
def sequence_variance_scores(records, *, normalize: bool = True) -> dict[int, float]:
    ...   # {pool index: score}, higher = less confident
```

## Running

```bash
uv sync
uv run pytest       # offline unit tests — no network, no API key, no model weights

# Sanity run: 9 sentences per arm, 2 epochs, ~10s. Hits the existing score cache,
# so it makes no API calls at all.
uv run python -m uq_pet.main --config configs/smoke.yaml --skip-scoring

# The real run.
uv run python -m uq_pet.main --config configs/nhr_gemma.yaml
```

The raw PET jsonl is downloaded on the first run. `NHR_FAU_API_KEY` in `.env` is
needed **only** when the score cache doesn't already cover the pool — a complete cache
means no client is ever constructed.

Useful flags: `--skip-scoring` (never call the API; fail if the cache is short),
`--limit N` (score only the first N pool sentences), `--dry-run` (stop after selection
and print both arms), `--run-name`, `--no-plot`, `-v`.

### Conda instead of uv

If you prefer conda (e.g. on a managed CUDA server), the platform markers in
`pyproject.toml` handle the rest (torch picks the CUDA build on Linux):

```bash
conda create -n uq-pet python=3.12 -y
conda activate uq-pet
pip install --use-pep517 seqeval   # seqeval's legacy setup.py build is broken; force PEP 517
pip install -e .                   # add: pip install pytest  — to run the tests
```

Then drop the `uv run` prefix. Note conda installs won't match `uv.lock` exactly — pip
resolves fresh from `pyproject.toml`, so use uv when you need the pinned versions.

If the env came with torch preinstalled (typical on Jupyter images), pip's torch
upgrade will strand the old `torchvision`/`torchaudio` builds, and transformers then
crashes with `operator torchvision::nms does not exist`. This project needs neither —
`pip uninstall -y torchvision torchaudio` fixes it. Afterwards confirm the GPU is still
visible: `python -c "import torch; print(torch.cuda.is_available())"`.

### Changing the prompt

The prompt is a pair of `string.Template`s in `src/uq_pet/prompt.py`. Editing either
changes `prompt_fingerprint()`, which is stamped onto every new cache record. Records
written before fingerprinting exist are grandfathered in, so **nothing will stop you
from mixing two prompts in one cache** — if you edit a template, bump `llm.cache_suffix`
in your config or delete `data/processed/llm_scores/*.jsonl`.

## Layout

Modules are listed in dependency order. Every one is import-side-effect-free: importing
`uq_pet.anything` does no I/O, builds no globals from I/O, and reconfigures no logging.

| Path | Purpose |
| ---- | ------- |
| `configs/` | YAML run definitions (`nhr_gemma.yaml` real, `smoke.yaml` cheap) |
| `data/raw/` | downloaded PET jsonl (gitignored, never edited by hand) |
| `data/processed/llm_scores/` | cached LLM samples, resumable and shared across runs (gitignored) |
| `src/uq_pet/config.py` | tags, project paths, cache-identity constants, config dataclasses + YAML loader |
| `src/uq_pet/dataset.py` | PET download/loading, the few-shot/pool/test split, sentence keys |
| `src/uq_pet/prompt.py` | prompt templates and the parser for the format they ask for |
| `src/uq_pet/llm.py` | repeated sampling through an OpenAI-compatible gateway, JSONL cache |
| `src/uq_pet/uncertainty.py` | the metric registry (the `random` control included) **and** the one selection rule |
| `src/uq_pet/model_training.py` | fine-tuning, prediction, seqeval metrics |
| `src/uq_pet/plotting.py` | the figures — presentation only, nothing here feeds back into a number |
| `src/uq_pet/main.py` | the whole pipeline and its CLI — nothing imports from here |
| `results/<run_id>/` | one directory per run (gitignored) |
| `tests/` | offline unit tests |

Each run directory holds a `config.yaml` snapshot (written before any work),
`selection.json` (which sentences each arm got at each budget — the reproducibility
audit trail), `results.csv`, `summary.csv`, `per_type_f1.csv`, `metrics.json`,
`run.log`, and `figures/arm_f1.png`.

## Reading the results

`figures/arm_f1.png` is the decision plot — a learning curve across budgets, or a bar
chart when there is only one. The curve's error bars are +/-1 std over `train_seeds`,
and its budgets are evenly spaced categories rather than points on a linear axis.
Compare `metrics.json`'s `by_budget[…].gap_vs_random` against those bars: if a gap is
smaller than the seed-to-seed variation, there is no result yet, only noise — and if
two arms' bars overlap at a budget, they are tied there. Add seeds before believing a
small gap.

Three caveats the numbers won't show you:

- **The random baseline is a single draw.** All `train_seeds` share one selected set per
  (budget, arm) — the seeds vary training, not selection. So the error bars cover
  training noise but not selection noise, and part of any gap could be luck of that one
  draw. Since `random` is an ordinary metric, a second control arm
  (`- strategy: random` / `seed: 7` / `label: random-7`) costs nothing but training time
  and shows you how wide that luck is.
- **Every arm is nested across budgets**, the control included: each is the top n of one
  fixed ranking, so the 25% set contains the 10% set.
- The score is an unnormalized average negative logprob, so it mildly favors sentences
  the model finds hard *anywhere* in the response. Longer sentences buy more tokens per
  budget — a known confound; `per_type_f1.csv` helps show whether a gap is concentrated
  in one entity type.
- `metrics.json` reports `n_truncated`: choices that hit the `max_tokens` ceiling, whose
  logprobs therefore cover a cut-off array. With the shipped config that's 13 of 1,640.
