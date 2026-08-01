# uq-pet

Is **LLM uncertainty quantification a good criterion for choosing training data**, compared to random choice? This repo runs that experiment on the [PET dataset](https://github.com/patriziobellan86/PETv1.1) (process-extraction NER, 417 sentences, 15 BIO tags).

## Experiment design

```text
PET NER dataset (417 sentences)
├── 5 few-shot examples (fixed, used in every prompt)
├── 328 experiment pool
│     │  LLM repeated sampling (K=5, temp 1.0, with token logprobs)
│     │  → one uncertainty score per sentence
│     ├── top-N% most uncertain  → fine-tune distilbert → evaluate
│     └── random N%              → fine-tune distilbert → evaluate
└── 84 held-out test ─────────────────────────────────→ used for both evaluations
```

Both arms get **exactly the same number of sentences**, the same recipe and the same
seeds — *which* sentences were selected is the only variable. Each arm is trained once
per seed in `train_seeds`, because a ~32-sentence fine-tune is noisy enough that a
single-seed gap between the arms would not be a result.

- **Uncertainty score**: mean over the K samples of the average negative token
  logprob — higher means the model was less confident. Two variants, set by `score:`
  in the config:
  - `pure` — every token in the response;
  - `filtered` — only the tag-ID tokens, so brackets and commas (which the model is
    always confident about, and which there are more of in long sentences) don't
    dilute the signal.
- **Trained model**: `distilbert-base-cased` token classifier, manual torch loop.
- **Evaluation**: entity-level micro F1 (seqeval), per-type F1, token accuracy.

Scores are recomputed from the cached samples on every run and never stored, so the
score variant, the budget and the seeds can all be changed without new LLM calls.

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
| `src/uq_pet/uncertainty.py` | uncertainty scoring **and** the selection strategies it's compared against |
| `src/uq_pet/model_training.py` | fine-tuning, prediction, seqeval metrics |
| `src/uq_pet/main.py` | the whole pipeline and its CLI — nothing imports from here |
| `results/<run_id>/` | one directory per run (gitignored) |
| `tests/` | offline unit tests |

Each run directory holds a `config.yaml` snapshot (written before any work),
`selection.json` (which sentences each arm got — the reproducibility audit trail),
`results.csv`, `summary.csv`, `per_type_f1.csv`, `metrics.json`, `run.log`, and
`figures/arm_f1.png`.

## Reading the results

`figures/arm_f1.png` is the decision plot: bar = mean test F1 over seeds, dots = the
individual seeds. Compare `metrics.json`'s `gap` (uncertainty minus random) against the
spread of those dots — if the gap is smaller than the seed-to-seed variation, there is
no result yet, only noise. Add seeds before believing a small gap.

Two caveats the numbers won't show you:

- The score is an unnormalized average negative logprob, so it mildly favors sentences
  the model finds hard *anywhere* in the response. Longer sentences buy more tokens per
  budget — a known confound; `per_type_f1.csv` helps show whether a gap is concentrated
  in one entity type.
- `metrics.json` reports `n_truncated`: choices that hit the `max_tokens` ceiling, whose
  logprobs therefore cover a cut-off array. With the shipped config that's 13 of 1,640.
