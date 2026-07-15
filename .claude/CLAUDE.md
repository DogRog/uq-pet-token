# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An experiment testing whether LLM uncertainty quantification is a good criterion for selecting training data (vs. random selection) on the PET process-extraction NER dataset. See README.md for the experiment design and how to read the results.

## Commands

```bash
uv sync                                    # install (mlx deps are darwin-only via platform markers)
uv run pytest                              # offline unit tests — no network, no API key needed
uv run pytest tests/test_uncertainty.py   # single test file
uv run pytest tests/test_selection.py::test_name -v   # single test

# Pipeline (each step feeds the next)
uv run uq-pet download-data                          # PET jsonl → data/raw/
uv run uq-pet score-pool --config configs/<cfg>.yaml # LLM sampling → data/processed/llm_scores/
uv run uq-pet run --config configs/<cfg>.yaml        # selection→train→eval grid → results/<run_id>/
uv run uq-pet run --resume <run_id>                  # continue an interrupted run
uv run uq-pet report [--run-id <run_id>]             # figures + summary.md (default: latest run)
```

`configs/smoke.yaml` is the cheap sanity config (2 cells, 2 epochs; reuses grid_full's score cache). `score-pool --limit N` scores only N sentences for smoke testing. `scripts/run_grid_*.sh` chain the whole pipeline per config.

API scoring (`llm.backend: openrouter`) needs `OPENROUTER_API_KEY` in `.env`; local backends (`mlx` on Apple Silicon, `hf` on CUDA) need no key.

## Architecture

Pipeline stages, each a module in `src/uq_pet/`:

1. **`data.py`** — downloads/loads PET, does the fixed 80/20 pool/test split (seed 3407, `SEED` in config.py). `sentence_key()` is the stable cache key for a sentence.
2. **`llm_scoring.py`** — samples the LLM K times per sentence and caches raw responses + parsed tag sequences as JSONL in `data/processed/llm_scores/`. Backends: `openrouter` (async API), `mlx_scoring.py` and `hf_scoring.py` (in-process, additionally record per-token `token_entropies` for white-box metrics). Caches are resumable (existing keys skipped) and shared across runs.
3. **`uncertainty.py`** — metric registry (`METRICS`), populated via the `register(name, box)` decorator. "black" metrics score disagreement between the K parsed samples (any backend); "white" metrics read `token_entropies` (local backends only). Metrics are **recomputed from the cache at run time, never stored**, so budgets/metrics/repeats can be swept without new LLM calls. Also holds selection strategies (`random`, `uncertainty:<metric>`, `full`).
4. **`experiment.py`** — orchestrates the (budget × strategy × seed) grid, writing per-cell `records.jsonl` to `results/<run_id>/`. Completed cells are skipped on `--resume`. `workers > 1` trains cells in parallel processes.
5. **`train.py`** — fixed distilbert fine-tuning recipe (manual torch loop); the selected data is the only variable across cells. Metrics via seqeval (entity-level micro F1).
6. **`reporting.py`** — learning curves, summary table, UQ-vs-error diagnostic from a run dir. `llm_eval.py` adds the LLM-alone baseline (needs the test split scored: `score-pool --split test`/`both`).

Config: dataclasses in `config.py` (`ExperimentConfig` → `LLMScoreConfig` + `TrainConfig`), loaded from YAML in `configs/`; unknown keys raise. All project paths are constants in `config.py` (overridable root via `UQ_PET_ROOT`).

## Cache invariants — don't break these

- `LLMScoreConfig.cache_path()` derives the cache filename from model/K/temperature/seed/prompt/split. The default prompt and pool split keep the **historical filename** (no suffix) so old caches stay valid — preserve that behavior when touching it.
- **Never edit `prompts/ner_v1.txt` in place**: the rendered prompt text is fingerprinted in existing cache headers, and a mismatch makes `score-pool` refuse the cache. New prompt variants go in a new file (`prompts/ner_v2.txt`) selected via `llm.prompt` in the config, which gets its own cache.
- The few-shot example is always pool sentence `FEW_SHOT_EXAMPLE_INDEX` (also when scoring the test split), so prompts match the pool cache and no test sentence appears in its own prompt.
