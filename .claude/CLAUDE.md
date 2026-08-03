# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An experiment testing whether LLM uncertainty quantification is a good criterion for
selecting training data (vs. random selection) on the PET process-extraction NER
dataset. The grid is budgets x arms x training seeds. See README.md for the design and
how to read the results.

## Commands

```bash
uv sync
uv run pytest                          # offline: no network, no API key, no model weights
uv run pytest tests/test_uncertainty.py
uv run pytest -m slow                  # the one test that downloads a checkpoint
uv run ruff check .

# Sanity run: hits the existing score cache, makes no API calls, ~10s
uv run python -m uq_pet.main --config configs/smoke.yaml --skip-scoring
# Real run
uv run python -m uq_pet.main --config configs/nhr_gemma.yaml
```

There is no console script and no subcommands — `main.py` is the single entry point.

## Architecture

Eight modules in `src/uq_pet/`, in dependency order:

1. **`config.py`** — cache-identity constants, project paths, and the YAML-backed
   dataclasses (`ExperimentConfig` → `LLMConfig` + `TrainConfig`). `load_config` raises
   on unknown keys.
2. **`dataset.py`** — PET download/load, the few-shot/pool/test split (5/328/84),
   `to_examples()` (Dataset → plain dicts) and `sentence_key()`.
3. **`prompt.py`** — the system/user `string.Template`s, `prompt_fingerprint()`, and
   `parse_tag_ids()` (the response format is dictated by the template, so one module
   owns both halves of the contract).
4. **`llm.py`** — repeated sampling through an OpenAI-compatible gateway into a
   resumable JSONL cache. Nothing else does network or cache I/O.
5. **`uncertainty.py`** — the `METRICS` registry **and** the one selection rule,
   `select` = top-n of a metric's ranking.
   **To add a metric: write one function and `@register` it — nothing else changes.**
   Its keyword-only parameters automatically become the parameters its arm accepts in
   YAML, and `validate_arm` derives them from the signature, so a typo in a config
   fails at start-up rather than being silently ignored.
   The `random` control is a registered metric like the rest — uniform scores, so
   top-n of them is a uniform sample of n — which is why nothing downstream branches on
   it. The single exception is in `stage_select`: `random` is scored over one bare
   record per pool sentence instead of over the cache, so the control draws from the
   whole pool and `arms: [random]` runs without a cache at all. `main` and `plotting`
   still use the `RANDOM` name, but only to find the baseline for reporting.
6. **`model_training.py`** — the fixed distilbert recipe, prediction, seqeval metrics.
7. **`plotting.py`** — the palette and the two figures, reading the results DataFrame.
   Presentation only: nothing here is imported by a stage that produces a number, so a
   layout or color change cannot move a reported score. matplotlib is imported *inside*
   the plotting functions because `matplotlib.use("Agg")` is a global side effect.
8. **`main.py`** — the pipeline and its argparse CLI. **Nothing imports from `main`.**

`config.py` imports nothing from the package and must stay that way, so `ArmConfig`
validates only structure; metric names and their parameters are checked by
`uncertainty.validate_arm`, called at the top of `main()`. Note `config_to_yaml` needs
`arm_to_dict` rather than plain `asdict` — `asdict` emits a nested shape `load_config`
cannot read, which would break every run's `config.yaml` snapshot.

### The rule that matters most

**Every module is import-side-effect-free**: no I/O, no globals built from I/O, no
logging reconfiguration at import time. This repo was broken once by pasting notebook
cells into modules verbatim — four of seven modules raised `NameError` on import and
ran a `ThreadPoolExecutor` at module scope. If you are moving code out of
`notebooks/test.ipynb`, turn the cell-level globals into function parameters first.

Corollaries already enforced: HF logging is configured only via
`model_training.configure_hf_logging()`, called from `main()`; the OpenAI client is
built lazily inside `score_split` so a complete cache needs no API key; and
`random.seed()` is called only by `model_training.set_seed()` — selection uses local
`random.Random` instances so it can't perturb training determinism.

## Cache invariants — don't break these

The cache is `data/processed/llm_scores/nhr_gemma_pool_dist.jsonl`: 328 records, ~1,640
gateway generations. It is gitignored and not backed up anywhere.

1. **Records are keyed by `idx` = position in the `pool` split.** So `SEED`,
   `TEST_SIZE`, `N_FEW_SHOT_EXAMPLES`, `FEW_SHOT_SPLIT_SEED` in `config.py` and the
   exact `datasets==2.19.2` pin *define* the cache. `SEED` and `TEST_SIZE` are module
   constants rather than YAML knobs on purpose — changing one silently repoints every
   record at a different sentence. `tests/test_dataset.py::test_split_sizes_and_first_key`
   is the canary; `llm.verify_cache_alignment()` gates every run and must stay above
   0.80 (measured 0.957 when aligned, <0.05 under any shift).

   The two few-shot values are the defaults of the YAML knobs
   `ExperimentConfig.n_few_shot` / `few_shot_seed`. Overriding either is safe *only*
   because `few_shot_tag()` then appends `_fs<n>s<seed>` to the cache filename, giving
   that split its own file. Anything that reads the pool cache must pass
   `tag=cfg.few_shot_tag()` to `cache_path`, or a non-default run will append records
   into the 328-record cache under indices that mean something else.
2. **`LLMConfig()`'s defaults must keep producing `nhr_gemma_pool_dist.jsonl` and the
   cached `params` dict.** `tests/test_config.py` locks both. `smoke.yaml` deliberately
   omits every sampling field so it inherits those defaults and hits the same cache.
3. **The filename does not encode the sampling parameters** — `load_cache` validates
   `model`/`params`/`prompt_sha`/`key` against the config in hand instead, and drops
   what disagrees. So if you deliberately change the recipe, bump `llm.cache_suffix`;
   otherwise you'll re-score the pool without noticing.
4. **Records predating `key`/`prompt_sha` are grandfathered.** All 328 current records
   are in that category, so editing `SYSTEM_TEMPLATE` will *not* invalidate them
   automatically. Delete the cache yourself when you change a prompt.
5. Failed sentences are deliberately **not** written to the cache, so the next run
   retries them. That looks like a bug in `score_split`; it isn't.
