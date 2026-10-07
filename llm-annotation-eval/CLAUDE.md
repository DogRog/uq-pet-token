# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Uses LLMs to annotate business process descriptions from the PET dataset with process element mentions (Activity, Actor, Activity Data, XOR/AND Gateway, Condition/Further Specification) and the relations between them (flow, uses, actor performer/recipient, same gateway, further specification), then scores predictions against the gold annotations in `res/gold/pet.jsonl` (45 documents). The results are reported in a paper, so runs log everything needed to describe them.

## Commands

Environment is managed with uv (Python ≥3.10, `package = false` — nothing is built or installed; scripts run directly from `src/`, so `src/` is the import root for `autoannotate`, `eval`, `experiments`, `data`).

```bash
uv sync
uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml   # annotate + score one config
uv run python src/run-configs.py configs/md/ --workers 4            # every config in a folder
uv run marimo edit notebooks/eval_table.py   # results table: scores every pet.jsonl in res/pred/
uv run python src/export-test-predictions.py # test-predictions.jsonl for uq-pet-token, per run
uv run marimo check notebooks/eval_table.py  # lint the notebook
uv run pytest                       # all tests
uv run pytest tests/test_mentions.py -k repeated   # a subset
uv run ruff check src/ tests/      # lint; add --fix for the auto-fixable ones
```

Tests in `tests/` never call an LLM: `conftest.py` provides `make_doc` (builds a `Document` from whitespace-tokenized sentences) and `fake_client` (stands in for `openai.OpenAI`; the answer can be a function of the request, which may raise). Known bugs are kept as `xfail(strict=True)` tests, so fixing one makes its test fail until the marker is removed. Ruff is configured in `pyproject.toml` (Google docstring convention, 100 columns, missing docstrings allowed, no docstring rules in `tests/`); the code does not pass it cleanly yet.

API keys come from `.env` (`OPENAI_API_KEY`, `NHR_API_KEY`; see `.env.template`).

Source files mix CRLF and LF line endings; keep each file's existing endings when editing.

## Experiments

A run config is a YAML file (`*.yaml`/`*.yml`) in `configs/` or a subfolder like `configs/md/` (keys: `task` (`end-to-end` or `md` = mentions only), `provider`, `model`, `reasoning_effort`, `max_documents`, `repeats`; see `configs/README.md`, which also records which `reasoning_effort` values each NHR model accepts). The file stem is the config name. `src/run-configs.py` validates all configs and API keys before the first call, then for each config and repeat: `experiments.runner.annotate_run` → `experiments.report.score_run`; a failing run skips the rest of its config. Finally it rebuilds `res/pred/runs.csv` and `res/pred/summary.csv` from **every** run folder on disk.

Run folder `res/pred/<config>/run-<i>/`: `<task>/pet.jsonl` (predictions), `calls.jsonl` (one line per LLM call: tokens, duration, finish reason, raw response, reasoning, parser issues), `run.json` (settings, rendered prompts, a session per start with git commit/dirty flag, package versions, gold sha256, plus totals), `metrics.json`. Documents are annotated by a thread pool (`--workers`); `_write_lock` guards the shared files, and `BaseAnnotator.annotate` creates a parser per call so issues don't mix between threads. Transient API errors (`runner.TRANSIENT_ERRORS`) skip the document (session outcome `incomplete`); other errors or `MAX_CONSECUTIVE_FAILURES` in a row stop the run. Resuming skips documents already in `pet.jsonl` but raises if `RunConfig.settings()` (task, provider, model, reasoning effort) or the rendered prompts differ from `run.json`. `totals` counts only the latest successful call per (document, step) of completed documents; the rest are `discarded_calls`.

Mention/relation definitions and few-shot examples are shared by all configs, in `src/experiments/definitions.py`.

## Pipeline architecture

Two LLM calls per document, end to end:

1. **Mentions** — `LLMMentionAnnotator` sends the document as numbered sentences (`i: tok tok tok`) and asks for `text<TAB>type<TAB>sentence_id` lines. `MentionParser` maps those back to token indices using LangExtract's `WordAligner` per sentence (exact match first, fuzzy fallback; a fuzzy match may not reuse tokens already claimed by an exact match).
2. **Relations** — `LLMRelationsAnnotator` sends the text with predicted mentions inlined as XML-ish tags carrying their list index (`<actor id=1> clerk </actor>`, via `util.format_document_text_with_entity_mentions`) and asks for `type<TAB>head_id<TAB>tail_id` lines. `RelationParser` turns these into `Relation`s that reference mentions **by index into `document.mentions`** — so mention order/removal matters (`Document.remove_mention` reindexes relations).

Parsers never raise on bad LLM output: they skip the line and call `BaseParser.report(kind, message)`, which prints and records it. Kinds are prefixed `dropped_` (line not used), `placed_` (used after fuzzy/out-of-order alignment) or are `ignored_line` (no tab, e.g. chatter); `runs.csv` sums the `dropped_` ones. `BaseAnnotator.annotate` returns an `LLMAnnotation` with the parsed doc plus everything the run log needs (tokens incl. `reasoning_tokens` when reported, raw response, `reasoning_content`/`reasoning` from the message's extra fields, finish reason, duration, issues).

Shared plumbing in `src/autoannotate/`:
- `base.BaseAnnotator`: builds chat-completion params; prompts are authored with a `developer` role and rewritten to the provider's `instruction_role` (`system` for NHR/vLLM-served open-weight models). `reasoning_effort=None` omits the parameter. `get_instruction()` on the mention/relation annotators returns the document-independent instruction prompt.
- `providers.py`: `PROVIDERS` registry; NHR@FAU is an OpenAI-compatible gateway (LiteLLM), so everything goes through the `openai` client. The gateway caches responses to identical requests, so `Provider.extra_body` sends `{"cache": {"no-cache": True}}` with every NHR request; without it, repeated runs are copies of the first.
- `prompts.Prompt`: templates in `resources/prompts/*.txt` with `{{param}}` placeholders; missing or unknown params assert.
- `data.py`: `Document`/`Token`/`Mention`/`Relation`/`Entity` plus JSONL import/export. The importer lowercases mention and relation types, so gold `"Activity Data"` and predicted types compare as `"activity data"`.
- `entities.py` (`LLMEntitiesAnnotator`, coreference into entities) and `BaseAnnotator.batch_line` exist but are not used by the current pipeline.

## Evaluation

All matching lives in `src/eval/scoring.py`, used by both the experiment reports and the results notebook. Matching is lenient: a predicted mention matches a gold one if types are equal and token sets **overlap at all**; a relation matches if type is equal and both head and tail mentions overlap. Matching is greedy one-to-one per document. `scoring.summarize` reports macro (mean ± population std of per-document P/R/F1), micro (summed counts) and per-type micro scores. A document with no gold and no predicted annotations scores 1.0 (`eval.metrics.Stats`). `summary.csv` uses the sample SD across runs.

`notebooks/eval_table.py` scores every `res/pred/<folder>/run-<i>/<task>/pet.jsonl` (config runs) and `res/pred/<folder>/<task>/pet.jsonl` (predictions from before the configs) with `scoring.score` rather than reading `metrics.json`; the folder name says what to score: `end-to-end` (`md` and `re`), `md`, or `re`/`isolated-re`. Notebook cells must not define the same name twice, so cell-local variables start with `_` (`uv run marimo check` catches it).

This project is a baseline for `../uq-pet-token`, which trains token classifiers on PET v1.1 and evaluates on a fixed 84-sentence test split. `src/eval/test_split.py` reproduces that evaluation: `res/gold/pet-v1.1-test.jsonl` is the split (generated with uq-pet-token's `load_pet_splits`, same format and order; PET v1.1 labels differ from `res/gold/pet.jsonl` in 8 test sentences), predicted mentions of test sentences become BIO tags (overlaps: first listed wins; unknown types kept so they count as wrong), and `evaluate` mirrors uq-pet-token's `evaluate_predictions` with seqeval 1.2.2 (strict span match, micro). `export` writes `test-predictions.jsonl` next to a run's `pet.jsonl`; `run-configs.py` calls it after scoring each run. Task `md-test-sentences` annotates each test sentence on its own: `runner.input_documents` turns the test split into one-sentence documents with ids `<document>#<sentence-ID>` (`test_split.sentence_documents`), `run-configs.py` passes the test file as the run's input, and `report.score_run` scores those runs only on the test split. `metrics.json` of every run that covers the test sentences has a `test` entry, which `runs.csv`/`summary.csv` show as `test_*` columns. Verified identical to uq-pet-token's own scorer on real predictions.

`src/data/pet.py` is an older parallel copy of the data model (`PetDocument`, etc.); only `src/eval/metrics.py` imports it. Everything else uses `autoannotate.data`.
