# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Uses LLMs to annotate business process descriptions from the PET dataset with process element mentions (Activity, Actor, Activity Data, XOR/AND Gateway, Condition/Further Specification) and the relations between them (flow, uses, actor performer/recipient, same gateway, further specification), then scores predictions against the gold annotations in `res/gold/pet.jsonl` (44 documents).

## Commands

Environment is managed with uv (Python 3.12, `package = false` — nothing is built or installed; scripts run directly from `src/`, so `src/` is the import root for `autoannotate`, `eval`, `data`).

```bash
uv sync
uv run python src/llm-annotate.py   # annotate gold docs with an LLM
uv run python src/llm-metrics.py    # score predictions
uv run pytest                       # all tests
uv run pytest tests/test_mentions.py -k repeated   # a subset
```

Tests in `tests/` never call an LLM: `conftest.py` provides `make_doc` (builds a `Document` from whitespace-tokenized sentences) and `fake_client` (stands in for `openai.OpenAI` with a canned answer). `llm-metrics.py` is loaded via `importlib` because of the hyphen in its name. Known bugs are kept as `xfail(strict=True)` tests, so fixing one makes its test fail until the marker is removed.

There is no linter and the scripts take no CLI arguments. Both scripts are configured by editing module-level variables:

- `src/llm-annotate.py` (bottom of file): `provider_name` (`"openai"` or `"nhr"`), `model_name`, `reasoning_effort`, `max_documents` (int for a quick run on the first n docs, `None` for all). The mention/relation type definitions and few-shot format examples fed into the prompts also live in this script.
- `src/llm-metrics.py`: the `models` list of folder names under `res/pred/`.

API keys come from `.env` (`OPENAI_API_KEY`, `NHR_API_KEY`; see `.env.template`).

## Pipeline architecture

`llm-annotate.py` runs two LLM calls per document, end to end:

1. **Mentions** — `LLMMentionAnnotator` sends the document as numbered sentences (`i: tok tok tok`) and asks for `text<TAB>type<TAB>sentence_id` lines. `MentionParser` maps those back to token indices using LangExtract's `WordAligner` per sentence (exact match first, fuzzy fallback; a fuzzy match may not reuse tokens already claimed by an exact match). Unplaceable lines are dropped with a printed message, not raised.
2. **Relations** — `LLMRelationsAnnotator` sends the text with predicted mentions inlined as XML-ish tags carrying their list index (`<actor id=1> clerk </actor>`, via `util.format_document_text_with_entity_mentions`) and asks for `type<TAB>head_id<TAB>tail_id` lines. `RelationParser` turns these into `Relation`s that reference mentions **by index into `document.mentions`** — so mention order/removal matters (`Document.remove_mention` reindexes relations).

Shared plumbing in `src/autoannotate/`:
- `base.BaseAnnotator`: builds chat-completion params; prompts are authored with a `developer` role and rewritten to the provider's `instruction_role` (`system` for NHR/vLLM-served open-weight models). `reasoning_effort=None` omits the parameter (most NHR models reject it).
- `providers.py`: `PROVIDERS` registry; NHR@FAU is an OpenAI-compatible gateway, so everything goes through the `openai` client.
- `prompts.Prompt`: templates in `resources/prompts/*.txt` with `{{param}}` placeholders; missing or unknown params assert.
- `data.py`: `Document`/`Token`/`Mention`/`Relation`/`Entity` plus JSONL import/export. The importer lowercases mention and relation types, so gold `"Activity Data"` and predicted types compare as `"activity data"`.
- `entities.py` (`LLMEntitiesAnnotator`, coreference into entities) and `BaseAnnotator.batch_line` exist but are not used by the current pipeline.

Output: `res/pred/<model_name with "/" → "_">/end-to-end/pet.jsonl` (`res/pred/` is git-ignored). Each document is appended as soon as it finishes; re-running skips document ids already in that file, so delete the file to redo a model.

## Evaluation

`llm-metrics.py` iterates `res/pred/<model>/<target>/` where target is `end-to-end` (scores `md` and `re`), `md`, or `re`/`isolated-re`; any other folder name raises. Matching is lenient: a predicted mention matches a gold one if types are equal and token sets **overlap at all**; a relation matches if type is equal and both head and tail mentions overlap. Matching is greedy one-to-one per document. Scores are precision/recall/F1 per document, reported as mean ± std across documents (macro), using `eval.metrics.Stats`.

`src/data/pet.py` is an older parallel copy of the data model (`PetDocument`, etc.) used only by `src/eval/metrics.py`; the scripts use `autoannotate.data`.
