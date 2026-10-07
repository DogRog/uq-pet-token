# llm-annotation-eval

Uses LLMs to annotate business process descriptions from the [PET dataset](https://huggingface.co/datasets/patriziobellan/PET) with process element mentions and the relations between them, then scores the predictions against the gold annotations.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.template .env
```

In `.env`, fill in the key for the provider you use:

- `OPENAI_API_KEY` for OpenAI models
- `NHR_API_KEY` for the [NHR@FAU LLM gateway](https://hpc.fau.de/request-llm-api-key/), which serves open-weight models through an OpenAI-compatible API

## Usage

### Run experiments

Each experiment is a config file in `configs/`, e.g.:

```yaml
provider: nhr                             # or openai
model: deepseek-ai/DeepSeek-V4.1-Flash
task: md                                  # optional: md = mentions only, default end-to-end
reasoning_effort: low                     # optional, left out: not sent
max_documents: 5                          # optional, for a quick test
repeats: 3                                # optional, default 1
```

[configs/README.md](configs/README.md) explains the keys and which `reasoning_effort` values each NHR model accepts. To annotate the gold documents and score the predictions, run:

```bash
uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml
```

Pass several files, or a folder for all configs in it. `--workers N` sets how many documents are annotated at the same time (default 4). A document whose request times out or hits a server error is skipped and annotated when you start the config again; a rejected request, or 3 failed documents in a row, stops that config and the script moves on to the next. Each run of a config gets its own folder, `res/pred/<config name>/run-<i>/`, containing:

- `end-to-end/pet.jsonl` (or `md/pet.jsonl` for `task: md`): the predicted documents
- `calls.jsonl`: one line per LLM call, with token usage (including reasoning tokens where the provider reports them), duration, finish reason, the raw response, the reasoning text, and the lines the parser had to drop or fix
- `run.json`: the settings, the exact prompts, and per session the start and end time, git commit, Python and package versions, and a hash of the gold file, plus totals over all calls
- `metrics.json`: precision, recall and F1 for mentions (`md`) and, for end-to-end runs, relations (`re`), as the mean ± SD over documents (macro), micro-averaged, and per type

After every invocation, two tables are rebuilt from all run folders:

- `res/pred/runs.csv`: one row per run, with scores, tokens, time and parser problems
- `res/pred/summary.csv`: one row per config, the mean and sample SD over its complete runs

Each document is saved as soon as it is annotated. If a run is interrupted (Ctrl+C, a timeout), start it again and it continues with the documents that are left. It refuses to continue if the model, provider, reasoning effort or prompts changed since; rename the config for a new experiment, or delete its folder to start over.

### Results table

```bash
uv run marimo edit notebooks/eval_table.py
```

The notebook scores every `pet.jsonl` in `res/pred/` against the gold annotations, including predictions from before the configs (`res/pred/<folder>/<task>/pet.jsonl`), and shows one row per folder and task: precision, recall and F1 for mentions (`md`) and, where predicted, relations (`re`), and output tokens per document. Scores are macro-averaged over documents; a dropdown switches ± between the standard deviation over runs and over documents. Only runs with all their documents are included.

A dropdown chooses what to evaluate on: all 45 documents as above, or the test split of [uq-pet-token](../uq-pet-token), for which these LLMs are a baseline (see below).

### Test split of uq-pet-token

uq-pet-token trains token classifiers on PET v1.1 and evaluates them on 84 test sentences. To compare the LLMs with them on the same data and metric:

- `res/gold/pet-v1.1-test.jsonl` holds those 84 sentences with their PET v1.1 BIO tags, in uq-pet-token's format and order. It was generated with uq-pet-token's `load_pet_splits` at commit 1ea66af (an 80/20 split of the 417 sentences with seed 3407); `src/eval/test_split.py` documents the details. PET v1.1 corrects some labels of `res/gold/pet.jsonl`, in 8 of the 84 test sentences.
- Runs of the usual tasks annotate whole documents; only their mentions in the test sentences are scored. Runs with `task: md-test-sentences` (configs in `configs/md-test-sentences/`) instead send each test sentence on its own, without the rest of its document, the input uq-pet-token's models get; their predictions are single sentences with ids like `doc-2.1#27`, scored only on the test split. Mentions are converted to BIO tags (of overlapping mentions, the one listed first is kept). Scores use seqeval like uq-pet-token: a mention counts only if its type and both span boundaries match, micro-averaged over the test split. They are therefore lower than the overlap-based scores on all documents.
- Every run gets a `test-predictions.jsonl` next to its `pet.jsonl`: the 84 test sentences with predicted tags, for uq-pet-token to score with its own `evaluate_predictions`. `run-configs.py` writes it after each run; for predictions made before, run:

```bash
uv run python src/export-test-predictions.py
```

### Tests

```bash
uv run pytest
```

## Layout

- `configs/`: experiment configs
- `notebooks/eval_table.py`: marimo notebook with the results table
- `src/run-configs.py`: runs configs
- `src/experiments/`: config loading, the run loop and its logs, reporting; `definitions.py` holds the mention and relation definitions and few-shot examples used in the prompts
- `src/autoannotate/`: annotators, response parsers, prompt templates and provider configs
- `src/eval/`: matching and metrics
- `res/gold/pet.jsonl`: gold annotations
- `res/gold/pet-v1.1-test.jsonl`: uq-pet-token's test split, with PET v1.1 tags
- `res/pred/`: model predictions and run logs (not tracked by git)
