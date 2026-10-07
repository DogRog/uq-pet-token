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
- `res/pred/`: model predictions and run logs (not tracked by git)
