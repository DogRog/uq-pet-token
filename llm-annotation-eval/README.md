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

### Annotate

Set these values near the bottom of `src/llm-annotate.py`:

- `provider_name`: `"openai"` or `"nhr"`
- `model_name`: e.g. `"gpt-6-sol"`, or an NHR model ID such as `"moonshotai/Kimi-K3"` (the available NHR chat models are listed in a comment there)
- `max_documents`: an int for a quick test on the first few documents, or `None` for all

Then run:

```bash
uv run python src/llm-annotate.py
```

Predictions are written to `res/pred/<model>/end-to-end/pet.jsonl`. Any `/` in the model ID becomes `_` in the folder name.

Each document is saved as soon as it is annotated. If a run is interrupted (Ctrl+C, a timeout), start it again and it skips the documents already in that file. To redo a model from scratch, delete the file first.

### Evaluate

Add the model's folder name to the `models` list in `src/llm-metrics.py`, then run:

```bash
uv run python src/llm-metrics.py
```

This prints precision, recall and F1 for mentions (`md`) and relations (`re`), as mean ± standard deviation over documents.

## Layout

- `src/autoannotate/`: annotators, response parsers, prompts and provider configs
- `src/eval/`: metric helpers
- `res/gold/pet.jsonl`: gold annotations
- `res/pred/`: model predictions
