# llm-annotation-eval

Uses LLMs to annotate business process descriptions from the [PET dataset](https://huggingface.co/datasets/patriziobellan/PET) with process element mentions and the relations between them, then scores the predictions against the gold annotations. Mention detection also runs on the two other datasets of [uq-pet-token](../README.md), the repository this folder is part of: [Quishpi](https://github.com/PADS-UPC/atdp-extractor/tree/bpm2020/input) and [MedicalProcessInstruks](https://huggingface.co/datasets/dagrat/MedicalProcessInstruks) (see [Datasets](#datasets)).

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
dataset: quishpi                          # optional: pet (default), quishpi or medical
task: md                                  # optional: md = mentions only, default end-to-end
reasoning_effort: low                     # optional, left out: not sent
max_documents: 5                          # optional, for a quick test
repeats: 3                                # optional, default 1
```

A file can also list several models, each with its reasoning efforts, instead of `model` and `reasoning_effort`; each model and effort is then its own experiment with its own output folder. The mention-detection experiments are written this way, one file per dataset and task, e.g. `configs/quishpi/md.yaml`.

[configs/README.md](configs/README.md) explains the keys and which `reasoning_effort` values each NHR model accepts. Experiments are run from a marimo notebook:

```bash
uv run marimo edit notebooks/run_experiments.py
```

In it you pick a config file and can change it in place (or save it), choose which of its configs to run, see the runs it makes and their state (to do, to resume, done, or can't be resumed as its settings or prompts changed), and preview the exact request of each step for any document. Running shows the progress and the scores so far after every document: mention (and relation) F1 on the documents done, and exact and partial F1 on the test sentences among them. Without the editor, e.g. on a server, the notebook runs a config as a script:

```bash
uv run python notebooks/run_experiments.py --config configs/pet/md.yaml --only 'gemma-4-31b-*' --workers 4
```

`--only` takes a pattern of config names (`*` for any text) and can be repeated; one config file per call. `--workers` sets how many documents are annotated at the same time (default 4). A document whose request times out or hits a server error is skipped and annotated when you start the config again; a rejected request, or 3 failed documents in a row, stops that config and the next one is run. Each run of a config gets its own folder, `res/pred/<config name>/run-<i>/`, containing:

- `end-to-end/pet.jsonl` (or `md/pet.jsonl` for `task: md`): the predicted documents
- `calls.jsonl`: one line per LLM call, with token usage (including reasoning tokens where the provider reports them), duration, finish reason, the raw response, the reasoning text, and the lines the parser had to drop or fix
- `run.json`: the settings, the exact prompts, and per session the start and end time, git commit, Python and package versions, and a hash of the gold file, plus totals over all calls
- `metrics.json`: precision, recall and F1 for mentions (`md`) and, for end-to-end runs, relations (`re`), as the mean ± SD over documents (macro), micro-averaged, and per type

After the configs are run, two tables are rebuilt from all run folders:

- `res/pred/runs.csv`: one row per run, with scores, tokens, time and parser problems
- `res/pred/summary.csv`: one row per config, the mean and sample SD over its complete runs

Each document is saved as soon as it is annotated. If a run is interrupted (Ctrl+C, a timeout), start it again and it continues with the documents that are left. It refuses to continue if the dataset, model, provider, reasoning effort or prompts changed since; rename the config for a new experiment, or delete its folder to start over.

### Results table

```bash
uv run marimo edit notebooks/eval_table.py
```

The notebook scores every `pet.jsonl` in `res/pred/` against the gold annotations, including predictions from before the configs (`res/pred/<folder>/<task>/pet.jsonl`), and shows one row per folder and task: precision, recall and F1 for mentions (`md`) and, where predicted, relations (`re`), and output tokens per sentence. Scores are macro-averaged over documents; a dropdown switches ± between the standard deviation over runs and over documents. Only runs with all their documents are included.

A dropdown chooses the dataset, another what to evaluate on: all documents as above (for PET its 45 gold documents), or the test split of [uq-pet-token](../README.md), for which these LLMs are a baseline (see below). Two more tables compare, on the test split, whole-document with single-sentence input for each model and reasoning effort, and, for whole-document runs, the scores on all 417 PET v1.1 sentences with those on the 84 test sentences (same tags and scoring), which shows how representative the test split is.

### Datasets

uq-pet-token runs its experiments on PET, Quishpi and MedicalProcessInstruks, each cut into sentences and split 80/20 the same way. A config's `dataset` key chooses which one the LLM annotates; Quishpi and MedicalProcessInstruks only have mention tasks (`md`, `md-test-sentences`).

| Dataset | Mention types | Sentences | Test sentences | Whole-document runs annotate |
|---|---|---|---|---|
| `pet` | Actor, Activity, Activity Data, Further Specification, XOR Gateway, Condition Specification, AND Gateway | 417 | 84 | `res/gold/pet.jsonl` (45 documents, with relations) |
| `quishpi` | Action, Entity, Condition | 201 | 41 | the sentences grouped into their 18 texts |
| `medical` | T03 … T43 (19 types without names) | 325 | 65 | the sentences grouped into their 13 guidelines |

`res/gold/<dataset>.jsonl` holds every sentence and `res/gold/<dataset>-test.jsonl` the test split, in uq-pet-token's format and order. They are written with uq-pet-token's own code, which also checks that PET's files are unchanged:

```bash
uv run --project .. python scripts/uq-pet-token-splits.py
```

MedicalProcessInstruks is gated, so its files are not tracked by git: accept its conditions on the Hub, log in with `uv run --project .. hf auth login`, and run the script before the first medical run.

Before reporting results next to uq-pet-token's, check that both are on the same data and scored the same way, with "Check the data" in `notebooks/run_experiments.py`, or:

```bash
uv run python notebooks/run_experiments.py --check
```

It compares the test gold stored in each of uq-pet-token's reported supervised test runs with `res/gold/`, rescores those runs' predictions with `eval.test_split.evaluate` against their `results.csv`, checks that every run here used today's gold files (their sha256 in `run.json`) and that no test sentence is in a prompt, and fails if one doesn't hold (`src/experiments/checks.py`). Optionally (`--check-rebuild`) it also rebuilds every split with uq-pet-token's code, the script above with `--check`, which writes nothing. It also lists the PET test sentences labelled differently in `pet.jsonl` and PET v1.1.

The prompts differ per dataset ([prompts/](prompts/README.md)). PET's and Quishpi's mention definitions are written by hand; Quishpi's follow its annotations, which mark mostly single verbs as actions, the objects they act on as entities (not who performs them), and condition clauses. MedicalProcessInstruks names none of its types, and uq-pet-token does not guess names, so each type is described by its 10 most frequent texts in the training sentences, and the format example is the 3 training sentences that together show the most types. No test sentence appears in a prompt.

### Test split of uq-pet-token

uq-pet-token trains token classifiers on PET v1.1 and evaluates them on 84 test sentences (41 for Quishpi, 65 for MedicalProcessInstruks, scored the same way). To compare the LLMs with them on the same data and metric:

- `res/gold/pet-v1.1-test.jsonl` holds those 84 sentences with their PET v1.1 BIO tags, in uq-pet-token's format and order. It was generated with uq-pet-token's `load_pet_splits` at commit 1ea66af (an 80/20 split of the 417 sentences with seed 3407); `src/eval/test_split.py` documents the details. PET v1.1 corrects some labels of `res/gold/pet.jsonl`, in 8 of the 84 test sentences.
- Runs of the usual tasks annotate whole documents; only their mentions in the test sentences are scored. Runs with `task: md-test-sentences` (configs in `configs/<dataset>/md-test-sentences.yaml`) instead send each test sentence on its own, without the rest of its document, the input uq-pet-token's models get; their predictions are single sentences with ids like `doc-2.1#27`, scored only on the test split. Mentions are converted to BIO tags (of overlapping mentions, the one listed first is kept). Scores use seqeval like uq-pet-token: a mention counts only if its type and both span boundaries match, micro-averaged over the test split. They are therefore lower than the overlap-based scores on all documents.
- Every run gets a `test-predictions.jsonl` next to its `pet.jsonl` (the predictions file has that name for every dataset): the test sentences of its dataset with predicted tags, for uq-pet-token to score with its own `evaluate_predictions`. Each run writes it when it is scored; to write it for every run again, e.g. for predictions made before, use "Export test predictions" in `notebooks/run_experiments.py`, or:

```bash
uv run python notebooks/run_experiments.py --export
```

### Tests

```bash
uv run pytest
```

## Layout

- `configs/`: experiment configs
- `notebooks/run_experiments.py`: marimo notebook that runs configs, previews their prompts, exports test predictions and checks the data against uq-pet-token
- `notebooks/eval_table.py`: marimo notebook with the results table
- `prompts/`: everything in the prompts: the instructions, and each dataset's type descriptions and format examples (see [prompts/README.md](prompts/README.md))
- `src/experiments/`: config loading, prompt loading (`definitions.py`), the run loop and its logs (`runner.py`), running configs (`batch.py`), reporting, the data checks (`checks.py`)
- `src/autoannotate/`: annotators, response parsers and provider configs
- `src/eval/`: matching and metrics; `datasets.py` lists the datasets with their files and mention types
- `scripts/uq-pet-token-splits.py`: writes the Quishpi and medical sentence files with uq-pet-token's code
- `res/gold/pet.jsonl`: gold annotations
- `res/gold/pet-v1.1.jsonl`: all 417 PET v1.1 sentences with BIO tags (uq-pet-token's `PETv1.1-entities.jsonl`)
- `res/gold/pet-v1.1-test.jsonl`: uq-pet-token's test split, with PET v1.1 tags
- `res/gold/quishpi.jsonl`, `res/gold/quishpi-test.jsonl`: Quishpi's sentences and test split
- `res/gold/medical.jsonl`, `res/gold/medical-test.jsonl`: MedicalProcessInstruks' sentences and test split (generated, not tracked by git)
- `res/pred/`: model predictions and run logs (not tracked by git)
