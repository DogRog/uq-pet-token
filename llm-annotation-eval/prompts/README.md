# Prompts

Everything the LLMs are told is in this folder; change a file here to change a prompt.

| File | Used for |
|---|---|
| `mentions.txt` | the instruction for mention detection, for every dataset |
| `relations.txt` | the instruction for relation extraction (PET end-to-end runs) |
| `pet.yaml` | PET's mention and relation types with their descriptions, the procedures, disambiguation hints and format examples |
| `quishpi.yaml` | Quishpi's mention types, procedure, disambiguation hints and format example |
| `medical.yaml` | MedicalProcessInstruks' type definitions (after its authors' label definitions, as the release tags types by id only), how their most frequent training texts are added, and its procedure, disambiguation hints and format example |

## Structure

The instructions follow the modular prompt of Neuberger et al., "A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text Using Large Language Models" (ER 2024; their prompts are at https://github.com/JulianNeuberger/llm-process-generation/tree/er2024, `res/prompts/pet/`), keeping the parts their ablation study found useful:

- **Context**: the persona (a business process modelling expert) and what the input looks like. The paper found both of little effect (±0.01 F1); they are short.
- **Task Description**: the type definitions (`{{type_definitions}}`, one line per type, ` - **<type>**: <description>`, in the order of the YAML file) and a step-by-step procedure (`{{procedure}}`, chain of thought) that finds the types other types depend on first, e.g. Activities before their Actors and Activity Data.
- **Restrictions**: general rules, disambiguation hints (`{{disambiguation}}`: example sentences with the mentions or relations they contain, for types that are easily mistaken), the output format with a placeholder per field, and the format example (`{{example}}`: an input and the expected output). Removing the format example cost the paper 0.22 F1 in mention detection, mostly through unparseable answers.

## Examples

Every format example shows the same sentences: uq-pet-token's **bootstrap sentences**, the 5 labelled training sentences its token classifiers start from (`res/gold/<dataset>-bootstrap.jsonl`, written by `scripts/uq-pet-token-splits.py` with uq-pet-token's own split code), so the LLMs and the token classifiers see the same labelled examples. The `example` in a YAML file is only a template: `{{sentences}}` and `{{mentions}}` are filled with the bootstrap sentences and their gold mentions (`src/experiments/definitions.py`, `mention_example`), and PET's relation example's `{{text}}` and `{{relations}}` with the same sentences, their gold mentions tagged and the relations between them, from `res/gold/pet.jsonl` (`relation_example_document`; relations to other sentences are left out, so `actor recipient` and `same gateway` don't occur in it). The 5 sentences come from different documents but are shown as one input, numbered like a document's sentences; sentences without mentions have no output lines. They are never in the test split. PET whole-document runs annotate the documents they come from, so those documents' scores are not entirely unseen; test-split scores are.

Left out, as the paper found them useless or harmful once the prompt is settled: a list of facts about the process before the answer, and an explanation per extracted line. The procedure is how to work, not what to write: the answer stays only the tab-separated lines, as the parser reads them, listed in text order so the aligner places repeated texts on the right occurrences.

Each document follows in a separate message: numbered sentences (`<i>: <words>`) for mentions, the text with the predicted mentions tagged for relations. Format examples show their input the same way.

In the YAML files, `>-` joins a description's lines into one paragraph, so wrap them as you like; `|-` keeps the lines of procedures, hints and example templates as they are. `uv run pytest tests/test_prompts.py` checks that every example parses, that its input looks like the documents, that it shows the bootstrap sentences, and that the type names match the dataset's tags.

Type names must be spelled like the dataset's tags (see `src/eval/datasets.py`), as predictions are scored by them.

## Changing a prompt changes the experiment

Every run records its rendered prompts in `run.json`, and a run refuses to resume, or to get more repeats, once its prompts differ. After changing a prompt, give the configs new names (e.g. a different `name` in the `models` list) so their runs go to new folders, or delete the old run folders to start over. Results of runs with different prompts should not be pooled.
