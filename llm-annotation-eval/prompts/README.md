# Prompts

Everything the LLMs are told is in this folder; change a file here to change a prompt.

| File | Used for |
|---|---|
| `mentions.txt` | the instruction for mention detection, for every dataset |
| `relations.txt` | the instruction for relation extraction (PET end-to-end runs) |
| `pet.yaml` | PET's mention and relation types with their descriptions, the procedures, disambiguation hints and format examples |
| `quishpi.yaml` | Quishpi's mention types, procedure, disambiguation hints and format example |
| `medical.yaml` | how MedicalProcessInstruks' type descriptions and example are generated from its training sentences, as it names none of its types, and its procedure and disambiguation hint |

## Structure

The instructions follow the modular prompt of Neuberger et al., "A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text Using Large Language Models" (ER 2024; their prompts are at https://github.com/JulianNeuberger/llm-process-generation/tree/er2024, `res/prompts/pet/`), keeping the parts their ablation study found useful:

- **Context**: the persona (a business process modelling expert) and what the input looks like. The paper found both of little effect (±0.01 F1); they are short.
- **Task Description**: the type definitions (`{{type_definitions}}`, one line per type, ` - **<type>**: <description>`, in the order of the YAML file) and a step-by-step procedure (`{{procedure}}`, chain of thought) that finds the types other types depend on first, e.g. Activities before their Actors and Activity Data.
- **Restrictions**: general rules, disambiguation hints (`{{disambiguation}}`: example sentences with the mentions or relations they contain, for types that are easily mistaken), the output format with a placeholder per field, and the format example (`{{example}}`: an input and the expected output). Removing the format example cost the paper 0.22 F1 in mention detection, mostly through unparseable answers.

Left out, as the paper found them useless or harmful once the prompt is settled: a list of facts about the process before the answer, and an explanation per extracted line. The procedure is how to work, not what to write: the answer stays only the tab-separated lines, as the parser reads them, listed in text order so the aligner places repeated texts on the right occurrences.

Each document follows in a separate message: numbered sentences (`<i>: <words>`) for mentions, the text with the predicted mentions tagged for relations. Format examples show their input the same way.

In the YAML files, `>-` joins a description's lines into one paragraph, so wrap them as you like; `|-` keeps the lines of procedures, hints and examples as they are. In examples, the fields of an output line (text, type and sentence for mentions; type, source and target for relations) are separated by a **tab**, the format the answers are parsed in. Make sure your editor does not turn them into spaces; `uv run pytest tests/test_prompts.py` checks that every example parses, that its input looks like the documents, and that the type names match the dataset's tags.

Type names must be spelled like the dataset's tags (see `src/eval/datasets.py`), as predictions are scored by them.

## Changing a prompt changes the experiment

Every run records its rendered prompts in `run.json`, and a run refuses to resume, or to get more repeats, once its prompts differ. After changing a prompt, give the configs new names (e.g. a different `name` in the `models` list) so their runs go to new folders, or delete the old run folders to start over. Results of runs with different prompts should not be pooled.
