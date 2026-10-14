# Prompts

Everything the LLMs are told is in this folder; change a file here to change a prompt.

| File | Used for |
|---|---|
| `mentions.txt` | the instruction for mention detection, for every dataset |
| `relations.txt` | the instruction for relation extraction (PET end-to-end runs) |
| `pet.yaml` | PET's mention and relation types with their descriptions, and the format examples |
| `quishpi.yaml` | Quishpi's mention types and format example |
| `medical.yaml` | how MedicalProcessInstruks' type descriptions and example are generated from its training sentences, as it names none of its types |

The instructions contain `{{type_definitions}}` and `{{example}}`. The first becomes one line per type, ` - **<type>**: <description>`, in the order of the YAML file; the second is the dataset's `example`. Each document follows in a separate message: numbered sentences for mentions, the text with the predicted mentions tagged for relations.

In the YAML files, `>-` joins a description's lines into one paragraph, so wrap them as you like; `|-` keeps the example's lines as they are. In examples, the fields of an output line (text, type and sentence for mentions; type, source and target for relations) are separated by a **tab**, the format the answers are parsed in. Make sure your editor does not turn them into spaces; `uv run pytest tests/test_prompts.py` checks that every example parses and that the type names match the dataset's tags.

Type names must be spelled like the dataset's tags (see `src/eval/datasets.py`), as predictions are scored by them.

## Changing a prompt changes the experiment

Every run records its rendered prompts in `run.json`, and a run refuses to resume, or to get more repeats, once its prompts differ. After changing a prompt, give the configs new names (e.g. a different `name` in the `models` list) so their runs go to new folders, or delete the old run folders to start over. Results of runs with different prompts should not be pooled.
