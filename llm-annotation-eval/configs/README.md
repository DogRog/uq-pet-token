# Run configs

Each `*.yaml` (or `*.yml`) file here holds one experiment or several. Each experiment ("config") has a name, which is also the name of its output folder, `res/pred/<name>/`.

```bash
uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml
uv run python src/run-configs.py configs/          # all files in the folder (not subfolders)
uv run python src/run-configs.py configs/pet/md.yaml --workers 8
uv run python src/run-configs.py configs/quishpi/ --only 'quishpi-gemma-4-31b-*'
```

`--workers` sets how many documents of a run are annotated at the same time (default 4). Each session in `run.json` records it, as parallel requests can slow down each call. `--only` runs just the configs whose names match one of the given patterns (`*` for any text).

A file with `model` is one config, named after the file, like the end-to-end configs in this folder. A file with `models` lists models with their reasoning efforts, and every model and effort is a config; the other keys apply to all of them:

```yaml
task: md
provider: nhr
repeats: 5
models:
  - name: gemma-4-31b                       # short name, for the output folders
    model: RedHatAI/gemma-4-31B-it-FP8-block
    reasoning_effort: [none, low, high]     # a list, one value, or left out for the default
  - name: magistral-small-2509
    model: GaleneAI/Magistral-Small-2509-FP8-Dynamic
    reasoning_effort: none
```

Its configs are named `<name>-<task>`, followed by `-reasoning-<effort>` unless the effort is `none` (`-reasoning-default` if it is left out), and start with `<dataset>-` for datasets other than PET: `gemma-4-31b-md`, `gemma-4-31b-md-reasoning-low`, `quishpi-gemma-4-31b-md-test-sentences-reasoning-high`.

`configs/<dataset>/md.yaml` holds the mention-detection experiments on a dataset (`pet`, `quishpi`, `medical`): 6 NHR models, each without reasoning and with `low` and `high` reasoning where the model supports it (see the table below), 5 runs each. `configs/<dataset>/md-test-sentences.yaml` repeats them on the test sentences alone.

| Key | Required | Meaning |
|---|---|---|
| `dataset` | no | `pet` (default), `quishpi` or `medical`, see the README. Quishpi and medical only have the tasks `md` and `md-test-sentences` |
| `provider` | yes | `openai` or `nhr` (NHR@FAU gateway), see `src/autoannotate/providers.py` |
| `model` | yes, or `models` | model ID, e.g. `gpt-6-sol` or `deepseek-ai/DeepSeek-V4.1-Flash` |
| `models` | yes, or `model` | models to run, each with `name`, `model` and `reasoning_effort` (see above) |
| `task` | no | `end-to-end` (default): mentions, then relations between them. `md`: mention detection only, no relation calls and no relation scores. `md-test-sentences`: mention detection on uq-pet-token's test sentences of the dataset (84 for PET), one sentence per request without its document (see the README) |
| `reasoning_effort` | no | `none`, `minimal`, `low`, `medium` or `high`. Left out (or `null`), the parameter is not sent and the model uses its default. With `models`, given per model |
| `max_documents` | no | only annotate the first n gold documents (or test sentences), e.g. for a quick test. Left out: all |
| `repeats` | no | number of runs, default 1. Each run goes to `run-<i>/`, and `summary.csv` reports mean ± SD over runs |

The prompts are the same for all configs of a dataset, in [prompts/](../prompts/README.md).

A run can't be resumed with a different `dataset`, `task`, `provider`, `model`, `reasoning_effort` or prompt. Rename the config (or the model's `name`) for a new experiment, or delete its folder to start over. Raising `max_documents` or `repeats` is fine and continues where the runs left off.

## Response cache on the NHR@FAU gateway

The gateway caches responses: an identical request returns the earlier answer within milliseconds, so repeated runs would be copies of the first. Every NHR request therefore sends `"cache": {"no-cache": true}` in its body (`Provider.extra_body` in `src/autoannotate/providers.py`), and `run.json` records it under `api`. A `Cache-Control: no-cache` header does not work.

## reasoning_effort on the NHR@FAU gateway

Tested on 2026-10-06 with one short prompt per setting, and for the models in `configs/pet/md.yaml` again with the mention prompt on one gold document. "reasons" means reasoning text came back separately from the answer.

| Model | left out | `none` | `low` | `high` |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-V4.1-Flash | reasons | off | reasons | reasons (`medium` is rejected; it accepts `low`, `high`, `xhigh`, `max`) |
| Aleph-Alpha/Kolibri-1 | reasons | off | less | more |
| XiaomiMiMo/MiMo-V2.6-Flash-RL, moonshotai/Kimi-K3, zai-org/GLM-5.2-FP8 | reasons | off | reasons | reasons |
| Qwen/Qwen3.8-27B-FP8 | reasons (very long: 18k tokens on one document) | off | reasons | error 400 (`medium` works) |
| gpt-oss-120b | reasons | error 400 | less | more |
| RedHatAI/gemma-4-31B-it-FP8-block, google/gemma-4-E4B-it | off | off | reasons | reasons |
| mistralai/Mistral-Medium-3.5-128B | off | off | error 400 | reasons |
| zai-org/GLM-5.3, zai-org/GLM-5.3-Flash | reasons | reasoning ends up in the answer | off | off or very little |
| Qwen/Qwen3.6-35B-A3B-FP8 | reasons, setting ignored | | | |
| GaleneAI/Magistral-Small-2509-FP8-Dynamic | answers directly | same | error 400 | same (`medium` also error 400) |
| RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8, Microsoft/Phi-4-mini-instruct | no reasoning | error 400 | ignored | ignored |
| moonshotai/Kimi-K2.6, ibm-granite/granite-4.1-3b | no reasoning, setting ignored | | | |

deepseek-ai/DeepSeek-V4-Flash-0731 is listed by the gateway but did not respond. Only DeepSeek, Kolibri, MiMo and GLM-5.3 report reasoning tokens separately in the usage, so `reasoning_tokens` is empty for the others.
