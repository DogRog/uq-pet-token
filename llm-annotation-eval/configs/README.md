# Run configs

Each `*.yaml` (or `*.yml`) file here is one experiment. The file name is the config's name and the name of its output folder, `res/pred/<name>/`.

```bash
uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml
uv run python src/run-configs.py configs/          # all configs in the folder (not subfolders)
uv run python src/run-configs.py configs/md/ --workers 8
```

`--workers` sets how many documents of a run are annotated at the same time (default 4). Each session in `run.json` records it, as parallel requests can slow down each call.

`configs/md/` holds the mention-detection experiments: 6 NHR models, each without reasoning and with `low` and `high` reasoning where the model supports it (see the table below), 5 runs each. `configs/md-test-sentences/` repeats them on the test sentences alone, one config for each of `configs/md/`.

| Key | Required | Meaning |
|---|---|---|
| `provider` | yes | `openai` or `nhr` (NHR@FAU gateway), see `src/autoannotate/providers.py` |
| `model` | yes | model ID, e.g. `gpt-6-sol` or `deepseek-ai/DeepSeek-V4.1-Flash` |
| `task` | no | `end-to-end` (default): mentions, then relations between them. `md`: mention detection only, no relation calls and no relation scores. `md-test-sentences`: mention detection on uq-pet-token's 84 test sentences, one sentence per request without its document (see the README) |
| `reasoning_effort` | no | `none`, `minimal`, `low`, `medium` or `high`. Left out (or `null`), the parameter is not sent and the model uses its default |
| `max_documents` | no | only annotate the first n gold documents, e.g. for a quick test. Left out: all |
| `repeats` | no | number of runs, default 1. Each run goes to `run-<i>/`, and `summary.csv` reports mean ± SD over runs |

The mention and relation definitions and few-shot examples are the same for all configs, in `src/experiments/definitions.py`.

A run can't be resumed with a different `task`, `provider`, `model`, `reasoning_effort` or prompt. Rename the config for a new experiment, or delete its folder to start over. Raising `max_documents` or `repeats` is fine and continues where the runs left off.

## Response cache on the NHR@FAU gateway

The gateway caches responses: an identical request returns the earlier answer within milliseconds, so repeated runs would be copies of the first. Every NHR request therefore sends `"cache": {"no-cache": true}` in its body (`Provider.extra_body` in `src/autoannotate/providers.py`), and `run.json` records it under `api`. A `Cache-Control: no-cache` header does not work.

## reasoning_effort on the NHR@FAU gateway

Tested on 2026-10-06 with one short prompt per setting, and for the `configs/md/` models again with the mention prompt on one gold document. "reasons" means reasoning text came back separately from the answer.

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
