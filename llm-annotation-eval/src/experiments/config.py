import dataclasses
import pathlib
import typing

import yaml

import autoannotate

REASONING_EFFORTS = typing.get_args(typing.get_args(autoannotate.ReasoningEffort)[0])
# "end-to-end": mentions, then relations between them; "md": mention detection only.
# Also the name of the predictions folder in a run, as llm-metrics.py expects.
TASKS = ("end-to-end", "md")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One experiment, read from a YAML file in configs/; see configs/README.md for the keys."""
    # file name without .yaml, also the name of the output folder in res/pred/
    name: str
    provider: str
    model: str
    task: str = "end-to-end"
    # None: don't send the parameter, many models on the NHR@FAU gateway reject some values
    reasoning_effort: autoannotate.ReasoningEffort = None
    # None: all gold documents
    max_documents: typing.Optional[int] = None
    repeats: int = 1

    def settings(self) -> typing.Dict[str, typing.Any]:
        """Settings that change the predictions, a run must not be resumed with different ones."""
        return {
            "task": self.task, "provider": self.provider, "model": self.model, "reasoning_effort": self.reasoning_effort
        }


def load_config(path: pathlib.Path) -> RunConfig:
    with open(path, encoding="utf8") as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected keys like 'provider: nhr', one per line.")

    fields = {f.name for f in dataclasses.fields(RunConfig)} - {"name"}
    unknown = values.keys() - fields
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, expected some of {sorted(fields)}.")
    for key in ("provider", "model"):
        if key not in values:
            raise ValueError(f"{path}: '{key}' is required.")
        # YAML reads unquoted values like 1.5 as numbers
        if not isinstance(values[key], str):
            raise ValueError(f"{path}: '{key}' must be text, put it in quotes.")

    config = RunConfig(name=path.stem, **values)
    if config.provider not in autoannotate.PROVIDERS:
        raise ValueError(f"{path}: unknown provider '{config.provider}', expected one of {list(autoannotate.PROVIDERS)}.")
    if config.task not in TASKS:
        raise ValueError(f"{path}: task must be one of {list(TASKS)}.")
    if config.reasoning_effort is not None and config.reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"{path}: reasoning_effort must be one of {list(REASONING_EFFORTS)}, or left out.")
    if config.max_documents is not None and not (isinstance(config.max_documents, int) and config.max_documents > 0):
        raise ValueError(f"{path}: max_documents must be a positive integer, or left out.")
    if not (isinstance(config.repeats, int) and config.repeats > 0):
        raise ValueError(f"{path}: repeats must be a positive integer.")
    return config
