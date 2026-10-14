import dataclasses
import pathlib
import typing

import yaml

import autoannotate
from eval import datasets

REASONING_EFFORTS = typing.get_args(typing.get_args(autoannotate.ReasoningEffort)[0])
# "end-to-end": mentions, then relations between them; "md": mention detection only;
# "md-test-sentences": mention detection on uq-pet-token's test sentences, each sent on its own
# (see eval.test_split). Also the name of the predictions folder in a run, which the results
# notebook scores by.
TASKS = ("end-to-end", "md", "md-test-sentences")
# relations are only defined for PET
MENTION_TASKS = ("md", "md-test-sentences")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One experiment, read from a YAML file in configs/; see configs/README.md for the keys."""
    # file name without .yaml, also the name of the output folder in res/pred/
    name: str
    provider: str
    model: str
    # see eval.datasets.DATASETS
    dataset: str = "pet"
    task: str = "end-to-end"
    # None: don't send the parameter, many models on the NHR@FAU gateway reject some values
    reasoning_effort: autoannotate.ReasoningEffort = None
    # None: all gold documents
    max_documents: typing.Optional[int] = None
    repeats: int = 1

    def settings(self) -> typing.Dict[str, typing.Any]:
        """Settings that change the predictions, a run must not be resumed with different ones."""
        return {
            "dataset": self.dataset, "task": self.task, "provider": self.provider, "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


# the keys of each entry under "models" in a config file with several models
MODEL_KEYS = {"name", "model", "reasoning_effort"}


def run_name(dataset: str, name: str, task: str, reasoning_effort: autoannotate.ReasoningEffort) -> str:
    """The name of one model and reasoning effort of a config file with several models.

    E.g. "gemma-4-31b-md" (reasoning "none"), "gemma-4-31b-md-reasoning-low" or
    "quishpi-gemma-4-31b-md-test-sentences-reasoning-high"; "default" if the effort is left out.
    """
    prefix = "" if dataset == "pet" else f"{dataset}-"
    suffix = "" if reasoning_effort == "none" else f"-reasoning-{reasoning_effort or 'default'}"
    return f"{prefix}{name}-{task}{suffix}"


def load_configs(path: pathlib.Path) -> typing.List[RunConfig]:
    """Reads a config file: one run config, or one per model and reasoning effort under "models".

    A file with "model" is one config named after the file. A file with "models" lists models,
    each with a short "name", its "model" ID and one reasoning effort or a list of them; the
    other keys apply to all of them, and each model and effort is a config named by run_name.
    """
    return parse_configs(path.read_text(encoding="utf8"), path.stem, str(path))


def parse_configs(text: str, name: str, where: str = "config") -> typing.List[RunConfig]:
    """Reads the YAML text of a config file, see load_configs.

    Args:
        text: the file's content
        name: the name of a config with "model", for a file its name without .yaml
        where: what error messages call the text, e.g. the file
    """
    values = yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError(f"{where}: expected keys like 'provider: nhr', one per line.")
    if "models" not in values:
        return [_validated(where, name, values)]

    if "model" in values or "reasoning_effort" in values:
        raise ValueError(f"{where}: give 'model' and 'reasoning_effort' per entry under 'models'.")
    models = values.pop("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{where}: 'models' must be a list of entries with name, model and reasoning_effort.")
    configs = []
    for i, entry in enumerate(models):
        entry_where = f"{where}, models entry {i + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_where}: expected keys name, model and reasoning_effort.")
        unknown = entry.keys() - MODEL_KEYS
        if unknown:
            raise ValueError(f"{entry_where}: unknown keys {sorted(unknown)}, expected some of {sorted(MODEL_KEYS)}.")
        if not isinstance(entry.get("name"), str):
            raise ValueError(f"{entry_where}: 'name' is required and must be text, it names the output folders.")
        efforts = entry.get("reasoning_effort")
        if efforts == []:
            raise ValueError(f"{entry_where}: 'reasoning_effort' is an empty list, leave it out for the default.")
        for effort in efforts if isinstance(efforts, list) else [efforts]:
            run_values = {**values, "reasoning_effort": effort}
            if "model" in entry:
                run_values["model"] = entry["model"]
            name = run_name(values.get("dataset", "pet"), entry["name"], values.get("task", "end-to-end"), effort)
            configs.append(_validated(entry_where, name, run_values))
    return configs


def _validated(where: str, name: str, values: typing.Dict[str, typing.Any]) -> RunConfig:
    fields = {f.name for f in dataclasses.fields(RunConfig)} - {"name"}
    unknown = values.keys() - fields
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, expected some of {sorted(fields)}.")
    for key in ("provider", "model"):
        if key not in values:
            raise ValueError(f"{where}: '{key}' is required.")
        # YAML reads unquoted values like 1.5 as numbers
        if not isinstance(values[key], str):
            raise ValueError(f"{where}: '{key}' must be text, put it in quotes.")

    config = RunConfig(name=name, **values)
    if config.provider not in autoannotate.PROVIDERS:
        raise ValueError(f"{where}: unknown provider '{config.provider}', expected one of {list(autoannotate.PROVIDERS)}.")
    if config.dataset not in datasets.DATASETS:
        raise ValueError(f"{where}: dataset must be one of {list(datasets.DATASETS)}.")
    if config.task not in TASKS:
        raise ValueError(f"{where}: task must be one of {list(TASKS)}.")
    if config.dataset != "pet" and config.task not in MENTION_TASKS:
        raise ValueError(
            f"{where}: {config.dataset} has no relations, task must be one of {list(MENTION_TASKS)}."
        )
    if config.reasoning_effort is not None and config.reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"{where}: reasoning_effort must be one of {list(REASONING_EFFORTS)}, or left out.")
    if config.max_documents is not None and not (isinstance(config.max_documents, int) and config.max_documents > 0):
        raise ValueError(f"{where}: max_documents must be a positive integer, or left out.")
    if not (isinstance(config.repeats, int) and config.repeats > 0):
        raise ValueError(f"{where}: repeats must be a positive integer.")
    return config
