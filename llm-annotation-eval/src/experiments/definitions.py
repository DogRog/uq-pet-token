"""Reads the prompts of each dataset from prompts/ (see prompts/README.md).

prompts/mentions.txt is the instruction, with {{type_definitions}},
{{procedure}}, {{disambiguation}} and {{example}} filled in from prompts/<dataset>.yaml, where
every dataset's types are described by hand. Every example shows uq-pet-token's bootstrap
sentences (the labelled sentences its token classifiers start from, see
eval.datasets.Dataset.bootstrap) with their gold annotations, filled into the example template of
the YAML file; the test split never appears in a prompt.

A run's manifest (run.json) records the rendered prompts, and resuming a run refuses to continue
if they changed in between.
"""
import pathlib
import typing

import yaml

import autoannotate
from eval import datasets

PROMPT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "prompts"

# the parts of a task's section in prompts/<dataset>.yaml that fill the template's placeholders
# of the same name, besides the type definitions
TEMPLATE_VALUES = ("procedure", "disambiguation", "example")

test_split = datasets.test_split
Row = test_split.Row


def mention_template() -> pathlib.Path:
    return PROMPT_DIR / "mentions.txt"


def _read(dataset: str) -> typing.Dict[str, typing.Any]:
    path = PROMPT_DIR / f"{dataset}.yaml"
    with open(path, encoding="utf8") as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict) or "mentions" not in values:
        raise ValueError(f"{path}: expected a 'mentions' section, see prompts/README.md.")
    return values


def _types(section: typing.Dict[str, typing.Any], where: str) -> typing.Dict[str, str]:
    types = section.get("types")
    if not isinstance(types, dict) or not all(isinstance(d, str) for d in types.values()):
        raise ValueError(f"{where}: expected 'types' with one description per type.")
    return types


def _example_template(section: typing.Dict[str, typing.Any], where: str) -> str:
    if not isinstance(section.get("example"), str):
        raise ValueError(f"{where}: expected an 'example'.")
    return section["example"]


def _template_values(section: typing.Dict[str, typing.Any], where: str) -> typing.Dict[str, str]:
    for name in TEMPLATE_VALUES:
        if not isinstance(section.get(name), str):
            raise ValueError(f"{where}: expected a '{name}'.")
    return {name: section[name] for name in TEMPLATE_VALUES}


def mention_prompt(
        dataset: datasets.Dataset
) -> typing.Tuple[typing.List[autoannotate.MentionDefinition], typing.Dict[str, str]]:
    """The mention definitions of a dataset and the other values of the mention template."""
    mentions = _read(dataset.name)["mentions"]
    where = f"prompts/{dataset.name}.yaml, mentions"
    types = _types(mentions, where)
    definitions = [autoannotate.MentionDefinition(type=t, description=d) for t, d in types.items()]
    example = mention_example(dataset.bootstrap(), _example_template(mentions, where))
    return definitions, _template_values({**mentions, "example": example}, where)


def _mentions(row: Row) -> typing.List[typing.Tuple[str, str]]:
    """The (text, type) of the tagged mentions of a sentence."""
    return [
        (" ".join(row["tokens"][start:end]), tag_type)
        for start, end, tag_type in datasets.mention_spans(row["ner-tags"])
    ]


def mention_example(sentences: typing.List[Row], template: str) -> str:
    """The format example of the mention prompt: sentences and their tagged mentions.

    Args:
        sentences: the example's sentences, see eval.datasets.Dataset.bootstrap
        template: the example in prompts/<dataset>.yaml, with {{sentences}} for the "<i>: <words>"
            lines, as documents are given, and {{mentions}} for one tab-separated line per mention
    """
    return autoannotate.Prompt(text=template)(
        sentences="\n".join(f"{i}: {' '.join(r['tokens'])}" for i, r in enumerate(sentences)),
        mentions="\n".join(
            f"{text}\t{tag_type}\t{i}"
            for i, r in enumerate(sentences)
            for text, tag_type in _mentions(r)
        ),
    )

