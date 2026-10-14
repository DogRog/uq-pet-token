"""Reads the prompts of each dataset from prompts/ (see prompts/README.md).

prompts/mentions.txt and prompts/relations.txt are the instructions, with {{type_definitions}},
{{procedure}}, {{disambiguation}} and {{example}} filled in from prompts/<dataset>.yaml. PET's and
Quishpi's types are described there by hand. MedicalProcessInstruks names none of its types, so
their descriptions and the example are generated from its training sentences, with the wording
and settings in prompts/medical.yaml; the test split never appears in a prompt.

A run's manifest (run.json) records the rendered prompts, and resuming a run refuses to continue
if they changed in between.
"""
import collections
import pathlib
import typing

import yaml

import autoannotate
from eval import datasets

PROMPT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "prompts"

# the parts of a task's section in prompts/<dataset>.yaml that fill the template's placeholders
# of the same name, besides the type definitions
TEMPLATE_VALUES = ("procedure", "disambiguation", "example")

Row = datasets.test_split.Row


def mention_template() -> pathlib.Path:
    return PROMPT_DIR / "mentions.txt"


def relation_template() -> pathlib.Path:
    return PROMPT_DIR / "relations.txt"


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
    if dataset.name == "medical":
        train = dataset.train()
        definitions = medical_mention_definitions(train, mentions)
        example = medical_mention_format_example(train, mentions)
        return definitions, _template_values({**mentions, "example": example}, where)
    types = _types(mentions, where)
    definitions = [autoannotate.MentionDefinition(type=t, description=d) for t, d in types.items()]
    return definitions, _template_values(mentions, where)


def relation_prompt(
        dataset: datasets.Dataset
) -> typing.Tuple[typing.List[autoannotate.RelationDefinition], typing.Dict[str, str]]:
    """The relation definitions of a dataset and the other values of the relation template.

    Only PET has relations.
    """
    relations = _read(dataset.name).get("relations")
    if relations is None:
        raise ValueError(f"prompts/{dataset.name}.yaml defines no relations.")
    where = f"prompts/{dataset.name}.yaml, relations"
    types = _types(relations, where)
    definitions = [autoannotate.RelationDefinition(type=t, description=d) for t, d in types.items()]
    return definitions, _template_values(relations, where)


def _mentions(row: Row) -> typing.List[typing.Tuple[str, str]]:
    """The (text, type) of the tagged mentions of a sentence."""
    return [
        (" ".join(row["tokens"][start:end]), tag_type)
        for start, end, tag_type in datasets.mention_spans(row["ner-tags"])
    ]


def medical_mention_definitions(
        train: typing.List[Row], settings: typing.Dict[str, typing.Any]
) -> typing.List[autoannotate.MentionDefinition]:
    """Describes each unnamed medical type by its most frequent texts in the training sentences.

    Args:
        train: the training sentences, see eval.datasets.Dataset.train
        settings: the mentions section of prompts/medical.yaml
    """
    described = autoannotate.Prompt(text=settings["description"])
    unseen = autoannotate.Prompt(text=settings["description_unseen"])
    texts: typing.Dict[str, typing.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in train:
        for text, tag_type in _mentions(row):
            texts[tag_type][text] += 1
    definitions = []
    for tag_type in datasets.DATASETS["medical"].types:
        counts = texts[tag_type]
        if counts:
            total = sum(counts.values())
            most_common = counts.most_common(settings["texts_per_type"])
            description = described(
                mention_count=f"{total} mention{'s' if total > 1 else ''}",
                texts=", ".join(f'"{text}"' for text, _ in most_common),
            )
        else:
            description = unseen()
        definitions.append(autoannotate.MentionDefinition(type=tag_type, description=description))
    return definitions


def medical_mention_format_example(
        train: typing.List[Row], settings: typing.Dict[str, typing.Any]
) -> str:
    """Training sentences that together show the most types, as the format example.

    Args:
        train: the training sentences, see eval.datasets.Dataset.train
        settings: the mentions section of prompts/medical.yaml
    """
    candidates = [r for r in train if len(r["tokens"]) <= settings["example_max_words"]]
    chosen: typing.List[Row] = []
    shown: typing.Set[str] = set()
    for _ in range(settings["example_sentences"]):
        # the first sentence with the most types not shown yet
        best = max(candidates, key=lambda r: len({t for _, t in _mentions(r)} - shown))
        chosen.append(best)
        candidates.remove(best)
        shown |= {t for _, t in _mentions(best)}
    return autoannotate.Prompt(text=settings["example"])(
        sentences="\n".join(f"{i}: {' '.join(r['tokens'])}" for i, r in enumerate(chosen)),
        mentions="\n".join(
            f"{text}\t{tag_type}\t{i}"
            for i, r in enumerate(chosen)
            for text, tag_type in _mentions(r)
        ),
    )
