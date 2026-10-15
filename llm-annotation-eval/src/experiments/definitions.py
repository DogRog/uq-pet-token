"""Reads the prompts of each dataset from prompts/ (see prompts/README.md).

prompts/mentions.txt and prompts/relations.txt are the instructions, with {{type_definitions}},
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
from autoannotate import data, util
from eval import datasets

PROMPT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "prompts"

# the parts of a task's section in prompts/<dataset>.yaml that fill the template's placeholders
# of the same name, besides the type definitions
TEMPLATE_VALUES = ("procedure", "disambiguation", "example")

test_split = datasets.test_split
Row = test_split.Row


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
    template = _example_template(relations, where)
    example = relation_example(relation_example_document(dataset), template)
    return definitions, _template_values({**relations, "example": example}, where)


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


def relation_example_document(dataset: datasets.Dataset) -> data.Document:
    """The bootstrap sentences as one document, with their gold mentions and relations.

    The annotations are those of the gold documents (res/gold/pet.jsonl), which have relations:
    the mentions inside the sentences, in text order, and the relations between them; relations
    to other sentences are left out.
    """
    documents = {d.id: d for d in datasets.read_documents(dataset.documents_file)}
    tokens: typing.List[data.Token] = []
    mentions: typing.List[data.Mention] = []
    relations: typing.List[data.Relation] = []
    for sentence_index, row in enumerate(dataset.bootstrap()):
        document = documents[row["document name"]]
        sentence = document.sentences[row["sentence-ID"]]
        if [t.text for t in sentence] != row["tokens"]:
            raise ValueError(f"{test_split.sentence_id(row)} differs in {dataset.documents_file}")
        # document token index -> example token index
        moved = {t.index_in_document: len(tokens) + i for i, t in enumerate(sentence)}
        inside = sorted(
            (i for i, m in enumerate(document.mentions)
             if all(j in moved for j in m.token_document_indices)),
            key=lambda i: document.mentions[i].token_document_indices,
        )
        new_index = {i: len(mentions) + n for n, i in enumerate(inside)}
        mentions += [
            data.Mention(
                type=document.mentions[i].type,
                token_document_indices=tuple(
                    moved[j] for j in document.mentions[i].token_document_indices
                ),
            )
            for i in inside
        ]
        relations += sorted(
            (
                data.Relation(
                    type=r.type,
                    head_mention_index=new_index[r.head_mention_index],
                    tail_mention_index=new_index[r.tail_mention_index],
                )
                for r in document.relations
                if r.head_mention_index in new_index and r.tail_mention_index in new_index
            ),
            key=lambda r: (r.head_mention_index, r.tail_mention_index),
        )
        tokens += [
            data.Token(
                text=t.text,
                index_in_document=moved[t.index_in_document],
                pos_tag=t.pos_tag,
                sentence_index=sentence_index,
            )
            for t in sentence
        ]
    return data.Document(
        id="bootstrap",
        name="bootstrap",
        text=" ".join(t.text for t in tokens),
        category="",
        tokens=tokens,
        mentions=mentions,
        relations=relations,
        entities=[],
    )


def relation_example(document: data.Document, template: str) -> str:
    """The format example of the relation prompt: a document's tagged text and its relations.

    Args:
        document: see relation_example_document
        template: the example in prompts/<dataset>.yaml, with {{text}} for the text with the
            mentions tagged, as documents are given, and {{relations}} for one tab-separated line
            per relation
    """
    return autoannotate.Prompt(text=template)(
        text=util.format_document_text_with_entity_mentions(document),
        relations="\n".join(
            f"{r.type}\t{r.head_mention_index}\t{r.tail_mention_index}" for r in document.relations
        ),
    )
