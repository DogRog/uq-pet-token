"""The datasets runs annotate, all evaluated like uq-pet-token evaluates its token classifiers.

uq-pet-token splits PET v1.1, Quishpi and MedicalProcessInstruks into sentences the same way (see
eval.test_split). For each, ``res/gold/`` holds every sentence, the test split and the bootstrap
sentences, one sentence per line with BIO tags; ``scripts/uq-pet-token-splits.py`` writes them
with uq-pet-token's own code. Whole-document runs annotate a dataset's sentences grouped back
into documents, with the tagged mentions as gold.

MedicalProcessInstruks is gated, so its files are not tracked by git; they must be generated
before a medical run.
"""

import dataclasses
import json
import pathlib
import typing

from autoannotate import data
from eval import test_split

GOLD_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "res" / "gold"


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    # the mention types as the BIO tags spell them; the parser lowercases predicted types
    types: typing.Tuple[str, ...]
    # every sentence, and uq-pet-token's test split
    sentences_file: pathlib.Path
    test_file: pathlib.Path
    # uq-pet-token's bootstrap sentences: the labelled training sentences its token classifiers
    # start from, and the prompts' examples
    bootstrap_file: pathlib.Path

    @property
    def gold_file(self) -> pathlib.Path:
        """What whole-document runs annotate: the sentences, grouped by read_documents."""
        return self.sentences_file

    def input_file(self, task: str) -> pathlib.Path:
        """md-test-sentences runs annotate the test sentences, the others whole documents."""
        return self.test_file if task == "md-test-sentences" else self.gold_file

    def train(self) -> typing.List[test_split.Row]:
        """The sentences outside the test split, the only ones prompts may take examples from."""
        test_keys = {
            (r["document name"], r["sentence-ID"])
            for r in test_split.load_test_split(self.test_file)
        }
        return [
            r
            for r in test_split.load_test_split(self.sentences_file)
            if (r["document name"], r["sentence-ID"]) not in test_keys
        ]

    def bootstrap(self) -> typing.List[test_split.Row]:
        """uq-pet-token's bootstrap sentences, in its order."""
        return test_split.load_test_split(self.bootstrap_file)


# the 19 types of MedicalProcessInstruks that occur, named by their begin tag id like uq-pet-token
MEDICAL_TYPE_IDS = (3, 5, 9, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43)

DATASETS = {
    d.name: d
    for d in (
        Dataset(
            name="pet",
            types=(
                "Actor",
                "Activity",
                "Activity Data",
                "Further Specification",
                "XOR Gateway",
                "Condition Specification",
                "AND Gateway",
            ),
            sentences_file=GOLD_DIR / "pet-v1.1.jsonl",
            test_file=GOLD_DIR / "pet-v1.1-test.jsonl",
            bootstrap_file=GOLD_DIR / "pet-v1.1-bootstrap.jsonl",
        ),
        Dataset(
            name="quishpi",
            types=("Action", "Entity", "Condition"),
            sentences_file=GOLD_DIR / "quishpi.jsonl",
            test_file=GOLD_DIR / "quishpi-test.jsonl",
            bootstrap_file=GOLD_DIR / "quishpi-bootstrap.jsonl",
        ),
        Dataset(
            name="medical",
            types=tuple(f"T{type_id:02d}" for type_id in MEDICAL_TYPE_IDS),
            sentences_file=GOLD_DIR / "medical.jsonl",
            test_file=GOLD_DIR / "medical-test.jsonl",
            bootstrap_file=GOLD_DIR / "medical-bootstrap.jsonl",
        ),
    )
}


def mention_spans(tags: typing.List[str]) -> typing.List[typing.Tuple[int, int, str]]:
    """The (first, end, type) word spans of BIO tags; an I- tag after another type starts one."""
    spans = []
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        prefix, tag_type = tag.split("-", 1)
        if prefix == "I" and spans and spans[-1][1] == i and spans[-1][2] == tag_type:
            spans[-1] = (spans[-1][0], i + 1, tag_type)
        else:
            spans.append((i, i + 1, tag_type))
    return spans


def documents_from_sentences(sentences: typing.List[test_split.Row]) -> typing.List[data.Document]:
    """Groups sentences into documents, in order, with their tagged mentions as gold.

    Each document must list its sentences in order from 0, as the documents' token indices
    follow from them.
    """
    by_document: typing.Dict[str, typing.List[test_split.Row]] = {}
    for row in sentences:
        by_document.setdefault(row["document name"], []).append(row)
    documents = []
    for name, rows in by_document.items():
        if [r["sentence-ID"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"The sentences of {name} are not numbered 0, 1, ... in order.")
        tokens, mentions = [], []
        for row in rows:
            first = len(tokens)
            tokens += [
                data.Token(
                    text=text,
                    index_in_document=first + i,
                    pos_tag="",
                    sentence_index=row["sentence-ID"],
                )
                for i, text in enumerate(row["tokens"])
            ]
            mentions += [
                # lowercase, like JsonImporter reads the types of gold documents
                data.Mention(
                    type=tag_type.lower(),
                    token_document_indices=tuple(range(first + start, first + end)),
                )
                for start, end, tag_type in mention_spans(row["ner-tags"])
            ]
        documents.append(
            data.Document(
                id=name,
                name=name,
                text=" ".join(t.text for t in tokens),
                category="",
                tokens=tokens,
                mentions=mentions,
                entities=[],
            )
        )
    return documents


def read_documents(path: pathlib.Path) -> typing.List[data.Document]:
    """Reads documents: one sentence per line with BIO tags, or the predictions' document format."""
    with open(path, encoding="utf8") as f:
        first = json.loads(next(line for line in f if line.strip()))
    if "ner-tags" in first:
        return documents_from_sentences(test_split.load_test_split(path))
    return data.JsonImporter(path).do_import()
