"""Scores predicted mentions on the PET test split of uq-pet-token, the way it scores its models.

uq-pet-token trains token classifiers on PET v1.1 (PETv1.1-entities.jsonl, one sentence per
line with BIO tags) and evaluates them on a fixed test split: the 417 sentences split 80/20 with
the datasets library's ``train_test_split(test_size=0.2, seed=3407)``. ``res/gold/
pet-v1.1-test.jsonl`` holds those 84 sentences, in the split's order and with their PET v1.1
tags, generated with uq-pet-token's ``load_pet_splits`` at commit 1ea66af (datasets 2.19.2,
PETv1.1-entities.jsonl with sha256 11c42b26...). PET v1.1 corrects some labels of the PET release
in ``res/gold/pet.jsonl`` and has no overlapping mentions, so test scores use its tags.

Predictions come either from annotating whole documents, of which only the mentions in test
sentences are scored, or from annotating each test sentence on its own (the md-test-sentences
task, see ``sentence_documents``), the input uq-pet-token's models get. Both are scored with
seqeval's entity-level micro scores like uq-pet-token: a mention counts only if its type and both
span boundaries match exactly.
"""
import json
import pathlib
import typing

from seqeval import metrics

from autoannotate import data

# the mention types as the parser stores them (lowercase) and as PET v1.1 tags spell them
TAG_TYPES = {
    t.lower(): t
    for t in (
        "Actor",
        "Activity",
        "Activity Data",
        "Further Specification",
        "XOR Gateway",
        "Condition Specification",
        "AND Gateway",
    )
}

Row = typing.Dict[str, typing.Any]


def load_test_split(path: pathlib.Path) -> typing.List[Row]:
    """Reads the test sentences, in PETv1.1-entities.jsonl's format."""
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sentence_id(row: Row) -> str:
    """The id of a test sentence annotated on its own, e.g. "doc-2.1#27"."""
    return f"{row['document name']}#{row['sentence-ID']}"


def sentence_documents(test: typing.List[Row]) -> typing.List[data.Document]:
    """Makes a one-sentence document of every test sentence, to annotate it without context."""
    documents = []
    for row in test:
        tokens = [
            data.Token(text=text, index_in_document=i, pos_tag="", sentence_index=0)
            for i, text in enumerate(row["tokens"])
        ]
        documents.append(data.Document(
            id=sentence_id(row),
            name=sentence_id(row),
            text=" ".join(row["tokens"]),
            category="",
            tokens=tokens,
            mentions=[],
            relations=[],
            entities=[],
        ))
    return documents


def sentence_tags(document: data.Document, sentence_id: int) -> typing.Tuple[typing.List[str], int]:
    """Converts the predicted mentions of one sentence to BIO tags.

    BIO tags can't represent overlapping mentions; a mention overlapping one the LLM listed
    earlier is dropped. A type outside PET's 7 types is kept as written, so it counts as wrong.

    Returns:
        The tags, one per token of the sentence, and the number of dropped mentions.
    """
    sentence = document.sentences[sentence_id]
    first = sentence[0].index_in_document
    tags = ["O"] * len(sentence)
    dropped = 0
    for mention in document.mentions:
        indices = [i - first for i in mention.token_document_indices]
        if not 0 <= indices[0] < len(sentence):
            continue
        if any(tags[i] != "O" for i in indices):
            dropped += 1
            continue
        tag_type = TAG_TYPES.get(mention.type, mention.type)
        for position, i in enumerate(indices):
            tags[i] = ("B-" if position == 0 else "I-") + tag_type
    return tags, dropped


def predict_test_split(
    test: typing.List[Row], documents: typing.List[data.Document]
) -> typing.Tuple[typing.Optional[typing.List[Row]], int]:
    """Puts the predicted documents' mentions on the test sentences as BIO tags.

    Args:
        test: the test sentences
        documents: predicted whole documents, or test sentences annotated on their own

    Returns:
        The test sentences with predicted instead of gold tags, in the test split's order, or
        None if a test sentence's document was not predicted; and the number of mentions
        dropped for overlapping.
    """
    by_id = {d.id: d for d in documents}
    predicted = []
    dropped = 0
    for row in test:
        if sentence_id(row) in by_id:
            document, index = by_id[sentence_id(row)], 0
        elif row["document name"] in by_id:
            document, index = by_id[row["document name"]], row["sentence-ID"]
        else:
            return None, 0
        tokens = [t.text for t in document.sentences[index]]
        if tokens != row["tokens"]:
            raise ValueError(f"Tokens of {sentence_id(row)} differ from the test split.")
        tags, sentence_dropped = sentence_tags(document, index)
        dropped += sentence_dropped
        predicted.append({**row, "ner-tags": tags})
    return predicted, dropped


def evaluate(test: typing.List[Row], predicted: typing.List[Row]) -> typing.Dict[str, float]:
    """Scores like uq-pet-token's ``evaluate_predictions``, with the same keys."""
    for gold_row, predicted_row in zip(test, predicted, strict=True):
        assert (gold_row["document name"], gold_row["sentence-ID"]) == (
            predicted_row["document name"],
            predicted_row["sentence-ID"],
        ), "predictions are not in the test split's order"
    gold = [row["ner-tags"] for row in test]
    tags = [row["ner-tags"] for row in predicted]
    total = sum(len(sentence) for sentence in gold)
    correct = sum(p == g for ps, gs in zip(tags, gold) for p, g in zip(ps, gs))

    def score(function, average):
        return float(function(gold, tags, average=average, zero_division=0))

    return {
        "entity_f1": score(metrics.f1_score, "micro"),
        "entity_macro_f1": score(metrics.f1_score, "macro"),
        "entity_precision": score(metrics.precision_score, "micro"),
        "entity_recall": score(metrics.recall_score, "micro"),
        "token_accuracy": correct / total if total else 0.0,
    }


def export(predictions_file: pathlib.Path, test: typing.List[Row]) -> typing.Optional[pathlib.Path]:
    """Writes test-predictions.jsonl next to a run's pet.jsonl, for uq-pet-token to score.

    Returns:
        The written file, or None if the predictions don't cover the test split.
    """
    documents = data.JsonImporter(predictions_file).do_import()
    predicted, _ = predict_test_split(test, documents)
    if predicted is None:
        return None
    out_file = predictions_file.with_name("test-predictions.jsonl")
    with open(out_file, "w", encoding="utf8") as f:
        for row in predicted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_file
