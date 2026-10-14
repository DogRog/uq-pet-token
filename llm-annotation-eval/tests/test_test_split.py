import json
import pathlib

import pytest

from autoannotate import JsonExporter, JsonImporter, data
from eval import datasets, test_split

RES_DIR = pathlib.Path(__file__).parent.parent / "res" / "gold"
PET_TYPES = datasets.DATASETS["pet"].types
TAGS = {"O"} | {f"{p}-{t}" for p in "BI" for t in PET_TYPES}


def sentence_row(document, sentence_id, tags):
    return {
        "document name": document.id,
        "sentence-ID": sentence_id,
        "tokens": [t.text for t in document.sentences[sentence_id]],
        "ner-tags": tags,
    }


@pytest.fixture(scope="module")
def test():
    return test_split.load_test_split(RES_DIR / "pet-v1.1-test.jsonl")


class TestTestSplitFile:
    def test_has_the_84_sentences_of_uq_pet_token(self, test):
        assert len(test) == 84
        assert len({(r["document name"], r["sentence-ID"]) for r in test}) == 84
        # the first sentences in the split's order, as uq-pet-token's load_pet_splits returns them
        first = [(r["document name"], r["sentence-ID"]) for r in test[:3]]
        assert first == [("doc-2.1", 27), ("doc-3.7", 3), ("doc-10.2", 5)]

    def test_tags_are_pet_bio_tags(self, test):
        assert all(len(r["tokens"]) == len(r["ner-tags"]) for r in test)
        # all of PET's types but AND Gateway, which no test sentence has
        tags = {tag for r in test for tag in r["ner-tags"]}
        assert tags == TAGS - {"B-AND Gateway", "I-AND Gateway"}

    def test_is_part_of_all_pet_v1_1_sentences(self, test):
        all_sentences = test_split.load_test_split(RES_DIR / "pet-v1.1.jsonl")
        assert len(all_sentences) == 417
        by_key = {(r["document name"], r["sentence-ID"]): r for r in all_sentences}
        assert all(by_key[(r["document name"], r["sentence-ID"])] == r for r in test)

    def test_sentences_match_the_gold_documents(self, test):
        gold = {d.id: d for d in JsonImporter(RES_DIR / "pet.jsonl").do_import()}
        for r in test:
            tokens = [t.text for t in gold[r["document name"]].sentences[r["sentence-ID"]]]
            assert tokens == r["tokens"]


class TestSentenceTags:
    def test_bio_tags_with_pet_type_names(self, make_doc):
        doc = make_doc(
            ["the clerk works", "then the clerk checks the form ."],
            mentions=[
                data.Mention("actor", (1,)),
                data.Mention("actor", (4, 5)),
                data.Mention("activity", (6,)),
                data.Mention("activity data", (7, 8)),
            ],
        )

        assert test_split.sentence_tags(doc, 1, PET_TYPES) == (
            ["O", "B-Actor", "I-Actor", "B-Activity", "B-Activity Data", "I-Activity Data", "O"],
            0,
        )

    def test_overlapping_mention_listed_later_is_dropped(self, make_doc):
        doc = make_doc(
            ["the clerk checks the form"],
            mentions=[data.Mention("activity data", (3, 4)), data.Mention("activity", (2, 3))],
        )

        tags = ["O", "O", "O", "B-Activity Data", "I-Activity Data"]
        assert test_split.sentence_tags(doc, 0, PET_TYPES) == (tags, 1)

    def test_unknown_type_is_kept_and_counts_as_wrong(self, make_doc):
        doc = make_doc(["the clerk works"], mentions=[data.Mention("<xor gateway>", (1,))])
        tags, _ = test_split.sentence_tags(doc, 0, PET_TYPES)
        assert tags == ["O", "B-<xor gateway>", "O"]

        gold = [sentence_row(doc, 0, ["O"] * 3)]
        assert test_split.evaluate(gold, [sentence_row(doc, 0, tags)])["entity_precision"] == 0


class TestPredictTestSplit:
    def test_tags_in_the_test_split_order(self, make_doc):
        a = make_doc(["the clerk works", "he rests"], [data.Mention("actor", (3,))], doc_id="a")
        b = make_doc(["the clerk works"], mentions=[data.Mention("actor", (1,))], doc_id="b")
        test = [sentence_row(b, 0, ["O"] * 3), sentence_row(a, 1, ["O"] * 2)]

        predicted, dropped = test_split.predict_test_split(test, [a, b], PET_TYPES)

        assert [r["ner-tags"] for r in predicted] == [["O", "B-Actor", "O"], ["B-Actor", "O"]]
        assert [(r["document name"], r["sentence-ID"]) for r in predicted] == [("b", 0), ("a", 1)]
        assert dropped == 0

    def test_missing_document_gives_none(self, make_doc):
        a = make_doc(["the clerk works"], doc_id="a")
        assert test_split.predict_test_split([sentence_row(a, 0, ["O"] * 3)], [], PET_TYPES) == (None, 0)

    def test_different_tokens_raise(self, make_doc):
        a = make_doc(["the clerk works"], doc_id="a")
        row = {**sentence_row(a, 0, ["O"] * 3), "tokens": ["the", "clerk", "rests"]}
        with pytest.raises(ValueError, match="differ"):
            test_split.predict_test_split([row], [a], PET_TYPES)


def test_evaluate_needs_exact_spans(make_doc):
    doc = make_doc(["the clerk checks the form"])
    data_tags = ["B-Activity Data", "I-Activity Data"]
    gold = [sentence_row(doc, 0, ["B-Actor", "I-Actor", "B-Activity", *data_tags])]
    # the actor is cut short, so only the activity and the data match
    predicted = [sentence_row(doc, 0, ["O", "B-Actor", "B-Activity", *data_tags])]

    scores = test_split.evaluate(gold, predicted)

    assert scores["entity_precision"] == pytest.approx(2 / 3)
    assert scores["entity_recall"] == pytest.approx(2 / 3)
    assert scores["entity_f1"] == pytest.approx(2 / 3)
    # per type F1: actor 0, activity 1, activity data 1
    assert scores["entity_macro_f1"] == pytest.approx(2 / 3)
    assert scores["token_accuracy"] == pytest.approx(3 / 5)
    # partially, the shortened actor matches too
    assert scores["entity_partial_precision"] == 1
    assert scores["entity_partial_recall"] == 1
    assert scores["entity_partial_f1"] == 1
    assert scores["entity_partial_macro_f1"] == 1


def test_partial_scores_match_each_gold_mention_once():
    # "the clerk and the manager": one gold actor over the whole sentence, two predicted actors inside it, and a wrong type
    gold = [["B-Actor", "I-Actor", "I-Actor", "I-Actor", "I-Actor"]]
    tags = [["B-Actor", "I-Actor", "B-Activity", "B-Actor", "I-Actor"]]

    scores = test_split.partial_scores(gold, tags)

    assert scores["entity_partial_precision"] == pytest.approx(1 / 3)
    assert scores["entity_partial_recall"] == 1
    assert scores["entity_partial_f1"] == pytest.approx(0.5)
    # per type F1: actor 2/3, activity 0
    assert scores["entity_partial_macro_f1"] == pytest.approx(1 / 3)


def test_evaluate_rejects_predictions_in_another_order(make_doc):
    a = make_doc(["x"], doc_id="a")
    b = make_doc(["y"], doc_id="b")
    rows = [sentence_row(a, 0, ["O"]), sentence_row(b, 0, ["O"])]
    with pytest.raises(AssertionError, match="order"):
        test_split.evaluate(rows, rows[::-1])


def test_export_writes_test_predictions_next_to_the_predictions(tmp_path, make_doc):
    doc = make_doc(["the clerk works"], mentions=[data.Mention("actor", (0, 1))], doc_id="a")
    predictions_file = tmp_path / "run-1" / "md" / "pet.jsonl"
    predictions_file.parent.mkdir(parents=True)
    JsonExporter(str(predictions_file)).export([doc])

    out_file = test_split.export(predictions_file, [sentence_row(doc, 0, ["O"] * 3)], PET_TYPES)

    assert out_file == tmp_path / "run-1" / "md" / "test-predictions.jsonl"
    [row] = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert row == {
        "document name": "a", "sentence-ID": 0, "tokens": ["the", "clerk", "works"],
        "ner-tags": ["B-Actor", "I-Actor", "O"],
    }


def test_export_skips_predictions_without_the_test_documents(tmp_path, make_doc):
    predictions_file = tmp_path / "pet.jsonl"
    JsonExporter(str(predictions_file)).export([make_doc(["x"], doc_id="other")])
    test = [sentence_row(make_doc(["x"], doc_id="a"), 0, ["O"])]

    assert test_split.export(predictions_file, test, PET_TYPES) is None
    assert not (tmp_path / "test-predictions.jsonl").exists()
