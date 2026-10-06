import importlib.util
import pathlib

import pytest

from autoannotate import data
from eval.metrics import Stats


def _load_llm_metrics():
    # the script's file name has a hyphen, so it can't be imported by name
    path = pathlib.Path(__file__).parent.parent / "src" / "llm-metrics.py"
    spec = importlib.util.spec_from_file_location("llm_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm_metrics = _load_llm_metrics()

SENTENCES = ["the clerk checks the form and the manager signs it"]
# tokens:     0   1     2      3   4    5   6   7       8     9


class TestStats:
    def test_scores(self):
        stats = Stats(num_pred=4, num_gold=8, num_ok=2)
        assert stats.precision == 0.5
        assert stats.recall == 0.25
        assert stats.f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)

    def test_nothing_predicted_nothing_expected_is_perfect(self):
        stats = Stats(num_pred=0, num_gold=0, num_ok=0)
        assert (stats.precision, stats.recall, stats.f1) == (1.0, 1.0, 1.0)

    def test_nothing_predicted(self):
        stats = Stats(num_pred=0, num_gold=3, num_ok=0)
        assert (stats.precision, stats.recall, stats.f1) == (0.0, 0.0, 0)

    def test_nothing_expected(self):
        stats = Stats(num_pred=3, num_gold=0, num_ok=0)
        assert (stats.precision, stats.recall, stats.f1) == (0.0, 0.0, 0)

    def test_add(self):
        assert Stats(1, 2, 1) + Stats(3, 4, 2) == Stats(4, 6, 3)


@pytest.fixture
def write(tmp_path, make_doc):
    """Writes gold and predicted documents, returns (prediction dir, gold file)."""
    def _write(gold, predictions):
        gold_file = tmp_path / "gold.jsonl"
        data.JsonExporter(str(gold_file)).export(gold)
        pred_dir = tmp_path / "pred" / "model" / "end-to-end"
        pred_dir.mkdir(parents=True)
        data.JsonExporter(str(pred_dir / "pet.jsonl")).export(predictions)
        return pred_dir, gold_file
    return _write


def mentions_doc(make_doc, *mentions, relations=(), doc_id="doc-1"):
    return make_doc(SENTENCES, mentions=[data.Mention(t, i) for t, i in mentions], relations=relations, doc_id=doc_id)


class TestMentionMetrics:
    def test_overlap_with_same_type_counts_as_match(self, write, make_doc):
        gold = mentions_doc(make_doc, ("actor", (0, 1)), ("activity data", (3, 4)), ("actor", (6, 7)))
        pred = mentions_doc(make_doc, ("actor", (1,)), ("activity", (3, 4)), ("actor", (5, 6, 7, 8)))

        [stats] = llm_metrics.load_ner_metrics(*write([gold], [pred]))

        # the second prediction overlaps but has the wrong type
        assert stats == Stats(num_pred=3, num_gold=3, num_ok=2)

    def test_each_gold_mention_matches_once(self, write, make_doc):
        gold = mentions_doc(make_doc, ("activity data", (3, 4)))
        pred = mentions_doc(make_doc, ("activity data", (3,)), ("activity data", (4,)))

        [stats] = llm_metrics.load_ner_metrics(*write([gold], [pred]))

        assert stats == Stats(num_pred=2, num_gold=1, num_ok=1)

    def test_documents_without_prediction_are_skipped(self, write, make_doc):
        gold = [mentions_doc(make_doc, ("actor", (1,)), doc_id="a"), mentions_doc(make_doc, ("actor", (1,)), doc_id="b")]
        pred = [mentions_doc(make_doc, ("actor", (1,)), doc_id="b")]

        stats = llm_metrics.load_ner_metrics(*write(gold, pred))

        assert stats == [Stats(num_pred=1, num_gold=1, num_ok=1)]


class TestRelationMetrics:
    def test_head_and_tail_must_overlap_and_type_must_match(self, write, make_doc):
        mentions = [("actor", (1,)), ("activity", (2,)), ("activity data", (3, 4)), ("actor", (7,)), ("activity", (8,))]
        gold = mentions_doc(make_doc, *mentions, relations=[
            data.Relation("actor performer", 1, 0),
            data.Relation("uses", 1, 2),
            data.Relation("actor performer", 4, 3),
        ])
        # predicted mentions are listed in another order and with other spans than in gold
        pred = mentions_doc(
            make_doc, ("activity", (2,)), ("actor", (0, 1)), ("activity data", (4,)), ("activity", (8,)), ("actor", (6, 7)),
            relations=[
                data.Relation("actor performer", 0, 1),  # match
                data.Relation("flow", 0, 2),  # wrong type
                data.Relation("actor performer", 3, 1),  # wrong tail
            ],
        )

        [stats] = llm_metrics.load_re_metrics(*write([gold], [pred]))

        assert stats == Stats(num_pred=3, num_gold=3, num_ok=1)

    def test_each_gold_relation_matches_once(self, write, make_doc):
        gold = mentions_doc(make_doc, ("actor", (1,)), ("activity", (2,)), relations=[data.Relation("actor performer", 1, 0)])
        pred = mentions_doc(make_doc, ("actor", (1,)), ("activity", (2,)), relations=[
            data.Relation("actor performer", 1, 0),
            data.Relation("actor performer", 1, 0),
        ])

        [stats] = llm_metrics.load_re_metrics(*write([gold], [pred]))

        assert stats == Stats(num_pred=2, num_gold=1, num_ok=1)
