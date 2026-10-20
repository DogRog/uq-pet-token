import pytest

from autoannotate import data
from eval import scoring
from eval.metrics import Stats

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


def mentions_doc(make_doc, *mentions, doc_id="doc-1"):
    return make_doc(SENTENCES, mentions=[data.Mention(t, i) for t, i in mentions], doc_id=doc_id)


class TestMentionMetrics:
    def test_overlap_with_same_type_counts_as_match(self, make_doc):
        gold = mentions_doc(make_doc, ("actor", (0, 1)), ("activity data", (3, 4)), ("actor", (6, 7)))
        pred = mentions_doc(make_doc, ("actor", (1,)), ("activity", (3, 4)), ("actor", (5, 6, 7, 8)))

        stats = scoring.total(scoring.mention_stats(gold, pred))

        # the second prediction overlaps but has the wrong type
        assert stats == Stats(num_pred=3, num_gold=3, num_ok=2)

    def test_each_gold_mention_matches_once(self, make_doc):
        gold = mentions_doc(make_doc, ("activity data", (3, 4)))
        pred = mentions_doc(make_doc, ("activity data", (3,)), ("activity data", (4,)))

        stats = scoring.total(scoring.mention_stats(gold, pred))

        assert stats == Stats(num_pred=2, num_gold=1, num_ok=1)

