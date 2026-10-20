import pytest

from autoannotate import data
from eval import scoring
from eval.metrics import Stats

SENTENCES = ["the clerk checks the form and the manager signs it"]
# tokens:     0   1     2      3   4    5   6   7       8     9


def doc(make_doc, *mentions, doc_id="doc-1"):
    return make_doc(SENTENCES, mentions=[data.Mention(t, i) for t, i in mentions], doc_id=doc_id)


def test_mention_stats_per_type(make_doc):
    gold = doc(make_doc, ("actor", (0, 1)), ("activity", (2,)), ("actor", (6, 7)))
    pred = doc(make_doc, ("actor", (1,)), ("activity", (3,)), ("activity data", (3, 4)))

    assert scoring.mention_stats(gold, pred) == {
        "activity": Stats(num_pred=1, num_gold=1, num_ok=0),
        "activity data": Stats(num_pred=1, num_gold=0, num_ok=0),
        "actor": Stats(num_pred=1, num_gold=2, num_ok=1),
    }


def test_exact_mentions_need_the_same_tokens(make_doc):
    gold = doc(make_doc, ("actor", (0, 1)), ("activity", (2,)))
    pred = doc(make_doc, ("actor", (1,)), ("activity", (2,)))

    assert scoring.total(scoring.mention_stats(gold, pred, exact=True)) == Stats(2, 2, 1)


def test_total():
    assert scoring.total({"a": Stats(1, 2, 1), "b": Stats(3, 0, 0)}) == Stats(4, 2, 1)


def test_summarize_macro_micro_and_per_type():
    per_document = [
        {"actor": Stats(num_pred=2, num_gold=2, num_ok=2)},  # p = r = 1
        {"actor": Stats(num_pred=1, num_gold=3, num_ok=0), "activity": Stats(num_pred=1, num_gold=1, num_ok=1)},  # p = .5, r = .25
    ]

    summary = scoring.summarize(per_document)

    assert summary["macro"]["precision"] == {"mean": 0.75, "std": 0.25}
    assert summary["macro"]["recall"] == {"mean": 0.625, "std": 0.375}
    # micro: 3 of 4 predictions correct, 3 of 6 gold found
    assert summary["micro"]["precision"] == 0.75
    assert summary["micro"]["recall"] == 0.5
    assert summary["micro"]["f1"] == pytest.approx(0.6)
    assert summary["per_type"]["actor"] == {
        "precision": 2 / 3, "recall": 0.4, "f1": pytest.approx(0.5), "num_gold": 5, "num_pred": 3, "num_ok": 2,
    }
    assert list(summary["per_type"]) == ["activity", "actor"]


def test_score_only_scores_predicted_documents(make_doc):
    gold = [doc(make_doc, ("actor", (1,)), doc_id="a"), doc(make_doc, ("actor", (1,)), doc_id="b")]
    pred = [doc(make_doc, ("actor", (1,)), doc_id="b")]

    result = scoring.score(gold, pred)

    assert result["documents"] == 1
    assert result["md"]["micro"]["f1"] == 1.0
    assert set(result) == {"documents", "md"}


def test_score_rejects_documents_not_in_gold(make_doc):
    with pytest.raises(AssertionError, match="not in the gold data"):
        scoring.score([doc(make_doc, doc_id="a")], [doc(make_doc, doc_id="x")])
