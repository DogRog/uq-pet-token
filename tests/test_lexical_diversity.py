import polars as pl
import pytest

from analysis.lexical_diversity import dataset_words, pairwise_window_tests, window_ttr

TAGS = ["O", "B-Actor", "I-Actor"]


def sentence(document, sentence_id, tokens, tags):
    return {
        "document_name": document,
        "sentence_id": sentence_id,
        "tokens": tokens,
        "ner_tags": tags,
    }


def test_dataset_words_reads_document_order_lowercases_and_drops_punctuation():
    sentences = [
        sentence("b", 0, ["Later", "."], [0, 0]),
        sentence("a", 1, ["The", "Clerk", "signs"], [0, 1, 0]),
        sentence("a", 0, ["Clerk", "Office", "!"], [1, 2, 0]),
    ]
    words = dataset_words({"PET": (sentences, TAGS)})
    assert words.select("document", "sentence_id", "word", "entity").rows() == [
        ("a", 0, "clerk", True),
        ("a", 0, "office", True),
        ("a", 1, "the", False),
        ("a", 1, "clerk", True),
        ("a", 1, "signs", False),
        ("b", 0, "later", False),
    ]


def test_window_ttr_drops_the_short_final_window_and_can_keep_entity_words_only():
    words = pl.DataFrame(
        {
            "dataset": ["PET"] * 5,
            "word": ["a", "a", "b", "c", "c"],
            "entity": [True, False, True, True, True],
        }
    )
    assert window_ttr(words, 2)["ttr"].to_list() == [0.5, 1.0]
    assert window_ttr(words, 2, entity_only=True)["ttr"].to_list() == [1.0, 0.5]


def test_pairwise_window_tests_compare_every_pair_and_holm_adjust_within_a_word_set():
    windows = pl.DataFrame(
        {
            "dataset": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
            "words_kept": ["all words"] * 18,
            "ttr": [0.5, 0.52, 0.51, 0.53, 0.5, 0.52]
            + [0.8, 0.82, 0.81, 0.83, 0.8, 0.82]
            + [0.51, 0.5, 0.53, 0.52, 0.5, 0.52],
        }
    )
    tests = pairwise_window_tests(windows, ["A", "B", "C"])
    assert tests.select("dataset", "against").rows() == [("B", "A"), ("C", "A"), ("C", "B")]
    raw = sorted(tests["p_value"].to_list())
    adjusted = tests.sort("p_value")["holm_p"].to_list()
    assert adjusted[0] == pytest.approx(3 * raw[0])
    assert adjusted[1] == pytest.approx(max(3 * raw[0], 2 * raw[1]))
    assert adjusted[2] == pytest.approx(min(1.0, max(adjusted[1], raw[2])))
