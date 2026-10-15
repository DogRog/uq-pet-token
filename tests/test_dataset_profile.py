from types import SimpleNamespace

import pytest

from analysis.dataset_profile import (
    MEASURES,
    lookup_predictions,
    overview,
    pool_profile,
    size_matched_profile,
    subword_fertility,
    type_frequencies,
)

TAGS = ["O", "B-Actor", "I-Actor", "B-Activity", "I-Activity", "B-Gateway", "I-Gateway"]


def sentence(split, tokens, tags, document="a", sentence_id=0):
    return {
        "split": split,
        "document_name": document,
        "sentence_id": sentence_id,
        "tokens": tokens.split(),
        "ner_tags": [TAGS.index(tag) for tag in tags.split()],
    }


POOL = [
    sentence("pool", "The Clerk Office signs .", "O B-Actor I-Actor B-Activity O"),
    sentence("pool", "Clerk checks signs", "B-Actor B-Activity O", sentence_id=1),
]
TEST = [
    sentence("test", "Clerk Office checks", "B-Actor I-Actor B-Activity", "b"),
    sentence("test", "Manager checks", "B-Actor B-Activity", "b", 1),
]


def test_pool_profile_measures_a_hand_counted_pool():
    profile = pool_profile(POOL, TEST, TAGS)
    assert set(profile) == set(MEASURES)
    assert profile["entity_words_pct"] == pytest.approx(62.5)
    # Two of three types seen, equally often: entropy log 2 over log 3.
    assert profile["type_entropy"] == pytest.approx(0.6309, abs=1e-4)
    assert profile["rare_type_share"] == 1.0
    assert pool_profile(POOL, TEST, TAGS, rare_below=2)["rare_type_share"] == pytest.approx(1 / 3)
    assert profile["multi_word_share"] == 0.25
    assert profile["mean_span_words"] == 1.25
    # "Clerk" opens a sentence and is left out; "Clerk Office" is the one capitalised of three.
    assert profile["capitalised_share"] == pytest.approx(1 / 3)
    assert profile["unseen_entity_rate"] == 0.25
    # "signs" is an Activity once and O once: 2 of 8 words, one bit each.
    assert profile["ambiguous_word_share"] == 0.25
    assert profile["type_entropy_given_word"] == 0.25
    # The lookup finds 3 of 4 gold entities and predicts nothing else.
    assert profile["lookup_f1"] == pytest.approx(6 / 7)


def test_lookup_predictions_continue_an_entity_across_words_of_one_type():
    assert lookup_predictions(POOL, TEST, TAGS) == [
        ["B-Actor", "I-Actor", "B-Activity"],
        ["O", "B-Activity"],
    ]


def test_overview_counts_splits_words_entities_and_types():
    bootstrap = sentence("bootstrap", "Clerk", "B-Actor", "c")
    row = overview({"Toy": ([bootstrap, *POOL, *TEST], TAGS)}).row(0, named=True)
    assert row == {
        "dataset": "Toy",
        "documents": 3,
        "bootstrap_sentences": 1,
        "pool_sentences": 2,
        "test_sentences": 2,
        "words": 14,
        "entities": 9,
        "types": 3,
        "entity_words_pct": pytest.approx(100 * 11 / 14),
    }


def test_type_frequencies_rank_pool_types_and_put_unseen_types_last():
    ranked = type_frequencies({"Toy": ([*POOL, *TEST], TAGS)})
    assert ranked.select("rank", "mentions", "share").rows() == [
        (1, 2, 0.5),
        (2, 2, 0.5),
        (3, 0, 0.0),
    ]
    assert ranked["type"][2] == "Gateway"


def test_size_matched_profile_uses_a_small_pool_whole_and_seeds_its_draws():
    datasets = {"Toy": ([*POOL, *TEST], TAGS)}
    whole = size_matched_profile(datasets, pool_sentences=5)
    assert whole["draws"].unique().to_list() == [1]
    assert (whole["low"] == whole["high"]).all()
    sampled = size_matched_profile(datasets, pool_sentences=1, draws=20, seed=3)
    assert sampled["draws"].unique().to_list() == [20]
    assert sampled.equals(size_matched_profile(datasets, pool_sentences=1, draws=20, seed=3))
    assert sampled.height == len(MEASURES)


def test_subword_fertility_counts_subwords_without_special_tokens():
    def tokenizer(batch, is_split_into_words, add_special_tokens):
        assert is_split_into_words and not add_special_tokens
        # Every word of six or more characters splits in two: Office, checks, Manager, checks.
        word_ids = [
            [index for index, word in enumerate(words) for _ in range(1 + (len(word) > 5))]
            for words in batch
        ]
        return SimpleNamespace(word_ids=lambda row: word_ids[row])

    assert subword_fertility(TEST, tokenizer) == pytest.approx(9 / 5)
