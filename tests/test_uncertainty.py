import math

import pytest

from uq_pet.uncertainty import (
    METRICS,
    RANDOM,
    confident_scores,
    length_scores,
    margin_scores,
    max_token_entropy_scores,
    mean_token_entropy_scores,
    metric_names,
    metric_params,
    n_from_percent,
    normalized_entropy,
    random_scores,
    rank_by_uncertainty,
    score_arm,
    select,
    validate_arm,
)


def records():
    return [
        {
            "idx": 0,
            "n_tokens": 2,
            "probabilities": [[1.0, 0.0], [0.9, 0.1]],
        },
        {
            "idx": 1,
            "n_tokens": 3,
            "probabilities": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        },
    ]


def test_normalized_entropy_spans_zero_to_one():
    assert normalized_entropy([1.0, 0.0]) == 0.0
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert normalized_entropy([0.7, 0.2, 0.1]) < 1.0


def test_entropy_normalizes_probability_mass():
    assert normalized_entropy([5, 5]) == pytest.approx(1.0)


def test_mean_and_max_entropy_rank_the_uncertain_sentence_first():
    assert mean_token_entropy_scores(records())[1] > mean_token_entropy_scores(records())[0]
    assert max_token_entropy_scores(records())[1] >= max_token_entropy_scores(records())[0]


def test_margin_is_higher_when_top_classes_are_close():
    assert margin_scores(records())[1] > margin_scores(records())[0]


def test_bad_and_empty_probability_vectors_are_omitted():
    bad = [
        {"idx": 0, "n_tokens": 1, "probabilities": []},
        {"idx": 1, "n_tokens": 1, "probabilities": [[math.nan, 1.0]]},
    ]
    assert mean_token_entropy_scores(bad) == {}


def test_length_reads_the_original_word_count():
    assert length_scores(records()) == {0: 2.0, 1: 3.0}


def test_confident_reverses_a_metric():
    entropy = mean_token_entropy_scores(records())
    assert confident_scores(records()) == {idx: -value for idx, value in entropy.items()}


def test_confident_rejects_unknown_or_recursive_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        confident_scores(records(), metric="missing")
    with pytest.raises(ValueError, match="itself"):
        confident_scores(records(), metric="confident")


def test_random_is_deterministic_by_seed_and_independent_of_record_order():
    forward = random_scores(records(), seed=7)
    backward = random_scores(list(reversed(records())), seed=7)
    assert forward == backward
    assert forward != random_scores(records(), seed=8)


@pytest.mark.parametrize(
    "name",
    [
        RANDOM,
        "mean_token_entropy",
        "max_token_entropy",
        "least_confident",
        "margin",
        "length",
        "confident",
    ],
)
def test_metric_registry_contains_the_bert_strategies(name):
    assert name in METRICS
    assert name in metric_names()


def test_metric_parameters_come_from_signatures():
    assert metric_params(RANDOM) == ["seed"]
    assert metric_params("confident") == ["metric"]
    assert metric_params("mean_token_entropy") == []


def test_arm_validation_and_dispatch():
    validate_arm("margin", {})
    validate_arm(RANDOM, {"seed": 4})
    assert score_arm("margin", records()) == margin_scores(records())
    with pytest.raises(ValueError, match="Unknown strategy"):
        validate_arm("vote_entropy", {})
    with pytest.raises(ValueError, match="unknown"):
        validate_arm("margin", {"window": 3})


def test_ranking_is_descending_and_ties_are_seeded():
    assert rank_by_uncertainty({0: 0.1, 1: 0.9, 2: 0.5}) == [1, 2, 0]
    scores = dict.fromkeys(range(10), 1.0)
    assert rank_by_uncertainty(scores, tie_seed=7) == rank_by_uncertainty(scores, tie_seed=7)
    assert rank_by_uncertainty(scores, tie_seed=7) != rank_by_uncertainty(scores, tie_seed=8)


def test_budget_and_selection(sample_examples):
    assert n_from_percent(328, 10) == 32
    chosen = select({0: 0.1, 1: 0.9, 2: 0.5, 3: 0.2}, sample_examples, 2)
    assert chosen == [sample_examples[1], sample_examples[2]]


def test_selection_refuses_incomplete_scoring(sample_examples):
    with pytest.raises(ValueError, match="incomplete"):
        select({0: 1.0}, sample_examples, 2)
