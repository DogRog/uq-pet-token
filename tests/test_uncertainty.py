import math
import random

import pytest

from uq_pet.uncertainty import (
    avg_neg_logprob,
    extract_logprobs,
    is_tag_token,
    n_from_percent,
    prepare_uncertain_sentences,
    random_sample_sentences,
    rank_by_uncertainty,
    score_records,
    select,
    sentence_scores,
)

# --- scoring ------------------------------------------------------------------


def test_avg_neg_logprob():
    assert avg_neg_logprob([-1.0, -3.0]) == 2.0


def test_avg_neg_logprob_empty_is_nan():
    assert math.isnan(avg_neg_logprob([]))


@pytest.mark.parametrize("token", ["1", " 12", "[0", "3,", "0]"])
def test_is_tag_token_accepts_tag_ids(token):
    assert is_tag_token(token)


@pytest.mark.parametrize("token", ["[", ",", " ", "B-Actor", "", "-1"])
def test_is_tag_token_rejects_the_rest(token):
    assert not is_tag_token(token)


def test_extract_logprobs_keeps_everything_by_default(sample_records):
    logprobs = extract_logprobs(sample_records)
    assert logprobs[0][0] == [-0.1, -1.0, -0.1, -3.0, -2.0]


def test_extract_logprobs_digits_only_drops_punctuation(sample_records):
    logprobs = extract_logprobs(sample_records, digits_only=True)
    assert logprobs[0][0] == [-1.0, -3.0, -2.0]  # '[' and ',' gone


def test_extract_logprobs_skips_error_records(sample_records):
    assert 2 not in extract_logprobs(sample_records)


def test_sentence_scores_averages_over_samples():
    # sample means: 2.0 and 4.0 -> 3.0
    assert sentence_scores({0: [[-1.0, -3.0], [-4.0, -4.0]]}) == {0: 3.0}


def test_sentence_scores_drops_sentences_with_no_usable_samples():
    assert sentence_scores({0: [[], []], 1: [[-2.0]]}) == {1: 2.0}


def test_score_records_end_to_end(sample_records):
    scores = score_records(sample_records, digits_only=True)
    assert set(scores) == {0}  # idx 1 has empty logprobs, idx 2 errored
    assert scores[0] == pytest.approx(2.0)


# --- ranking and budgets ------------------------------------------------------


def test_rank_by_uncertainty_is_descending():
    assert rank_by_uncertainty({0: 0.5, 1: 9.0, 2: 3.0}) == [1, 2, 0]


def test_rank_by_uncertainty_ties_are_deterministic_per_seed():
    scores = dict.fromkeys(range(10), 1.0)
    assert rank_by_uncertainty(scores, tie_seed=1) == rank_by_uncertainty(scores, tie_seed=1)


def test_rank_by_uncertainty_ties_do_not_follow_insertion_order():
    scores = dict.fromkeys(range(20), 1.0)
    assert rank_by_uncertainty(scores, tie_seed=0) != list(scores)


@pytest.mark.parametrize(
    ("total", "pct", "expected"),
    [(328, 10, 32), (328, 3, 9), (328, 100, 328), (328, 0.1, 1), (10, 25, 2)],
)
def test_n_from_percent(total, pct, expected):
    assert n_from_percent(total, pct) == expected


# --- selection ----------------------------------------------------------------


def test_prepare_uncertain_sentences_picks_the_highest_scores(sample_examples):
    scores = {0: 0.1, 1: 5.0, 2: 3.0, 3: 0.2}
    picked = prepare_uncertain_sentences(scores, sample_examples, 2)
    assert picked == [sample_examples[1], sample_examples[2]]


def test_prepare_uncertain_sentences_raises_when_scores_incomplete(sample_examples):
    """An incomplete cache must fail loudly, not shrink one arm's budget."""
    with pytest.raises(ValueError, match="incomplete"):
        prepare_uncertain_sentences({0: 1.0, 1: 2.0}, sample_examples, 3)


def test_prepare_uncertain_sentences_raises_when_budget_exceeds_pool(sample_examples):
    scores = dict.fromkeys(range(4), 1.0)
    with pytest.raises(ValueError, match="available"):
        prepare_uncertain_sentences(scores, sample_examples, 5)


def test_random_sample_sentences_is_deterministic_per_seed(sample_examples):
    assert random_sample_sentences(sample_examples, 2, seed=7) == random_sample_sentences(
        sample_examples, 2, seed=7
    )


def test_random_sample_sentences_varies_with_seed(sample_examples):
    picks = {
        tuple(ex["document name"] for ex in random_sample_sentences(sample_examples, 2, seed=s))
        for s in range(20)
    }
    assert len(picks) > 1


def test_random_sample_sentences_does_not_touch_the_global_rng(sample_examples):
    """Selection must not perturb the RNG that training determinism relies on."""
    random.seed(1)
    expected = random.random()

    random.seed(1)
    random_sample_sentences(sample_examples, 2, seed=99)
    assert random.random() == expected


def test_random_sample_sentences_raises_when_budget_exceeds_pool(sample_examples):
    with pytest.raises(ValueError, match="available"):
        random_sample_sentences(sample_examples, 99)


def test_select_dispatches_and_arms_have_equal_size(sample_examples):
    scores = {0: 0.1, 1: 5.0, 2: 3.0, 3: 0.2}
    uncertain = select("uncertainty", sample_examples, 2, scores=scores)
    control = select("random", sample_examples, 2, seed=42)
    assert len(uncertain) == len(control) == 2


def test_select_uncertainty_requires_scores(sample_examples):
    with pytest.raises(ValueError, match="needs scores"):
        select("uncertainty", sample_examples, 1)


def test_select_rejects_unknown_strategy(sample_examples):
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        select("entropy", sample_examples, 1)
