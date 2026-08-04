import math
import random

import pytest

from conftest import make_choice
from uq_pet.uncertainty import (
    METRICS,
    RANDOM,
    avg_neg_logprob,
    avg_neg_logprob_filtered_scores,
    avg_neg_logprob_pure_scores,
    confident_scores,
    disagreement_scores,
    entity_count_std_scores,
    extract_logprobs,
    extract_tag_ids,
    is_tag_token,
    least_confident_scores,
    length_scores,
    logprob_scores,
    metric_names,
    metric_params,
    n_from_percent,
    normalized_vote_entropy,
    pairwise_f1_disagreement_scores,
    plurality_disagreement,
    random_scores,
    rank_by_uncertainty,
    score_arm,
    select,
    span_f1,
    tag_spans,
    validate_arm,
    vote_entropy_scores,
    votes_by_position,
    worst_neg_logprob,
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


def test_logprob_scores_averages_over_samples():
    # sample means: 2.0 and 4.0 -> 3.0
    assert logprob_scores({0: [[-1.0, -3.0], [-4.0, -4.0]]}) == {0: 3.0}


def test_logprob_scores_drops_sentences_with_no_usable_samples():
    assert logprob_scores({0: [[], []], 1: [[-2.0]]}) == {1: 2.0}


def test_logprob_scores_reduce_replaces_the_per_sample_score():
    assert logprob_scores({0: [[-1.0, -3.0]]}, reduce=max) == {0: -1.0}


# --- least confidence ---------------------------------------------------------


def test_worst_neg_logprob_takes_the_least_confident_token():
    assert worst_neg_logprob([-1.0, -3.0, -0.5], 1) == 3.0


def test_worst_neg_logprob_averages_the_n_worst():
    assert worst_neg_logprob([-1.0, -3.0, -0.5], 2) == 2.0


def test_worst_neg_logprob_tolerates_fewer_tokens_than_asked_for():
    assert worst_neg_logprob([-2.0], 5) == 2.0


def test_worst_neg_logprob_empty_is_nan():
    assert math.isnan(worst_neg_logprob([], 1))


def test_least_confident_reads_tag_tokens_only(sample_records):
    # sample 0's tag tokens are -1.0, -3.0, -2.0 and sample 1's are -2.0 each:
    # worst of each is 3.0 and 2.0 -> 2.5. The '[' at -0.1 must not be considered.
    assert least_confident_scores(sample_records)[0] == pytest.approx(2.5)


def test_least_confident_is_not_diluted_by_easy_tokens():
    """The point of the metric: one hard tag in a long easy sentence still scores high."""
    hard_tag = [("1", -0.01)] * 20 + [("2", -5.0)]
    records = [{"idx": 0, "choices": [make_choice(hard_tag)]}]
    assert least_confident_scores(records)[0] == pytest.approx(5.0)
    assert avg_neg_logprob_filtered_scores(records)[0] < 0.3


def test_least_confident_n_worst_widens_the_window(sample_records):
    assert least_confident_scores(sample_records, n_worst=3) == pytest.approx(
        avg_neg_logprob_filtered_scores(sample_records)  # every tag token, i.e. the mean
    )


def test_least_confident_rejects_a_meaningless_window(sample_records):
    with pytest.raises(ValueError, match="at least 1"):
        least_confident_scores(sample_records, n_worst=0)


# --- output disagreement ------------------------------------------------------

# Hand-computed for votes (3, 4, 3): H = -(2/3 ln 2/3 + 1/3 ln 1/3) = 0.636514,
# normalized by ln 3 = 1.098612.
SPLIT_2_1_ENTROPY = 0.579380


def test_extract_tag_ids_parses_every_sample(tag_records):
    assert extract_tag_ids(tag_records({0: [[0, 3], [0, 4]]})) == {0: [[0, 3], [0, 4]]}


def test_extract_tag_ids_drops_unparsable_samples_and_error_records(tag_records, sample_records):
    assert extract_tag_ids(tag_records({0: [[0, 3], "sorry, I cannot"]})) == {0: [[0, 3]]}
    assert 2 not in extract_tag_ids(sample_records)  # the error record


def test_votes_by_position_pads_short_samples_with_none():
    assert list(votes_by_position([[0, 3, 4], [0, 3]])) == [[0, 0], [3, 3], [4, None]]


def test_normalized_vote_entropy_spans_zero_to_one():
    assert normalized_vote_entropy([3, 3, 3]) == 0.0
    assert normalized_vote_entropy([3, 4, 0]) == pytest.approx(1.0)
    assert normalized_vote_entropy([3, 4, 3]) == pytest.approx(SPLIT_2_1_ENTROPY, abs=1e-6)


def test_plurality_disagreement_counts_the_minority():
    assert plurality_disagreement([3, 3, 3]) == 0.0
    assert plurality_disagreement([3, 4, 0]) == pytest.approx(2 / 3)
    assert plurality_disagreement([3, 4, 3]) == pytest.approx(1 / 3)


@pytest.mark.parametrize("scores", [vote_entropy_scores, disagreement_scores])
def test_identical_samples_score_zero(tag_records, scores):
    records = tag_records({0: [[0, 3, 4], [0, 3, 4], [0, 3, 4]]})
    assert scores(records) == {0: 0.0}


def test_total_disagreement_hits_the_ceiling(tag_records):
    records = tag_records({0: [[0, 1], [1, 2], [2, 0]]})  # every position all-distinct
    assert vote_entropy_scores(records)[0] == pytest.approx(1.0)
    assert disagreement_scores(records)[0] == pytest.approx(2 / 3)


def test_partial_disagreement_is_averaged_over_positions(tag_records):
    # position 0 unanimous, position 1 splits 2-1.
    records = tag_records({0: [[0, 3], [0, 4], [0, 3]]})
    assert vote_entropy_scores(records)[0] == pytest.approx(SPLIT_2_1_ENTROPY / 2, abs=1e-6)
    assert disagreement_scores(records)[0] == pytest.approx(1 / 6)


def test_a_shorter_sample_counts_as_disagreement(tag_records):
    """Samples disagreeing about the token count is variation, so it must raise the score."""
    ragged = vote_entropy_scores(tag_records({0: [[0, 3, 4], [0, 3, 4], [0, 3]]}))
    equal = vote_entropy_scores(tag_records({0: [[0, 3], [0, 3], [0, 3]]}))
    assert ragged[0] > equal[0] == 0.0
    assert ragged[0] == pytest.approx(SPLIT_2_1_ENTROPY / 3, abs=1e-6)


@pytest.mark.parametrize("scores", [vote_entropy_scores, disagreement_scores])
def test_sentences_with_fewer_than_two_usable_samples_are_omitted(tag_records, scores):
    """A sentence nothing can disagree about must be absent, not scored 0."""
    records = tag_records({0: [[0, 3], "no array here"], 1: ["nope", "nope"], 2: [[0], [3]]})
    assert set(scores(records)) == {2}


@pytest.mark.parametrize("scores", [vote_entropy_scores, disagreement_scores])
def test_output_disagreement_needs_no_logprobs(tag_records, scores):
    """The whole point: it scores a cache from a gateway that returns no logprobs."""
    records = tag_records({0: [[0, 3], [0, 4]]})
    assert all("logprobs" not in choice for rec in records for choice in rec["choices"])
    assert scores(records)[0] > 0


# --- span-level disagreement --------------------------------------------------

# 0 = O, 1/2 = B-/I-Actor, 3/4 = B-/I-Activity, 5/6 = B-/I-Activity Data.


def test_tag_spans_reads_bio():
    assert tag_spans([1, 2, 3]) == {(0, 2, "Actor"), (2, 3, "Activity")}


def test_tag_spans_of_an_all_o_sentence_is_empty():
    assert tag_spans([0, 0, 0]) == set()


def test_tag_spans_opens_a_span_on_a_stray_inside_tag():
    assert tag_spans([0, 2, 2]) == {(1, 3, "Actor")}


def test_tag_spans_breaks_a_span_when_the_type_changes():
    assert tag_spans([1, 4]) == {(0, 1, "Actor"), (1, 2, "Activity")}


def test_tag_spans_ignores_ids_outside_the_tag_set():
    """Only reachable from a salvaged truncated response; must not raise."""
    assert tag_spans([1, 99, 1]) == {(0, 1, "Actor"), (2, 3, "Actor")}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [0, 0, 3]),
        ([1, 2, 3], [3, 4, 5]),
        ([1, 0, 3], [0, 2, 0]),
    ],
)
def test_span_f1_matches_seqeval(a, b):
    """Pins the span convention to the one the experiment is scored with."""
    from seqeval.metrics import f1_score

    from uq_pet.dataset import tag_ids_to_labels

    expected = f1_score([tag_ids_to_labels(a)], [tag_ids_to_labels(b)], average="micro")
    assert span_f1(tag_spans(a), tag_spans(b)) == pytest.approx(expected)


def test_span_f1_calls_two_empty_samples_agreement():
    """Where seqeval would report 0.0 — it cannot tell 'nothing to find' from 'found none'."""
    assert span_f1(set(), set()) == 1.0


def test_span_f1_is_symmetric():
    a, b = tag_spans([1, 2, 3]), tag_spans([1, 0, 3])  # the Activity span is the only one shared
    assert span_f1(a, b) == span_f1(b, a) == pytest.approx(0.5)


@pytest.mark.parametrize("scores", [pairwise_f1_disagreement_scores, entity_count_std_scores])
def test_identical_samples_have_no_span_disagreement(tag_records, scores):
    records = tag_records({0: [[1, 2, 3], [1, 2, 3], [1, 2, 3]]})
    assert scores(records) == {0: 0.0}


def test_pairwise_f1_disagreement_averages_over_pairs(tag_records):
    # pairs (0,1) and (0,2) share the Actor span but not the Activity one -> F1 2/3;
    # pair (1,2) is identical -> F1 1. Mean 7/9, so the score is 2/9.
    records = tag_records({0: [[1, 2, 3], [1, 2, 0], [1, 2, 0]]})
    assert pairwise_f1_disagreement_scores(records)[0] == pytest.approx(2 / 9)


def test_pairwise_f1_disagreement_tolerates_ragged_samples(tag_records):
    """Samples of different lengths are exactly what seqeval refuses to compare."""
    records = tag_records({0: [[1, 2, 3], [1, 2]]})
    assert pairwise_f1_disagreement_scores(records)[0] == pytest.approx(1 / 3)


def test_pairwise_f1_disagreement_scores_a_moved_boundary_in_full(tag_records):
    """The reason to score spans: one token decides whether the *entity* is right.

    The samples agree about 3 of 4 tokens, so the position vote reads it as near
    agreement — but they named two different entities, which is what the experiment
    is scored on.
    """
    records = tag_records({0: [[1, 2, 2, 2], [1, 2, 2, 0]]})
    assert pairwise_f1_disagreement_scores(records)[0] == 1.0
    assert disagreement_scores(records)[0] == pytest.approx(1 / 8)


def test_entity_count_std_ignores_where_the_entities_are(tag_records):
    """Same count, different places: the count metric sees agreement, the vote does not."""
    records = tag_records({0: [[1, 0, 0], [0, 0, 1]]})
    assert entity_count_std_scores(records) == {0: 0.0}
    assert disagreement_scores(records)[0] > 0


def test_entity_count_std_grows_with_the_spread(tag_records):
    records = tag_records({0: [[1, 3, 5], [0, 0, 0]]})  # 3 entities vs none
    assert entity_count_std_scores(records)[0] == pytest.approx(1.5)


@pytest.mark.parametrize("scores", [pairwise_f1_disagreement_scores, entity_count_std_scores])
def test_span_metrics_omit_sentences_with_fewer_than_two_samples(tag_records, scores):
    records = tag_records({0: [[1, 2], "no array here"], 1: [[1, 2], [1, 0]]})
    assert set(scores(records)) == {1}


@pytest.mark.parametrize("scores", [pairwise_f1_disagreement_scores, entity_count_std_scores])
def test_span_metrics_need_no_logprobs(tag_records, scores):
    records = tag_records({0: [[1, 2, 3], [0, 0, 0]]})
    assert all("logprobs" not in choice for rec in records for choice in rec["choices"])
    assert scores(records)[0] > 0


# --- controls -----------------------------------------------------------------


def test_length_is_the_token_count(tag_records):
    assert length_scores(tag_records({0: [[1, 2, 3], [1, 2, 3]]})) == {0: 3.0}


def test_length_takes_the_median_over_samples(tag_records):
    """A sample that ignored the token count must not decide the sentence's length."""
    assert length_scores(tag_records({0: [[1, 2], [1, 2], [1]]})) == {0: 2.0}


def test_length_omits_sentences_with_no_parsable_sample(tag_records):
    assert length_scores(tag_records({0: ["sorry, I cannot"], 1: [[1, 2]]})) == {1: 2.0}


def test_confident_reverses_the_metric_it_names(sample_records):
    uncertain = avg_neg_logprob_filtered_scores(sample_records)
    assert confident_scores(sample_records) == {idx: -s for idx, s in uncertain.items()}


def test_confident_ranks_the_most_confident_sentence_first(tag_records):
    records = tag_records({0: [[1, 2], [1, 2]], 1: [[1, 2], [0, 3]]})  # idx 0 unanimous
    assert rank_by_uncertainty(confident_scores(records, metric="disagreement"))[0] == 0
    assert rank_by_uncertainty(disagreement_scores(records))[0] == 1


def test_confident_keeps_the_wrapped_metrics_omissions(tag_records):
    records = tag_records({0: [[1, 2]], 1: [[1, 2], [1, 0]]})
    assert set(confident_scores(records, metric="vote_entropy")) == {1}


def test_confident_rejects_an_unknown_metric(sample_records):
    with pytest.raises(ValueError, match="Unknown metric"):
        confident_scores(sample_records, metric="avg_neg_logprob")


def test_confident_refuses_to_reverse_itself(sample_records):
    with pytest.raises(ValueError, match="itself"):
        confident_scores(sample_records, metric="confident")


# --- the metric registry ------------------------------------------------------


METRIC_NAMES = [
    "avg_neg_logprob_filtered",
    "avg_neg_logprob_pure",
    "least_confident",
    "vote_entropy",
    "disagreement",
    "pairwise_f1_disagreement",
    "entity_count_std",
    "length",
    "confident",
]


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_every_metric_variant_is_registered_under_its_own_name(name):
    """Each variant is its own entry, so an arm is just a name."""
    assert name in METRICS
    assert name in metric_names()


def test_random_is_a_metric_like_any_other():
    """The control is in the registry, so nothing downstream needs a special case."""
    assert RANDOM in METRICS
    assert RANDOM in metric_names()


@pytest.mark.parametrize(
    "name",
    [
        "avg_neg_logprob_filtered",
        "avg_neg_logprob_pure",
        "vote_entropy",
        "disagreement",
        "pairwise_f1_disagreement",
        "entity_count_std",
        "length",
    ],
)
def test_metric_variants_take_no_parameters(name):
    assert metric_params(name) == []


@pytest.mark.parametrize(
    ("name", "params"),
    [(RANDOM, ["seed"]), ("least_confident", ["n_worst"]), ("confident", ["metric"])],
)
def test_metric_params_are_derived_from_the_signature(name, params):
    assert metric_params(name) == params


def test_avg_neg_logprob_filtered_scores_end_to_end(sample_records):
    scores = avg_neg_logprob_filtered_scores(sample_records)
    assert set(scores) == {0}  # idx 1 has empty logprobs, idx 2 errored
    assert scores[0] == pytest.approx(2.0)


def test_filtered_and_pure_differ(sample_records):
    filtered = avg_neg_logprob_filtered_scores(sample_records)
    pure = avg_neg_logprob_pure_scores(sample_records)
    assert filtered != pure
    # "pure" keeps the '[' and ',' logprobs that "filtered" drops
    assert pure[0] == pytest.approx(((0.1 + 1.0 + 0.1 + 3.0 + 2.0) / 5 + 2.0) / 2)


def test_validate_arm_accepts_good_arms():
    validate_arm(RANDOM, {})
    validate_arm(RANDOM, {"seed": 7})
    validate_arm("avg_neg_logprob_pure", {})
    validate_arm("vote_entropy", {})


def test_validate_arm_rejects_a_param_on_a_parameterless_metric():
    with pytest.raises(ValueError, match="accepts no parameters"):
        validate_arm("vote_entropy", {"measure": "vote_entropy"})


def test_validate_arm_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        validate_arm("sequence_variance", {})


def test_validate_arm_rejects_the_old_parameterized_form():
    """`avg_neg_logprob` is now two metrics; the old name must fail loudly."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        validate_arm("avg_neg_logprob", {"tokens": "pure"})


def test_validate_arm_rejects_an_unknown_param_on_random():
    """A typo in a metric param must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown"):
        validate_arm(RANDOM, {"seeds": 7})


def test_score_arm_dispatches_to_random(sample_records):
    assert score_arm(RANDOM, sample_records, seed=7) == random_scores(sample_records, seed=7)


def test_score_arm_dispatches_to_the_metric(sample_records):
    assert score_arm("avg_neg_logprob_filtered", sample_records) == (
        avg_neg_logprob_filtered_scores(sample_records)
    )


def test_score_arm_dispatches_to_disagreement(tag_records):
    records = tag_records({0: [[0, 3], [0, 4]]})
    assert score_arm("disagreement", records) == disagreement_scores(records)


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


# --- the random control -------------------------------------------------------


def pool_records(n: int) -> list[dict]:
    """What `main` hands the control: one bare record per pool sentence."""
    return [{"idx": idx} for idx in range(n)]


def test_random_scores_every_record_it_is_given(sample_records):
    """Including the errored one — the control must not inherit the LLM's blind spots."""
    assert set(random_scores(sample_records)) == {0, 1, 2}


def test_random_scores_are_deterministic_per_seed(sample_records):
    assert random_scores(sample_records, seed=7) == random_scores(sample_records, seed=7)


def test_random_scores_do_not_depend_on_record_order(sample_records):
    """The draw follows the seed, not the order records happened to be cached in."""
    assert random_scores(sample_records[::-1]) == random_scores(sample_records)


def test_random_scores_do_not_touch_the_global_rng(sample_records):
    """Selection must not perturb the RNG that training determinism relies on."""
    random.seed(1)
    expected = random.random()

    random.seed(1)
    random_scores(sample_records, seed=99)
    assert random.random() == expected


# --- selection ----------------------------------------------------------------


def test_select_picks_the_highest_scores(sample_examples):
    scores = {0: 0.1, 1: 5.0, 2: 3.0, 3: 0.2}
    assert select(scores, sample_examples, 2) == [sample_examples[1], sample_examples[2]]


def test_select_raises_when_scores_incomplete(sample_examples):
    """An incomplete cache must fail loudly, not shrink one arm's budget."""
    with pytest.raises(ValueError, match="incomplete"):
        select({0: 1.0, 1: 2.0}, sample_examples, 3)


def test_select_raises_when_budget_exceeds_pool(sample_examples):
    scores = dict.fromkeys(range(4), 1.0)
    with pytest.raises(ValueError, match="available"):
        select(scores, sample_examples, 5)


def test_the_control_and_a_metric_share_one_selection_rule(sample_examples):
    scores = {0: 0.1, 1: 5.0, 2: 3.0, 3: 0.2}
    uncertain = select(scores, sample_examples, 2)
    control = select(random_scores(pool_records(len(sample_examples))), sample_examples, 2)
    assert len(uncertain) == len(control) == 2
    assert uncertain == [sample_examples[1], sample_examples[2]]


def test_random_selection_is_deterministic_per_seed(sample_examples):
    records = pool_records(len(sample_examples))
    assert select(random_scores(records, seed=7), sample_examples, 2) == select(
        random_scores(records, seed=7), sample_examples, 2
    )


def test_random_selection_varies_with_seed(sample_examples):
    records = pool_records(len(sample_examples))
    picks = {
        tuple(
            ex["document name"] for ex in select(random_scores(records, seed=s), sample_examples, 2)
        )
        for s in range(20)
    }
    assert len(picks) > 1
