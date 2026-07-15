import math

import numpy as np
import pytest

from uq_pet.llm_eval import (
    aurc,
    auroc,
    calibration_bins,
    confident_failure_stats,
    expected_calibration_error,
    llm_baseline_metrics,
    risk_coverage,
    token_agreement_correctness,
)

GOLD = ["B-Actor", "I-Actor", "B-Activity", "O"]


def _record(parsed_samples: list[list[str]], gt_tags: list[str]) -> dict:
    return {"parsed_samples": parsed_samples, "gt_tags": gt_tags}


def test_llm_baseline_metrics_perfect_majority_vote():
    cache = {"doc-0::0": _record([GOLD] * 5, GOLD)}
    metrics = llm_baseline_metrics(cache)
    assert metrics["entity_f1"] == 1.0
    assert metrics["token_accuracy"] == 1.0
    assert metrics["n_sentences"] == 1
    assert metrics["single_sample_f1_mean"] == 1.0
    assert metrics["single_sample_f1_std"] == 0.0


def test_llm_baseline_metrics_all_o_scores_zero():
    all_o = ["O"] * len(GOLD)
    cache = {"doc-0::0": _record([all_o] * 5, GOLD)}
    metrics = llm_baseline_metrics(cache)
    assert metrics["entity_f1"] == 0.0
    assert metrics["token_accuracy"] == 0.25  # only the gold "O" matches


def test_llm_baseline_metrics_majority_beats_single_samples():
    # 2 of 3 samples are gold, one is all-O: the vote recovers gold everywhere.
    all_o = ["O"] * len(GOLD)
    cache = {"doc-0::0": _record([GOLD, GOLD, all_o], GOLD)}
    metrics = llm_baseline_metrics(cache)
    assert metrics["entity_f1"] == 1.0
    assert metrics["single_sample_f1_mean"] == pytest.approx(2 / 3)


def test_llm_baseline_metrics_exclude_keys():
    cache = {
        "doc-0::0": _record([GOLD] * 3, GOLD),
        "doc-0::1": _record([GOLD] * 3, GOLD),
    }
    metrics = llm_baseline_metrics(cache, exclude_keys=frozenset({"doc-0::1"}))
    assert metrics["n_sentences"] == 1


def test_llm_baseline_metrics_empty_cache():
    assert llm_baseline_metrics({}) == {"n_sentences": 0}


def test_token_agreement_correctness():
    # Token 0: unanimous and right. Token 1: 3/5 say "O", gt disagrees.
    samples = [
        ["B-Actor", "O"],
        ["B-Actor", "O"],
        ["B-Actor", "O"],
        ["B-Actor", "B-Activity"],
        ["B-Actor", "B-Activity"],
    ]
    cache = {"doc-0::0": _record(samples, ["B-Actor", "B-Activity"])}
    confidence, correct = token_agreement_correctness(cache)
    assert confidence.tolist() == [1.0, 0.6]
    assert correct.tolist() == [True, False]


def test_auroc_perfect_and_inverted():
    correct = np.array([True, True, False, False])
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), correct) == 1.0
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), correct) == 0.0


def test_auroc_uninformative_confidence():
    assert auroc(np.full(6, 0.5), np.array([True, False] * 3)) == 0.5


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([True, True])))


def test_calibration_bins_per_unique_level():
    confidence = np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0])
    correct = np.array([True, True, False, False, True, True])
    bins = calibration_bins(confidence, correct)
    assert bins == [
        {"confidence": 0.8, "accuracy": 0.5, "count": 4},
        {"confidence": 1.0, "accuracy": 1.0, "count": 2},
    ]


def test_expected_calibration_error_hand_computed():
    bins = [
        {"confidence": 0.8, "accuracy": 0.5, "count": 4},
        {"confidence": 1.0, "accuracy": 1.0, "count": 2},
    ]
    # 4/6 * |0.5-0.8| + 2/6 * 0 = 0.2
    assert expected_calibration_error(bins) == pytest.approx(0.2)


def test_confident_failure_stats():
    confidence = np.array([1.0, 1.0, 0.6, 0.6])
    correct = np.array([True, False, True, False])
    stats = confident_failure_stats(confidence, correct)
    assert stats["unanimous_fraction"] == 0.5
    assert stats["confident_error_rate"] == 0.5  # one of two unanimous tokens is wrong
    assert stats["share_of_errors_confident"] == 0.5  # one of two errors is unanimous


def test_confident_failure_stats_no_unanimous_tokens():
    stats = confident_failure_stats(np.array([0.6, 0.8]), np.array([True, False]))
    assert math.isnan(stats["confident_error_rate"])
    assert stats["unanimous_fraction"] == 0.0


def test_risk_coverage_perfect_ranking():
    uncertainty = np.array([0.1, 0.2, 0.3, 0.9])
    error = np.array([0.0, 0.0, 0.0, 1.0])
    coverage, risk = risk_coverage(uncertainty, error)
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert np.all(np.diff(risk) >= 0)
    assert risk[-1] == pytest.approx(error.mean())
    assert aurc(risk) == pytest.approx(np.mean([0, 0, 0, 0.25]))


def test_risk_coverage_last_point_is_overall_error():
    uncertainty = np.array([0.5, 0.1, 0.9])
    error = np.array([1.0, 1.0, 0.0])
    _, risk = risk_coverage(uncertainty, error)
    assert risk[-1] == pytest.approx(error.mean())
