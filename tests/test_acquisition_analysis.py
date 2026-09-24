import polars as pl
import pytest

from utils.acquisition_analysis import (
    acquisition_diagnostics,
    paired_performance,
    threshold_efficiency,
)


def saved_logs():
    selections, evaluations = [], []
    for seed in (0, 1):
        for arm in ("random", "uncertainty"):
            order = [0, 1, 2] if arm == "random" else [2, 1, 0]
            scores = [0.2, 0.6, 0.5] if arm == "random" else [0.2, 0.4, 0.7]
            for round_idx, acquired in enumerate((0, 2, 3)):
                evaluations.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "round": round_idx,
                        "n_acquired": acquired,
                        "n_new": (0, 2, 1)[round_idx],
                        "scoreable_pool_tokens": 3,
                        "entity_f1": scores[round_idx],
                    }
                )
            for index, token_id in enumerate(order):
                selections.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "round": 1 if index < 2 else 2,
                        "pool_idx": 0,
                        "word_idx": token_id,
                        "document_name": "doc",
                        "sentence_id": 0,
                        "token": ["A", "a", "run"][token_id],
                        "label": ["O", "O", "B-Activity"][token_id],
                    }
                )
    return selections, pl.DataFrame(evaluations)


def test_round_ties_short_final_batch_and_zero_categories():
    selections, results = saved_logs()
    diagnostics = acquisition_diagnostics(selections, results)
    timing = diagnostics["timing"].filter(pl.col("seed") == 0).sort("word_idx")
    assert timing["advance_pp"].to_list() == [-50, 0, 50]
    assert timing["random_position_percent"].to_list() == [50, 50, 100]
    absent = diagnostics["composition"].filter(
        (pl.col("arm") == "random") & (pl.col("round") == 1) & (pl.col("category") == "Activity")
    )
    assert absent["share_percent"].to_list() == [0, 0]
    coverage = diagnostics["coverage"].filter((pl.col("arm") == "random") & (pl.col("round") == 1))
    assert coverage["distinct_words"].to_list() == [1, 1]
    assert coverage["repeated_word_label_percent"].to_list() == [50, 50]
    assert diagnostics["validation"]["same_final_set"].all()


def test_partial_logs_report_matched_subset():
    selections, results = saved_logs()
    diagnostics = acquisition_diagnostics(
        [row for row in selections if row["round"] == 1], results.filter(pl.col("round") <= 1)
    )
    assert not diagnostics["validation"]["full_pool"].any()
    assert not diagnostics["validation"]["same_final_set"].any()
    assert diagnostics["validation"]["matched_tokens"].to_list() == [1, 1, 1, 1]
    assert diagnostics["timing"]["word_idx"].to_list() == [1, 1]


def test_invalid_selection_logs_fail_visibly():
    selections, results = saved_logs()
    with pytest.raises(ValueError, match="more than once"):
        acquisition_diagnostics([*selections, selections[0]], results)
    with pytest.raises(ValueError, match="counts disagree"):
        acquisition_diagnostics(selections[1:], results)


def test_pairing_and_first_crossing_preserve_unreached_targets():
    _, results = saved_logs()
    gaps = paired_performance(results).filter(pl.col("n_acquired") == 2)
    assert gaps["gap_pp"].to_list() == pytest.approx([-20, -20])
    efficiency = threshold_efficiency(results, 0.6)
    assert efficiency["labels_saved"].to_list() == [-1, -1]
    assert threshold_efficiency(results, 0.1)["labels_saved"].to_list() == [0, 0]
    assert threshold_efficiency(results, 0.9)["labels_saved"].null_count() == 2
    one_reached = threshold_efficiency(results, 0.65)
    assert one_reached["random_labels"].null_count() == 2
    assert one_reached["uncertainty_labels"].to_list() == [3, 3]
    with pytest.raises(ValueError, match="same budgets"):
        paired_performance(results.slice(1))
