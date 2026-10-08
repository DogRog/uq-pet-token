import json

import polars as pl
import pytest

from utils.acquisition_analysis import (
    acquisition_diagnostics,
    enrichment_at_budget,
    paired_performance,
    pet_sentences,
    tag_selections,
    threshold_efficiency,
    timing_by_group,
    token_timing,
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


def test_tag_selections_join_progress_and_order_labels_by_pool_tokens():
    selections, results = saved_logs()
    results = results.with_columns((100 * pl.col("n_acquired") / 3).alias("percent_acquired"))

    frame, label_order = tag_selections(selections, results, "run-a")

    assert label_order == ["O", "B-Activity"]
    assert frame.height == len(selections)
    assert frame["run_id"].unique().to_list() == ["run-a"]
    assert frame.filter(pl.col("round") == 2)["percent_acquired"].unique().to_list() == [100.0]


def test_enrichment_pairs_shares_within_seed_at_one_budget():
    selections, results = saved_logs()
    composition = acquisition_diagnostics(selections, results)["composition"].filter(
        pl.col("grouping") == "Entity / O"
    )

    enrichment = enrichment_at_budget(composition, 2).rows_by_key("category", named=True)

    # At 2 tokens, random has two O tokens and UQ has one entity and one O, in both seeds.
    assert enrichment["Entity"][0]["enrichment_pp"] == pytest.approx(50.0)
    assert enrichment["Entity"][0]["seeds_enriched"] == 2
    assert enrichment["O"][0]["enrichment_pp"] == pytest.approx(-50.0)
    assert enrichment["O"][0]["seeds"] == 2


def test_timing_groups_tokens_into_entity_and_o_before_weighting_seeds():
    selections, results = saved_logs()
    timing = acquisition_diagnostics(selections, results)["timing"]

    seed_means, summary = timing_by_group(timing, "entity_status")

    assert set(seed_means["entity_status"]) == {"Entity", "O"}
    by_status = summary.rows_by_key("entity_status", named=True)
    assert by_status["Entity"][0]["mean_advance_pp"] > 0
    assert by_status["Entity"][0]["seeds_earlier"] == 2
    assert summary["entity_status"][0] == "Entity"


def test_token_timing_bolds_the_token_in_matching_context_only(tmp_path):
    selections, results = saved_logs()
    timing = acquisition_diagnostics(selections, results)["timing"]
    path = tmp_path / "pet.jsonl"
    path.write_text(
        json.dumps({"document name": "doc", "sentence-ID": 0, "tokens": ["A", "a", "run"]}) + "\n"
    )

    tokens = token_timing(timing, pet_sentences(path)).rows_by_key("word_idx", named=True)

    assert tokens[2][0]["sentence_context"] == "A a **run**"
    assert tokens[2][0]["paired_seeds"] == 2
    mismatched = token_timing(timing, {("doc", 0): ["B", "b", "walk"]})
    assert set(mismatched["sentence_context"]) == {
        "Local sentence context unavailable or token mismatch"
    }
    assert pet_sentences(tmp_path / "missing.jsonl") == {}
