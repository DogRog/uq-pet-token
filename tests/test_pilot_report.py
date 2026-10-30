"""Check the training-recipe pilot's criteria and pairing without training."""

import json
import shutil
from pathlib import Path

import pytest

import pilot_report

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def curve(seed: int, f1: list[float]) -> list[dict]:
    rounds = len(f1) - 1
    return [
        {
            "seed": seed,
            "arm": "random",
            "round": idx,
            "total_rounds": rounds,
            "percent_acquired": 100 * idx / rounds,
            "entity_f1": value,
        }
        for idx, value in enumerate(f1)
    ]


def test_run_stability_measures_drops_from_the_best_so_far_and_jitter():
    run = pilot_report.run_stability([0.0, 0.2, 0.5, 0.3, 0.6])
    assert run["bootstrap_f1"] == 0.0
    assert run["max_drop"] == pytest.approx(0.2)
    # Rounds 2 and 3 each sit 0.25 from their neighbours' midpoint.
    assert run["jitter"] == pytest.approx(0.25)


def test_recipe_summary_counts_every_seed_and_picks_the_tuning_winner():
    trials = {
        "config_0000": curve(0, [0.0, 0.5, 0.3, 0.5]) + curve(1, [0.1, 0.2, 0.3, 0.4]),
        "config_0001": curve(0, [0.3, 0.6, 0.6, 0.7]) + curve(1, [0.3, 0.5, 0.6, 0.6]),
    }
    summary = pilot_report.recipe_summary(trials, "random_validation_entity_f1_auc")
    assert summary["configs"] == 2
    assert summary["drop_share"] == 0.25
    assert summary["failed_bootstrap_share"] == 0.25
    assert summary["winner"] == "config_0001"
    assert summary["winner_final_sd"] == pytest.approx(0.0707, abs=1e-4)


def recipe(drop_share: float, winner_auc: float) -> dict:
    return {"drop_share": drop_share, "winner_auc": winner_auc}


def test_verdict_needs_three_settings_and_averaging_needs_the_split():
    better = {
        "constant": recipe(0.8, 0.50),
        "split": recipe(0.5, 0.495),
        "split_avg": recipe(0.3, 0.49),
    }
    # Fewer drops, but the winner loses more than the tolerance.
    costly = {**better, "split": recipe(0.5, 0.48)}
    summaries = {"a": better, "b": better, "c": better, "d": costly}
    result = pilot_report.verdict(summaries)
    assert result["split_improves"] == ["a", "b", "c"]
    assert result["adopt_split"] and result["adopt_averaging"]
    result = pilot_report.verdict({**summaries, "c": costly})
    assert not result["adopt_split"] and not result["adopt_averaging"]
    assert result["averaging_improves"] == ["a", "b", "c", "d"]


def test_follow_up_needs_improvement_and_the_gap_in_three_settings():
    better = {"constant": recipe(0.8, 0.50), "constant_avg": recipe(0.5, 0.495)}
    worse = {**better, "constant_avg": recipe(0.9, 0.60)}
    kept = {"constant": 0.03, "averaged": 0.021}
    lost = {"constant": 0.03, "averaged": 0.01}
    summaries = {"a": better, "b": better, "c": better, "d": worse}
    gaps = {"a": kept, "b": kept, "c": kept, "d": lost}
    result = pilot_report.follow_up_verdict(summaries, gaps)
    assert result["averaging_improves"] == ["a", "b", "c"]
    assert result["gap_kept"] == ["a", "b", "c"]
    assert result["adopt"]
    assert not pilot_report.follow_up_verdict(summaries, {**gaps, "c": lost})["adopt"]
    assert not pilot_report.follow_up_verdict({**summaries, "c": worse}, gaps)["adopt"]


def test_every_pilot_config_pairs_with_its_tuning_sweep():
    _, summaries, _ = pilot_report.collect(CONFIGS)
    assert len(summaries) == 4


def test_a_pilot_that_retunes_other_configurations_is_refused(tmp_path):
    shutil.copytree(CONFIGS / "pet", tmp_path / "pet")
    path = tmp_path / "pet" / "pilot" / "bert_base_split.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "sampler_seed": 1}))
    with pytest.raises(ValueError, match="does not pair"):
        pilot_report.collect(tmp_path)
