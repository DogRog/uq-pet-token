import json

import polars as pl
import pytest

from utils.paper_results import match_entities, random_sweep_comparisons, tuning_winners

GOLD = ["B-Action", "O", "B-Entity", "I-Entity", "O", "B-Condition"]


def test_exact_match_needs_same_words_and_type():
    predicted = ["B-Action", "O", "B-Entity", "O", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "exact")
    assert spans == [(0, 1, "Action", True), (2, 3, "Entity", False), (5, 6, "Entity", False)]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", False), (5, 6, "Condition", False)]


def test_partial_match_needs_one_shared_word_and_each_gold_matches_once():
    predicted = ["B-Action", "O", "B-Entity", "B-Entity", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "partial")
    assert spans == [
        (0, 1, "Action", True),
        (2, 3, "Entity", True),
        (3, 4, "Entity", False),
        (5, 6, "Entity", False),
    ]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", True), (5, 6, "Condition", False)]


def _write_tuning_sweep(root, name, dataset, complete):
    sweep = root / "random_baseline_search" / name
    sweep.mkdir(parents=True)
    plan = {
        "fixed_config": {"checkpoint": "roberta-base", "dataset": dataset},
        "search_space": {"k": [32, 64], "learning_rate": {"low": 1e-5, "high": 5e-4}},
    }
    (sweep / "plan.json").write_text(json.dumps(plan))
    (sweep / "summary.json").write_text(json.dumps({"complete": complete, "trials": []}))
    if complete:
        (sweep / "selection.json").write_text(
            json.dumps({"config_id": "config_0007", "validation_score": 0.25})
        )
        (sweep / "best_config.json").write_text(
            json.dumps({"checkpoint": "roberta-base", "k": 64, "learning_rate": 3e-5, "seed": 0})
        )


def test_tuning_winners_lists_tuned_settings_of_finished_sweeps(tmp_path):
    _write_tuning_sweep(tmp_path, "done", "quishpi", complete=True)
    _write_tuning_sweep(tmp_path, "running", "pet", complete=False)
    assert tuning_winners(tmp_path).to_dicts() == [
        {
            "dataset": "Quishpi",
            "model": "RoBERTa",
            "config_id": "config_0007",
            "validation_score": 0.25,
            "k": 64,
            "learning_rate": 3e-5,
        }
    ]


def test_random_sweep_comparisons_compute_gaps_for_any_saved_score(tmp_path):
    sweep = tmp_path / "random_search" / "pet-sweep"
    run_dir = sweep / "runs" / "config_0003"
    run_dir.mkdir(parents=True)
    (sweep / "plan.json").write_text(json.dumps({"fixed_config": {"checkpoint": "roberta-base"}}))
    comparison = {
        "status": "complete",
        "run_dir": "runs/config_0003",
        "config_id": "config_0003",
        "uq_metric": "margin",
        "parameters": {"k": 32},
    }
    (sweep / "summary.json").write_text(json.dumps({"comparisons": [comparison]}))
    # Two seeds; each arm runs two rounds over 0-100% of the pool.
    pl.DataFrame(
        {
            "seed": [0, 0, 0, 0, 1, 1, 1, 1],
            "arm": ["random", "random", "uncertainty", "uncertainty"] * 2,
            "round": [0, 1] * 4,
            "percent_acquired": [0.0, 100.0] * 4,
            "entity_macro_f1": [0.2, 0.6, 0.4, 0.6, 0.2, 0.4, 0.2, 0.8],
        }
    ).write_csv(run_dir / "results.csv")
    row = random_sweep_comparisons(tmp_path, "entity_macro_f1").row(0, named=True)
    assert (row["dataset"], row["model"], row["config_id"], row["k"]) == (
        "PET",
        "RoBERTa",
        "config_0003",
        32,
    )
    # AUCs: random 0.4 and 0.3, uncertainty 0.5 and 0.5; finals 0.6/0.6 and 0.4/0.8.
    assert row["gap_auc"] == pytest.approx(0.15)
    assert row["final_gap"] == pytest.approx(0.2)
    assert row["random_auc"] == pytest.approx(0.35)
    assert row["random_final"] == pytest.approx(0.5)
    assert random_sweep_comparisons(tmp_path, "entity_partial_macro_f1").is_empty()
