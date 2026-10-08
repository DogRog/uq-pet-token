import json

import polars as pl
import pytest

from utils.comparison_summaries import load_test_comparisons, uq_gap_ttests


def write_summary(root, name, comparisons, split="test"):
    (root / name).mkdir(parents=True)
    (root / name / "summary.json").write_text(
        json.dumps({"evaluation_split": split, "comparisons": comparisons})
    )


def test_load_test_comparisons_reads_final_scores_of_completed_test_runs(tmp_path):
    run = tmp_path / "bert/runs/entropy"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "seed": [0, 0, 0, 0, 1, 1, 1, 1],
            "arm": ["random", "uncertainty"] * 4,
            "round": [0, 0, 1, 1, 0, 0, 1, 1],
            "entity_f1": [0.1, 0.1, 0.5, 0.7, 0.1, 0.1, 0.6, 0.8],
        }
    ).write_csv(run / "results.csv")
    complete = {
        "config_id": "config_0000",
        "uq_metric": "entropy",
        "status": "complete",
        "run_dir": "runs/entropy",
        "mean_test_entity_f1_gap_auc": 0.04,
        "mean_final_test_entity_f1_gap": 0.2,
    }
    failed = {"config_id": "config_0000", "uq_metric": "margin", "status": "failed", "error": "OOM"}
    (tmp_path / "bert/summary.json").write_text(
        json.dumps({"evaluation_split": "test", "comparisons": [complete, failed]})
    )
    write_summary(tmp_path, "validation-sweep", [complete], split="validation")

    rows = load_test_comparisons(tmp_path).to_dicts()

    assert [(row["model_run"], row["uq_metric"]) for row in rows] == [
        ("bert", "entropy"),
        ("bert", "margin"),
    ]
    assert rows[0]["final_random_f1"] == pytest.approx(0.55)
    assert rows[0]["final_uq_f1"] == pytest.approx(0.75)
    assert rows[0]["gap_auc"] == 0.04
    assert rows[0]["run_path"] == str(run)
    assert rows[1]["final_random_f1"] is None
    assert rows[1]["error"] == "OOM"
    assert rows[1]["run_path"] == ""


def test_load_test_comparisons_returns_an_empty_frame_without_summaries(tmp_path):
    overview = load_test_comparisons(tmp_path)

    assert overview.is_empty()
    assert "status" in overview.columns


def test_uq_gap_ttests_adjusts_across_the_three_planned_metrics(tmp_path):
    gaps = {"entropy": [0.01, 0.02, 0.03], "least_confidence": [0.0, 0.01, -0.01], "margin": [0.05]}
    write_summary(
        tmp_path,
        "bert-random-5-seeds",
        [
            {"uq_metric": metric, "status": "complete", "mean_test_entity_f1_gap_auc": gap}
            for metric, values in gaps.items()
            for gap in values
        ]
        + [{"uq_metric": "entropy", "status": "failed"}],
    )

    rows = {row["UQ metric"]: row for row in uq_gap_ttests(tmp_path)}

    assert set(rows) == {"entropy", "least_confidence"}
    entropy = rows["entropy"]
    assert entropy["Model"] == "bert"
    assert entropy["Configurations"] == 3
    assert entropy["Mean improvement (pp)"] == pytest.approx(2.0)
    assert entropy["Adjusted p-value (Bonferroni)"] == pytest.approx(
        min(1.0, 3 * entropy["p-value"])
    )
    assert entropy["Adjusted p-value (Holm)"] == pytest.approx(3 * entropy["p-value"])
    assert rows["least_confidence"]["Adjusted p-value (Holm)"] == pytest.approx(1.0)
