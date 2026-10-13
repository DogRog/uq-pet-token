import json
import statistics

import polars as pl
import pytest

from utils.oracle_analysis import (
    curve_auc,
    find_oracle_comparison,
    search_winner,
    validation_against_oracle,
)

METRICS = ("entropy", "least_confidence", "margin")


def results_csv(path, arms):
    """Write one seed's two-point curve per arm: (start F1, final F1)."""
    path.mkdir(parents=True)
    rows = [
        {"seed": 0, "arm": arm, "round": round_idx, "percent_acquired": percent, "entity_f1": f1}
        for arm, (start, final) in arms.items()
        for round_idx, (percent, f1) in enumerate(((0.0, start), (100.0, final)))
    ]
    pl.DataFrame(rows).write_csv(path / "results.csv")


def test_curve_auc_normalizes_trapezoids_per_seed_and_arm():
    results = pl.DataFrame(
        {
            "seed": [0, 0, 0, 0],
            "arm": ["random", "random", "uncertainty", "uncertainty"],
            "round": [0, 1, 0, 1],
            "percent_acquired": [0.0, 50.0, 0.0, 50.0],
            "entity_f1": [0.2, 0.6, 0.2, 0.8],
        }
    )

    assert curve_auc(results) == {
        (0, "random"): pytest.approx(0.4),
        (0, "uncertainty"): pytest.approx(0.5),
    }


def write_sweep(root, checkpoint, files, dataset="pet"):
    root.mkdir(parents=True)
    (root / "search_config.json").write_text(
        json.dumps({"checkpoint": checkpoint, "dataset": dataset})
    )
    for name, content in files.items():
        (root / name).write_text(json.dumps(content))


def test_find_oracle_comparison_matches_checkpoint_dataset_and_completed_metric(tmp_path):
    comparisons = [
        {"config_id": "config_0000", "uq_metric": "entropy", "status": "pending"},
        {
            "config_id": "config_0000",
            "uq_metric": "margin",
            "status": "complete",
            "run_dir": "runs/margin",
            "parameters": {"k": 32},
        },
    ]
    summary = {"evaluation_split": "test", "comparisons": comparisons}
    write_sweep(tmp_path / "bert-uq", "bert-base-cased", {"summary.json": summary})
    write_sweep(tmp_path / "conll-uq", "bert-base-cased", {"summary.json": summary}, "conll2003")
    results_csv(tmp_path / "bert-uq/runs/margin", {"random": (0.1, 0.5)})

    found = find_oracle_comparison(tmp_path, "bert-base-cased", "pet", "margin")

    assert found["sweep"] == "bert-uq"
    assert found["run_dir"] == tmp_path / "bert-uq/runs/margin"
    assert found["parameters"] == {"k": 32}
    assert find_oracle_comparison(tmp_path, "bert-base-cased", "pet", "entropy") is None
    assert find_oracle_comparison(tmp_path, "roberta-base", "pet", "margin") is None
    assert find_oracle_comparison(tmp_path, "bert-base-cased", "conll2003", "margin") is None


def test_search_winner_reads_the_checkpoints_selection(tmp_path):
    write_sweep(tmp_path / "bert", "bert-base-cased", {"selection.json": {"config_id": "c7"}})
    write_sweep(tmp_path / "roberta", "roberta-base", {})

    assert search_winner(tmp_path, "bert-base-cased", "pet") == "c7"
    assert search_winner(tmp_path, "roberta-base", "pet") is None
    assert search_winner(tmp_path, "bert-base-cased", "conll2003") is None


def two_seed_results(random_final, uq_final):
    """Two seeds, two points each; seed 1 ends 0.2 higher than seed 0 in both arms."""
    return pl.DataFrame(
        [
            {
                "seed": seed,
                "arm": arm,
                "round": round_idx,
                "percent_acquired": percent,
                "entity_f1": start if round_idx == 0 else final + 0.2 * seed,
            }
            for seed in (0, 1)
            for arm, start, final in (("random", 0.1, random_final), ("uncertainty", 0.1, uq_final))
            for round_idx, percent in enumerate((0.0, 100.0))
        ]
    )


def test_validation_against_oracle_reports_both_arms_and_oracle_hyperparameters():
    table = validation_against_oracle(
        (two_seed_results(0.5, 0.7), {"k": 64, "learning_rate": 2e-5, "checkpoint": "x"}, "c1"),
        (two_seed_results(0.6, 0.7), {"k": 32, "learning_rate": 3e-5}, "c2"),
    )

    validation, oracle = table.to_dicts()
    assert validation["configuration"] == "Validation winner (headline)"
    assert oracle["configuration"] == "Test-tuned random oracle"
    assert validation["random_auc_pp"] == pytest.approx(100 * (0.3 + 0.4) / 2)
    assert validation["gap_auc_pp"] == pytest.approx(10.0)
    assert oracle["gap_auc_pp"] == pytest.approx(5.0)
    assert validation["random_auc_seed_sd_pp"] == pytest.approx(100 * statistics.stdev([0.3, 0.4]))
    assert validation["final_random_f1_pp"] == pytest.approx(60.0)
    assert oracle["final_uq_f1_pp"] == pytest.approx(80.0)
    assert (validation["search_config_id"], oracle["search_config_id"]) == ("c1", "c2")
    assert (validation["k"], oracle["k"]) == (64, 32)
    assert "checkpoint" not in table.columns


def test_validation_against_oracle_scores_the_chosen_entity_field():
    def partial(results):
        return results.with_columns(
            pl.col("entity_f1").alias("entity_partial_f1"), pl.lit(0.0).alias("entity_f1")
        )

    table = validation_against_oracle(
        (partial(two_seed_results(0.5, 0.7)), {"k": 64}, "c1"),
        (partial(two_seed_results(0.6, 0.7)), {"k": 32}, "c2"),
        "entity_partial_f1",
    )

    assert table["gap_auc_pp"].to_list() == pytest.approx([10.0, 5.0])
    assert table["final_random_f1_pp"][0] == pytest.approx(60.0)
