import json
import statistics

import polars as pl
import pytest

from utils.oracle_analysis import comparison_aucs, curve_auc, oracle_table, tuning_aucs

METRICS = ("entropy", "least_confidence", "margin")


def tuning_frame(random_by_config, seeds=(0, 1, 2)):
    return pl.DataFrame(
        [
            {"config_id": config_id, "seed": seed, "random_auc": auc}
            for config_id, aucs in random_by_config.items()
            for seed, auc in zip(seeds, aucs, strict=True)
        ]
    )


def comparison_frame(random_aucs, uq_offsets, seeds=(0, 1, 2)):
    """UQ AUC = random + offset[metric] + seed jitter, at one fixed configuration."""
    return pl.DataFrame(
        [
            {
                "config_id": "config_0000",
                "uq_metric": metric,
                "seed": seed,
                "random_auc": random_auc,
                "uq_auc": random_auc + offset + 0.001 * seed,
            }
            for metric, offset in uq_offsets.items()
            for seed, random_auc in zip(seeds, random_aucs, strict=True)
        ]
    )


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


def test_oracle_table_reports_gain_gap_band_and_rerun_agreement():
    tuning = tuning_frame(
        {
            "config_0000": [0.60, 0.62, 0.61],
            "config_0001": [0.70, 0.74, 0.72],
            "config_0002": [0.65, 0.66, 0.67],
        }
    )
    comparison = comparison_frame(
        [0.70, 0.74, 0.73], {"entropy": 0.02, "least_confidence": 0.01, "margin": -0.01}
    )

    table = oracle_table(tuning, comparison, "config_0001", validation_config_id="config_0002")

    entropy = table.filter(pl.col("uq_metric") == "entropy").row(0, named=True)
    assert entropy["configs_ranked"] == 3
    assert entropy["oracle_config_id"] == "config_0001"
    assert entropy["oracle_random_auc"] == pytest.approx(0.72)
    assert entropy["validation_random_auc"] == pytest.approx(0.66)
    assert entropy["oracle_gain_pp"] == pytest.approx(6.0)
    rerun = statistics.fmean([0.70, 0.74, 0.73])
    assert entropy["oracle_uq_auc"] == pytest.approx(rerun + 0.021)
    assert entropy["oracle_gap_pp"] == pytest.approx(2.1)
    band = statistics.stdev([0.70, 0.74, 0.73])
    assert entropy["oracle_seed_band_sd_pp"] == pytest.approx(100 * band)
    assert entropy["oracle_z"] == pytest.approx(0.021 / band)
    # Stage 2 reruns random at the oracle configuration; the drift is reported, not hidden.
    assert entropy["random_rerun_diff_pp"] == pytest.approx(100 * (rerun - 0.72))
    margin = table.filter(pl.col("uq_metric") == "margin").row(0, named=True)
    assert margin["oracle_gap_pp"] == pytest.approx(-0.9)


def test_oracle_table_leaves_gain_null_without_a_known_validation_winner():
    tuning = tuning_frame({"config_0001": [0.7, 0.7, 0.7]})
    comparison = comparison_frame([0.7, 0.7, 0.7], dict.fromkeys(METRICS, 0.01))

    table = oracle_table(tuning, comparison, "config_0001", validation_config_id="config_0099")

    assert table["validation_random_auc"].null_count() == table.height
    assert table["oracle_gain_pp"].null_count() == table.height
    assert table["oracle_z"].null_count() == table.height  # zero seed band


def test_oracle_table_rejects_random_arms_that_differ_across_metrics():
    comparison = comparison_frame([0.6, 0.6, 0.6], dict.fromkeys(METRICS, 0.0)).with_columns(
        pl.when(pl.col("uq_metric") == "margin")
        .then(pl.col("random_auc") + 0.01)
        .otherwise(pl.col("random_auc"))
        .alias("random_auc")
    )

    with pytest.raises(ValueError, match="identical across metrics"):
        oracle_table(tuning_frame({"config_0000": [0.6, 0.6, 0.6]}), comparison, "config_0000")


def test_oracle_table_needs_a_completed_comparison():
    empty = comparison_frame([0.6], {"entropy": 0.0}, seeds=(0,)).clear()

    with pytest.raises(ValueError, match="no completed UQ metrics"):
        oracle_table(tuning_frame({"config_0000": [0.6, 0.6, 0.6]}), empty, "config_0000")


def test_holm_adjusts_across_metrics():
    comparison = comparison_frame(
        [0.60, 0.65, 0.70], {"entropy": 0.05, "least_confidence": 0.02, "margin": 0.001}
    )

    table = oracle_table(tuning_frame({"c": [0.6, 0.65, 0.7]}), comparison, "c").sort("oracle_p")
    p = table["oracle_p"].to_list()
    holm = table["oracle_p_holm"].to_list()

    assert holm[0] == pytest.approx(min(1.0, 3 * p[0]))
    assert holm == sorted(holm)
    assert all(adjusted >= raw for adjusted, raw in zip(holm, p, strict=True))


def test_tuning_aucs_reads_completed_trials_of_test_tuning_only(tmp_path):
    results_csv(tmp_path / "tuning/config_0000/run", {"random": (0.1, 0.5)})
    (tmp_path / "plan.json").write_text(json.dumps({"tuning_evaluation_split": "test"}))
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "config_id": "config_0000",
                        "status": "complete",
                        "run_dir": "tuning/config_0000/run",
                    },
                    {"config_id": "config_0001", "status": "pending"},
                ]
            }
        )
    )

    assert tuning_aucs(tmp_path).to_dicts() == [
        {"config_id": "config_0000", "seed": 0, "random_auc": pytest.approx(0.3)}
    ]

    (tmp_path / "plan.json").write_text(json.dumps({"tuning_evaluation_split": "validation"}))
    with pytest.raises(ValueError, match="not a test-split tuning sweep"):
        tuning_aucs(tmp_path)


def test_comparison_aucs_reads_completed_test_comparisons_only(tmp_path):
    results_csv(
        tmp_path / "runs/config_0000/entropy/run",
        {"uncertainty": (0.1, 0.9), "random": (0.1, 0.5)},
    )
    comparisons = [
        {
            "config_id": "config_0000",
            "uq_metric": "entropy",
            "status": "complete",
            "run_dir": "runs/config_0000/entropy/run",
        },
        {"config_id": "config_0000", "uq_metric": "margin", "status": "pending"},
    ]
    (tmp_path / "summary.json").write_text(
        json.dumps({"evaluation_split": "test", "comparisons": comparisons})
    )

    assert comparison_aucs(tmp_path).to_dicts() == [
        {
            "config_id": "config_0000",
            "uq_metric": "entropy",
            "seed": 0,
            "random_auc": pytest.approx(0.3),
            "uq_auc": pytest.approx(0.5),
        }
    ]

    (tmp_path / "summary.json").write_text(json.dumps({"mode": "random_baseline_tuning"}))
    with pytest.raises(ValueError, match="not a test-split comparison"):
        comparison_aucs(tmp_path)
