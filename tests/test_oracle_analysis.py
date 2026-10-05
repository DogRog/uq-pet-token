import json
import statistics

import polars as pl
import pytest

from utils.oracle_analysis import (
    config_gap_table,
    curve_auc,
    oracle_table,
    sweep_aucs,
)

METRICS = ("entropy", "least_confidence", "margin")


def aucs_frame(random_by_config, uq_offsets, seeds=(0, 1, 2)):
    """Random AUC per config and seed; UQ AUC = random + offset[metric] + seed jitter."""
    rows = []
    for config_id, random_aucs in random_by_config.items():
        for metric in METRICS:
            for seed, random_auc in zip(seeds, random_aucs, strict=True):
                rows.append(
                    {
                        "config_id": config_id,
                        "uq_metric": metric,
                        "seed": seed,
                        "random_auc": random_auc,
                        "uq_auc": random_auc + uq_offsets[metric] + 0.001 * seed,
                    }
                )
    return pl.DataFrame(rows)


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


def test_oracle_picks_random_best_by_test_and_measures_gain_gap_and_band():
    aucs = aucs_frame(
        {
            "config_0000": [0.60, 0.62, 0.61],
            "config_0001": [0.70, 0.74, 0.72],
            "config_0002": [0.65, 0.66, 0.67],
        },
        {"entropy": 0.02, "least_confidence": 0.01, "margin": -0.01},
    )

    table = oracle_table(aucs, validation_config_id="config_0002")

    assert table["oracle_config_id"].unique().to_list() == ["config_0001"]
    entropy = table.filter(pl.col("uq_metric") == "entropy").row(0, named=True)
    assert entropy["configs_ranked"] == 3
    assert entropy["oracle_random_auc"] == pytest.approx(0.72)
    assert entropy["validation_random_auc"] == pytest.approx(0.66)
    assert entropy["oracle_gain_pp"] == pytest.approx(6.0)
    assert entropy["oracle_gap_pp"] == pytest.approx(2.1)
    band = statistics.stdev([0.70, 0.74, 0.72])
    assert entropy["oracle_seed_band_sd_pp"] == pytest.approx(100 * band)
    assert entropy["oracle_z"] == pytest.approx(0.021 / band)
    margin = table.filter(pl.col("uq_metric") == "margin").row(0, named=True)
    assert margin["oracle_gap_pp"] == pytest.approx(-0.9)
    # Each metric's own best configuration is reported against the same oracle random arm.
    assert entropy["uq_best_config_id"] == "config_0001"
    assert entropy["symmetric_gap_pp"] == pytest.approx(2.1)


def test_oracle_ties_go_to_lowest_config_id_and_unknown_validation_winner_is_null():
    aucs = aucs_frame(
        {"config_0003": [0.7, 0.7, 0.7], "config_0001": [0.7, 0.7, 0.7]},
        {"entropy": 0.01, "least_confidence": 0.01, "margin": 0.01},
    )

    table = oracle_table(aucs, validation_config_id="config_0099")

    assert table["oracle_config_id"].unique().to_list() == ["config_0001"]
    assert table["validation_random_auc"].null_count() == table.height
    assert table["oracle_gain_pp"].null_count() == table.height
    assert table["oracle_z"].null_count() == table.height  # zero seed band


def test_oracle_ignores_configs_missing_a_metric_or_seed():
    aucs = aucs_frame(
        {"config_0000": [0.60, 0.61, 0.62], "config_0001": [0.90, 0.91, 0.92]},
        {"entropy": 0.02, "least_confidence": 0.02, "margin": 0.02},
    ).filter(~((pl.col("config_id") == "config_0001") & (pl.col("uq_metric") == "margin")))

    table = oracle_table(aucs)

    assert table["oracle_config_id"].unique().to_list() == ["config_0000"]
    assert table["configs_ranked"].unique().to_list() == [1]


def test_oracle_rejects_random_arms_that_differ_across_metrics():
    aucs = aucs_frame(
        {"config_0000": [0.6, 0.6, 0.6]},
        {"entropy": 0.0, "least_confidence": 0.0, "margin": 0.0},
    ).with_columns(
        pl.when(pl.col("uq_metric") == "margin")
        .then(pl.col("random_auc") + 0.01)
        .otherwise(pl.col("random_auc"))
        .alias("random_auc")
    )

    with pytest.raises(ValueError, match="identical across metrics"):
        oracle_table(aucs)


def test_holm_adjusts_across_metrics():
    aucs = aucs_frame(
        {"config_0000": [0.60, 0.65, 0.70]},
        {"entropy": 0.05, "least_confidence": 0.02, "margin": 0.001},
    )

    table = oracle_table(aucs).sort("oracle_p")
    p = table["oracle_p"].to_list()
    holm = table["oracle_p_holm"].to_list()

    assert holm[0] == pytest.approx(min(1.0, 3 * p[0]))
    assert holm == sorted(holm)
    assert all(adjusted >= raw for adjusted, raw in zip(holm, p, strict=True))


def test_config_gap_table_counts_wins_across_all_configs():
    aucs = aucs_frame(
        {"config_0000": [0.6, 0.6, 0.6], "config_0001": [0.7, 0.7, 0.7]},
        {"entropy": 0.02, "least_confidence": 0.0, "margin": -0.02},
    )

    table = config_gap_table(aucs)

    entropy = table.filter(pl.col("uq_metric") == "entropy").row(0, named=True)
    assert (entropy["configs"], entropy["wins"], entropy["losses"]) == (2, 2, 0)
    assert entropy["median_gap_pp"] == pytest.approx(2.1)
    margin = table.filter(pl.col("uq_metric") == "margin").row(0, named=True)
    assert (margin["wins"], margin["losses"], margin["win_fraction"]) == (0, 2, 0.0)


def test_sweep_aucs_reads_completed_test_comparisons_only(tmp_path):
    comparisons = []
    for metric, status in (("entropy", "complete"), ("margin", "pending")):
        run_dir = f"runs/config_0000/{metric}/run"
        if status == "complete":
            (tmp_path / run_dir).mkdir(parents=True)
            pl.DataFrame(
                {
                    "seed": [0, 0, 0, 0],
                    "arm": ["uncertainty", "random", "uncertainty", "random"],
                    "round": [0, 0, 1, 1],
                    "percent_acquired": [0.0, 0.0, 100.0, 100.0],
                    "entity_f1": [0.1, 0.1, 0.9, 0.5],
                }
            ).write_csv(tmp_path / run_dir / "results.csv")
        comparisons.append(
            {"config_id": "config_0000", "uq_metric": metric, "status": status, "run_dir": run_dir}
        )
    (tmp_path / "summary.json").write_text(
        json.dumps({"evaluation_split": "test", "comparisons": comparisons})
    )

    aucs = sweep_aucs(tmp_path)

    assert aucs.to_dicts() == [
        {
            "config_id": "config_0000",
            "uq_metric": "entropy",
            "seed": 0,
            "random_auc": pytest.approx(0.3),
            "uq_auc": pytest.approx(0.5),
        }
    ]


def test_sweep_aucs_refuses_validation_sweeps(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"mode": "random_baseline_tuning"}))

    with pytest.raises(ValueError, match="not a test-split sweep"):
        sweep_aucs(tmp_path)
