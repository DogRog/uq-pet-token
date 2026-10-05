"""Test-tuned random oracle: compare UQ against random at random's best test configuration.

This is an appendix sensitivity check only. Stage 1 tunes the random arm on the test
split and freezes its winner, the test-tuned random oracle; stage 2 runs every UQ metric
against random at that configuration. Choosing the random arm's settings on the test
split biases the comparison toward random; headline results keep the winners tuned on
the validation holdout (ADR 0007).
"""

import json
import statistics
from pathlib import Path

import polars as pl
from scipy import stats

TUNING_SCHEMA = {"config_id": pl.String, "seed": pl.Int64, "random_auc": pl.Float64}
COMPARISON_SCHEMA = {
    "config_id": pl.String,
    "uq_metric": pl.String,
    "seed": pl.Int64,
    "random_auc": pl.Float64,
    "uq_auc": pl.Float64,
}


def curve_auc(results: pl.DataFrame) -> dict[tuple[int, str], float]:
    """Return normalized entity-F1 AUC over acquired-pool percentage per (seed, arm)."""
    aucs = {}
    for (seed, arm), rows in results.group_by("seed", "arm"):
        rows = rows.sort("round")
        x = rows["percent_acquired"].to_list()
        y = rows["entity_f1"].to_list()
        if len(x) < 2 or any(right <= left for left, right in zip(x, x[1:], strict=False)):
            raise ValueError(f"seed={seed}, arm={arm} needs increasing acquisition points")
        area = sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(x) - 1))
        aucs[int(seed), str(arm)] = area / (x[-1] - x[0])
    return aucs


def tuning_aucs(sweep_root: Path) -> pl.DataFrame:
    """Per-seed random test AUC for every completed trial of a test-split tuning sweep."""
    plan = json.loads((sweep_root / "plan.json").read_text())
    if plan.get("tuning_evaluation_split") != "test":
        raise ValueError(f"{sweep_root} is not a test-split tuning sweep")
    summary = json.loads((sweep_root / "summary.json").read_text())
    rows = []
    for trial in summary["trials"]:
        if trial["status"] != "complete":
            continue
        aucs = curve_auc(pl.read_csv(sweep_root / trial["run_dir"] / "results.csv"))
        rows.extend(
            {"config_id": trial["config_id"], "seed": seed, "random_auc": auc}
            for (seed, arm), auc in sorted(aucs.items())
            if arm == "random"
        )
    return pl.DataFrame(rows, schema=TUNING_SCHEMA)


def comparison_aucs(sweep_root: Path) -> pl.DataFrame:
    """Per-seed random and uncertainty AUC for every completed comparison in a test sweep."""
    summary = json.loads((sweep_root / "summary.json").read_text())
    if summary.get("evaluation_split") != "test":
        raise ValueError(f"{sweep_root} is not a test-split comparison")
    rows = []
    for comparison in summary["comparisons"]:
        if comparison["status"] != "complete":
            continue
        aucs = curve_auc(pl.read_csv(sweep_root / comparison["run_dir"] / "results.csv"))
        for seed in sorted({seed for seed, _ in aucs}):
            rows.append(
                {
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "seed": seed,
                    "random_auc": aucs[seed, "random"],
                    "uq_auc": aucs[seed, "uncertainty"],
                }
            )
    return pl.DataFrame(rows, schema=COMPARISON_SCHEMA)


def _holm(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values across the planned UQ metrics, preserving input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def oracle_table(
    tuning: pl.DataFrame,
    comparison: pl.DataFrame,
    oracle_config_id: str,
    validation_config_id: str | None = None,
) -> pl.DataFrame:
    """Compare each UQ metric against random at the frozen test-tuned random oracle.

    The oracle gain uses stage 1's random-only runs. The gap pairs stage 2's uncertainty
    and random arms, which share one bootstrap per model seed. The z-score divides the
    seed-mean gap by the seed band: the standard deviation of stage 2's random AUC across
    model seeds.
    """
    if comparison.is_empty():
        raise ValueError("The oracle comparison has no completed UQ metrics yet.")
    spread = comparison.group_by("seed").agg(
        (pl.col("random_auc").max() - pl.col("random_auc").min()).alias("spread")
    )
    if (spread["spread"] > 1e-12).any():
        raise ValueError("The random arm must be identical across metrics within a seed.")
    per_config = tuning.group_by("config_id").agg(pl.col("random_auc").mean())

    def tuned(config_id):
        row = per_config.filter(pl.col("config_id") == (config_id or ""))
        return row["random_auc"][0] if row.height else None

    oracle_random = tuned(oracle_config_id)
    validation_random = tuned(validation_config_id)
    rows = []
    for metric in sorted(comparison["uq_metric"].unique().to_list()):
        paired = comparison.filter(pl.col("uq_metric") == metric).sort("seed")
        gaps = (paired["uq_auc"] - paired["random_auc"]).to_list()
        band = statistics.stdev(paired["random_auc"].to_list())
        gap = statistics.fmean(gaps)
        rerun_random = paired["random_auc"].mean()
        rows.append(
            {
                "uq_metric": metric,
                "configs_ranked": per_config.height,
                "oracle_config_id": oracle_config_id,
                "validation_config_id": validation_config_id,
                "validation_random_auc": validation_random,
                "oracle_random_auc": oracle_random,
                "oracle_gain_pp": None
                if oracle_random is None or validation_random is None
                else 100 * (oracle_random - validation_random),
                "oracle_uq_auc": paired["uq_auc"].mean(),
                "oracle_gap_pp": 100 * gap,
                "oracle_seed_band_sd_pp": 100 * band,
                "oracle_z": gap / band if band > 0 else None,
                "oracle_p": float(stats.ttest_1samp(gaps, 0).pvalue),
                "random_rerun_diff_pp": None
                if oracle_random is None
                else 100 * (rerun_random - oracle_random),
            }
        )
    for row, adjusted in zip(rows, _holm([row["oracle_p"] for row in rows]), strict=True):
        row["oracle_p_holm"] = adjusted
    return pl.DataFrame(rows)
