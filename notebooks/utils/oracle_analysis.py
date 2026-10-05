"""Test-tuned random oracle: rank a test sweep by the random arm's own test score.

This is an appendix sensitivity check only. Choosing the random arm's settings on the
test split biases the comparison toward random; headline results keep the winners tuned
on the validation holdout (ADR 0007).
"""

import json
import statistics
from pathlib import Path

import polars as pl
from scipy import stats

AUC_SCHEMA = {
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


def sweep_aucs(sweep_root: Path) -> pl.DataFrame:
    """Per-seed random and uncertainty AUC for every completed comparison in a test sweep."""
    summary = json.loads((sweep_root / "summary.json").read_text())
    if summary.get("evaluation_split") != "test":
        raise ValueError(f"{sweep_root} is not a test-split sweep")
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
    return pl.DataFrame(rows, schema=AUC_SCHEMA)


def _complete_configs(aucs: pl.DataFrame) -> list[str]:
    """Configurations whose every seed has all metrics, after checking random is shared."""
    metrics = aucs["uq_metric"].n_unique()
    seeds = aucs["seed"].n_unique()
    per_seed = aucs.group_by("config_id", "seed").agg(
        (pl.col("random_auc").max() - pl.col("random_auc").min()).alias("spread"),
        pl.col("uq_metric").n_unique().alias("metrics"),
    )
    if (per_seed["spread"] > 1e-12).any():
        raise ValueError("The random arm must be identical across metrics within a seed.")
    complete = (
        per_seed.filter(pl.col("metrics") == metrics)
        .group_by("config_id")
        .agg(pl.len().alias("seeds"))
        .filter(pl.col("seeds") == seeds)
    )
    return sorted(complete["config_id"].to_list())


def _holm(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values across the planned UQ metrics, preserving input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def oracle_table(aucs: pl.DataFrame, validation_config_id: str | None = None) -> pl.DataFrame:
    """Pick random's best configuration by mean test AUC; compare each UQ metric there.

    Ties go to the lowest config ID, as in validation tuning. The z-score divides the
    seed-mean gap by the standard deviation of the random arm's AUC across model seeds
    at the oracle configuration (the seed band).
    """
    complete = _complete_configs(aucs)
    if not complete:
        raise ValueError("No configuration has every metric complete for every seed.")
    scored = aucs.filter(pl.col("config_id").is_in(complete))
    random = (
        scored.unique(["config_id", "seed"])
        .group_by("config_id")
        .agg(pl.col("random_auc").mean())
        .sort(["random_auc", "config_id"], descending=[True, False])
    )
    oracle_id, oracle_random = random.row(0)
    validation = random.filter(pl.col("config_id") == (validation_config_id or ""))
    validation_random = validation["random_auc"][0] if validation.height else None

    rows = []
    for metric in sorted(scored["uq_metric"].unique().to_list()):
        at_oracle = scored.filter(
            (pl.col("config_id") == oracle_id) & (pl.col("uq_metric") == metric)
        ).sort("seed")
        gaps = (at_oracle["uq_auc"] - at_oracle["random_auc"]).to_list()
        band = statistics.stdev(at_oracle["random_auc"].to_list())
        gap = statistics.fmean(gaps)
        uq_best_id, uq_best = (
            scored.filter(pl.col("uq_metric") == metric)
            .group_by("config_id")
            .agg(pl.col("uq_auc").mean())
            .sort(["uq_auc", "config_id"], descending=[True, False])
            .row(0)
        )
        rows.append(
            {
                "uq_metric": metric,
                "configs_ranked": len(complete),
                "oracle_config_id": oracle_id,
                "oracle_random_auc": oracle_random,
                "validation_config_id": validation_config_id,
                "validation_random_auc": validation_random,
                "oracle_gain_pp": None
                if validation_random is None
                else 100 * (oracle_random - validation_random),
                "oracle_uq_auc": at_oracle["uq_auc"].mean(),
                "oracle_gap_pp": 100 * gap,
                "oracle_seed_band_sd_pp": 100 * band,
                "oracle_z": gap / band if band > 0 else None,
                "oracle_p": float(stats.ttest_1samp(gaps, 0).pvalue),
                "uq_best_config_id": uq_best_id,
                "uq_best_auc": uq_best,
                "symmetric_gap_pp": 100 * (uq_best - oracle_random),
            }
        )
    for row, adjusted in zip(rows, _holm([row["oracle_p"] for row in rows]), strict=True):
        row["oracle_p_holm"] = adjusted
    return pl.DataFrame(rows)


def config_gap_table(aucs: pl.DataFrame) -> pl.DataFrame:
    """Summarize UQ-minus-random AUC across every completed configuration, without tuning."""
    gaps = (
        aucs.filter(pl.col("config_id").is_in(_complete_configs(aucs)))
        .group_by("uq_metric", "config_id")
        .agg(((pl.col("uq_auc") - pl.col("random_auc")).mean() * 100).alias("gap_pp"))
    )
    return (
        gaps.group_by("uq_metric")
        .agg(
            pl.len().alias("configs"),
            (pl.col("gap_pp") > 1e-10).sum().alias("wins"),
            (pl.col("gap_pp") < -1e-10).sum().alias("losses"),
            pl.col("gap_pp").median().alias("median_gap_pp"),
            pl.col("gap_pp").mean().alias("mean_gap_pp"),
            pl.col("gap_pp").min().alias("min_gap_pp"),
        )
        .with_columns((pl.col("wins") / pl.col("configs")).alias("win_fraction"))
        .sort("uq_metric")
    )
