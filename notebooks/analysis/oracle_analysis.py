"""Test-tuned random oracle: compare UQ against random at random's best test configuration.

This is an appendix sensitivity check only. Stage 1 tunes the random arm on the test
split and freezes its winner, the test-tuned random oracle; stage 2 runs every UQ metric
against random at that configuration. Choosing the random arm's settings on the test
split biases the comparison toward random; headline results keep the winners tuned on
the validation holdout (ADR 0007).
"""

import json
from pathlib import Path

import polars as pl


def curve_auc(results: pl.DataFrame, field: str = "entity_f1") -> dict[tuple[int, str], float]:
    """Return normalized ``field`` AUC over acquired-pool percentage per (seed, arm)."""
    aucs = {}
    for (seed, arm), rows in results.group_by("seed", "arm"):
        rows = rows.sort("round")
        x = rows["percent_acquired"].to_list()
        y = rows[field].to_list()
        if len(x) < 2 or any(right <= left for left, right in zip(x, x[1:], strict=False)):
            raise ValueError(f"seed={seed}, arm={arm} needs increasing acquisition points")
        area = sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(x) - 1))
        aucs[int(seed), str(arm)] = area / (x[-1] - x[0])
    return aucs


def find_oracle_comparison(
    uq_root: Path, checkpoint: str, dataset: str, uq_metric: str
) -> dict | None:
    """Return the completed stage 2 comparison for one checkpoint and UQ metric, or None.

    The returned ``run_dir`` is the absolute run directory and ``sweep`` names its sweep.
    """
    for summary_path in sorted(uq_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        search = json.loads((summary_path.parent / "search_config.json").read_text())
        if (
            summary.get("evaluation_split") != "test"
            or search["checkpoint"] != checkpoint
            or search.get("dataset", "pet") != dataset
        ):
            continue
        for comparison in summary["comparisons"]:
            run_dir = summary_path.parent / comparison.get("run_dir", "")
            if (
                comparison["uq_metric"] == uq_metric
                and comparison["status"] == "complete"
                and (run_dir / "results.csv").is_file()
            ):
                return {**comparison, "sweep": summary_path.parent.name, "run_dir": run_dir}
    return None


def search_winner(search_root: Path, checkpoint: str, dataset: str) -> str | None:
    """Return the winning config_id of one checkpoint's random-only tuning search, or None."""
    for config_path in sorted(search_root.glob("*/search_config.json")):
        selection_path = config_path.parent / "selection.json"
        search = json.loads(config_path.read_text())
        if (
            search["checkpoint"] == checkpoint
            and search.get("dataset", "pet") == dataset
            and selection_path.is_file()
        ):
            return json.loads(selection_path.read_text())["config_id"]
    return None


def arm_summary(results: pl.DataFrame, field: str = "entity_f1") -> dict[str, dict[str, float]]:
    """Seed-mean test ``field`` AUC, its seed SD, and seed-mean final ``field`` per arm, in pp."""
    aucs = pl.DataFrame(
        [
            {"seed": seed, "arm": arm, "auc": auc}
            for (seed, arm), auc in curve_auc(results, field).items()
        ]
    )
    finals = results.filter(pl.col("round") == pl.col("round").max().over("seed", "arm"))
    return (
        aucs.group_by("arm")
        .agg((100 * pl.col("auc").mean()).alias("auc"), (100 * pl.col("auc").std()).alias("sd"))
        .join(
            finals.group_by("arm").agg((100 * pl.col(field).mean()).alias("final")),
            on="arm",
        )
        .rows_by_key("arm", named=True, unique=True)
    )


def validation_against_oracle(
    validation: tuple[pl.DataFrame, dict, str | None],
    oracle: tuple[pl.DataFrame, dict, str | None],
    field: str = "entity_f1",
) -> pl.DataFrame:
    """Compare one UQ metric's runs at the validation winner and at the oracle on ``field``.

    Each argument is ``(results, hyperparameters, search_config_id)``. The search
    config_id comes from the shared 100-configuration search, because each rerun holds
    one configuration and its own config_id is always config_0000. Hyperparameter
    columns follow the oracle's keys.
    """
    rows = []
    for configuration, (results, hyperparameters, search_config_id) in (
        ("Validation winner (headline)", validation),
        ("Test-tuned random oracle", oracle),
    ):
        arms = arm_summary(results, field)
        rows.append(
            {
                "configuration": configuration,
                "random_auc_pp": arms["random"]["auc"],
                "random_auc_seed_sd_pp": arms["random"]["sd"],
                "final_random_f1_pp": arms["random"]["final"],
                "uq_auc_pp": arms["uncertainty"]["auc"],
                "gap_auc_pp": arms["uncertainty"]["auc"] - arms["random"]["auc"],
                "final_uq_f1_pp": arms["uncertainty"]["final"],
                "search_config_id": search_config_id,
            }
            | {name: hyperparameters.get(name) for name in oracle[1]}
        )
    return pl.DataFrame(rows)
