"""Execute experiments and summarize acquired label coverage."""

from collections.abc import Callable
from pathlib import Path

import polars as pl

from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.config import ExperimentConfig
from uq_pet.pet_data import RESULTS_DIR, download_pet_ner, load_pet_splits
from uq_pet.token_model import get_device, resolve_precision


def execute_experiment(
    config: ExperimentConfig,
    *,
    results_dir: Path = RESULTS_DIR,
    progress_callback: Callable[[list[dict]], None] | None = None,
) -> tuple[dict, dict, list[dict], list[dict], Path]:
    """Load PET, run both arms, persist the run, and return display-ready records."""
    device = get_device()
    effective_precision = resolve_precision(config.precision, device)
    data_path = download_pet_ner()
    seed_examples, pool_inputs, pool_gold, test_examples = load_pet_splits(data_path)
    results, selections = run_active_learning(
        seed_examples,
        pool_inputs,
        pool_gold,
        test_examples,
        **config.active_learning_kwargs(),
        device=device,
        progress_callback=progress_callback,
    )

    first_result = results[0]
    derived_config = {
        "scoreable_pool_tokens": first_result["scoreable_pool_tokens"],
        "token_budget": first_result["token_budget"],
        "rounds": first_result["total_rounds"],
        "effective_precision": effective_precision,
        "effective_pool_percent": (
            100 * first_result["token_budget"] / first_result["scoreable_pool_tokens"]
        ),
    }
    run_config = {**config.resolved_dict(), **derived_config, "evaluation_split": "test"}
    dataset_summary = {
        "mode": "experiment",
        "seed_sentences": len(seed_examples),
        "pool_sentences": len(pool_inputs),
        "pool_tokens": len(pool_gold),
        "scoreable_pool_tokens": derived_config["scoreable_pool_tokens"],
        "acquisition_budget_tokens": derived_config["token_budget"],
        "acquisition_rounds": derived_config["rounds"],
        "effective_pool_percent": derived_config["effective_pool_percent"],
        "test_sentences": len(test_examples),
        "device": str(device),
    }
    run_dir = write_run(run_config, results, selections, results_dir=results_dir)
    return run_config, dataset_summary, results, selections, run_dir


def summarize_label_pool_share(
    selections: pl.DataFrame, acquisition_percent: float
) -> pl.DataFrame:
    """Summarize revealed labels against the full scoreable pool in each run."""
    run_columns = ["run_id", "seed", "arm"]
    label_grid = (
        selections.select(*run_columns, "scoreable_pool_tokens")
        .unique()
        .join(selections.select("label").unique(), how="cross")
    )
    counts = (
        selections.filter(pl.col("percent_acquired") <= acquisition_percent)
        .group_by([*run_columns, "label"])
        .agg(pl.len().alias("n_acquired"))
    )
    return (
        label_grid.join(counts, on=[*run_columns, "label"], how="left", validate="1:1")
        .with_columns(pl.col("n_acquired").fill_null(0))
        .with_columns(
            (100 * pl.col("n_acquired") / pl.col("scoreable_pool_tokens")).alias("pool_share")
        )
        .group_by(["arm", "label"])
        .agg(
            pl.col("pool_share").mean().alias("pool_share_mean"),
            pl.col("pool_share").std().fill_null(0.0).alias("pool_share_std"),
            pl.col("n_acquired").mean().alias("n_acquired_mean"),
            pl.col("scoreable_pool_tokens").mean().alias("scoreable_pool_tokens_mean"),
        )
        .sort(["label", "arm"])
    )


def summarize_label_category_coverage(
    selections: pl.DataFrame, acquisition_percent: float
) -> pl.DataFrame:
    """Summarize acquisition as a percentage of each label's recorded total."""
    run_columns = ["run_id", "seed", "arm"]
    label_totals = (
        selections.select("run_id", "pool_idx", "word_idx", "label")
        .unique()
        .group_by(["run_id", "label"])
        .agg(pl.len().alias("n_label_tokens"))
    )
    label_grid = (
        selections.select(*run_columns)
        .unique()
        .join(label_totals, on="run_id", how="inner", validate="m:m")
    )
    counts = (
        selections.filter(pl.col("percent_acquired") <= acquisition_percent)
        .group_by([*run_columns, "label"])
        .agg(pl.len().alias("n_acquired"))
    )
    return (
        label_grid.join(counts, on=[*run_columns, "label"], how="left", validate="1:1")
        .with_columns(pl.col("n_acquired").fill_null(0))
        .with_columns(
            (100 * pl.col("n_acquired") / pl.col("n_label_tokens")).alias("label_coverage")
        )
        .group_by(["arm", "label"])
        .agg(
            pl.col("label_coverage").mean().alias("label_coverage_mean"),
            pl.col("label_coverage").std().fill_null(0.0).alias("label_coverage_std"),
            pl.col("n_acquired").mean().alias("n_acquired_mean"),
            pl.col("n_label_tokens").mean().alias("n_label_tokens_mean"),
        )
        .sort(["label", "arm"])
    )
