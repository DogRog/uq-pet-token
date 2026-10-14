"""Descriptive diagnostics from saved acquisition logs; no model or dataset loading."""

import json
from pathlib import Path

import polars as pl


def tag_selections(
    selections: list[dict], results: pl.DataFrame, run_id: str
) -> tuple[pl.DataFrame, list[str]]:
    """Join selections to their round's progress; order labels by distinct pool tokens."""
    frame = (
        pl.DataFrame(selections)
        .with_columns(pl.lit(run_id).alias("run_id"))
        .join(
            results.select("seed", "arm", "round", "percent_acquired", "scoreable_pool_tokens"),
            on=["seed", "arm", "round"],
            how="left",
            validate="m:1",
        )
    )
    label_order = (
        frame.select("pool_idx", "word_idx", "label")
        .unique()
        .group_by("label")
        .len()
        .sort(["len", "label"], descending=[True, False])["label"]
        .to_list()
    )
    return frame, label_order


def pet_sentences(path: Path) -> dict[tuple[str, int], list[str]]:
    """Map (document name, sentence ID) to words from a local PET JSONL file, if present."""
    if not path.is_file():
        return {}
    sentences = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        sentences[record["document name"], record["sentence-ID"]] = record["tokens"]
    return sentences


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
