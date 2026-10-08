"""Match saved runs that differ only in uncertainty acquisition options (ADR 0009).

Reads config.json, results.csv, and selections.json; no model or dataset loading.
"""

import json
from pathlib import Path

import polars as pl
from scipy import stats

# Settings that must agree for two runs to share a bootstrap and random arm. Seed
# concurrency and requested precision are left out; the precision actually used is kept.
MATCHED_SETTINGS = (
    "checkpoint",
    "dataset",
    "dataset_percent",
    "model_seeds",
    "uq_metric",
    "k",
    "max_pool_percent",
    "bootstrap_epochs",
    "update_passes",
    "replay_ratio",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_length",
    "effective_precision",
)
SETTING_DEFAULTS = {"dataset": "pet", "dataset_percent": 100.0}


def variant_name(config: dict) -> str:
    """Name a run's uncertainty selection; runs saved before ADR 0009 are top-K."""
    parts = []
    if config.get("gumbel_noise"):
        parts.append("Gumbel")
    if config.get("max_per_word_form") is not None:
        parts.append(f"cap {config['max_per_word_form']}")
    return " + ".join(parts) or "top-K"


def load_runs(results_root: Path) -> pl.DataFrame:
    """One row per completed paired run anywhere under ``results_root``."""
    rows = []
    for config_path in sorted(results_root.rglob("config.json")):
        run = config_path.parent
        if not (run / "results.csv").is_file() or not (run / "selections.json").is_file():
            continue
        config = json.loads(config_path.read_text())
        if config.get("random_only") or "uq_metric" not in config:
            continue
        settings = {key: config.get(key, SETTING_DEFAULTS.get(key)) for key in MATCHED_SETTINGS}
        rows.append(
            {
                "run": str(run.relative_to(results_root)),
                "variant": variant_name(config),
                "checkpoint": config["checkpoint"],
                "uq_metric": config["uq_metric"],
                "seeds": len(config["model_seeds"]),
                "settings": json.dumps(settings, sort_keys=True),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "run": pl.String,
            "variant": pl.String,
            "checkpoint": pl.String,
            "uq_metric": pl.String,
            "seeds": pl.Int64,
            "settings": pl.String,
        },
    )


def matched_variants(runs: pl.DataFrame) -> pl.DataFrame:
    """Pair every non-top-K run with a top-K run of identical settings."""
    baselines = (
        runs.filter(pl.col("variant") == "top-K")
        .group_by("settings")
        .agg(pl.col("run").sort().first().alias("baseline_run"))
    )
    return (
        runs.filter(pl.col("variant") != "top-K")
        .join(baselines, on="settings", how="inner")
        .select("variant", "checkpoint", "uq_metric", "seeds", "baseline_run", "run")
        .sort("checkpoint", "uq_metric", "variant", "run")
    )


def paired_curves(baseline_run: Path, variant_run: Path) -> pl.DataFrame:
    """Entity F1 per seed and round for random, top-K, and the variant."""

    def arms(run):
        return pl.read_csv(run / "results.csv").select(
            "seed", "round", "percent_acquired", "arm", "entity_f1"
        )

    baseline, variant = arms(baseline_run), arms(variant_run)
    keys = ["seed", "round", "percent_acquired"]
    return (
        baseline.filter(pl.col("arm") == "random")
        .select(*keys, random=pl.col("entity_f1"))
        .join(
            baseline.filter(pl.col("arm") == "uncertainty").select(
                *keys, top_k=pl.col("entity_f1")
            ),
            on=keys,
            validate="1:1",
        )
        .join(
            variant.filter(pl.col("arm") == "uncertainty").select(
                *keys, variant=pl.col("entity_f1")
            ),
            on=keys,
            validate="1:1",
        )
        .join(
            variant.filter(pl.col("arm") == "random").select(
                *keys, variant_random=pl.col("entity_f1")
            ),
            on=keys,
            validate="1:1",
        )
        .sort("seed", "round")
    )


def _normalized_auc(frame: pl.DataFrame, column: str) -> float:
    x = frame["percent_acquired"].to_list()
    y = frame[column].to_list()
    area = sum(
        (x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in zip(x, x[1:], y, y[1:], strict=False)
    )
    return area / (x[-1] - x[0])


def seed_gaps(curves: pl.DataFrame) -> pl.DataFrame:
    """Per-seed gap AUC and final gap, in entity-F1 points (0-1).

    ``random_rerun`` compares the two runs' random arms, which draw identical words; any
    gap there is training nondeterminism and sets the noise floor for the others.
    """
    gaps = curves.with_columns(
        variant_minus_top_k=pl.col("variant") - pl.col("top_k"),
        top_k_minus_random=pl.col("top_k") - pl.col("random"),
        variant_minus_random=pl.col("variant") - pl.col("random"),
        random_rerun=pl.col("variant_random") - pl.col("random"),
    )
    columns = ("variant_minus_top_k", "top_k_minus_random", "variant_minus_random", "random_rerun")
    rows = []
    for (seed,), frame in gaps.group_by("seed", maintain_order=True):
        rows.append(
            {
                "seed": seed,
                **{f"{column}_auc": _normalized_auc(frame, column) for column in columns},
                **{f"{column}_final": frame[column][-1] for column in columns},
            }
        )
    return pl.DataFrame(rows).sort("seed")


def gap_tests(gaps: pl.DataFrame) -> pl.DataFrame:
    """Seed mean, 95% t interval, two-sided p-value, and wins for every gap column."""
    rows = []
    for column in gaps.columns:
        if column == "seed":
            continue
        values = gaps[column].to_list()
        test = stats.ttest_1samp(values, 0)
        interval = test.confidence_interval(0.95)
        rows.append(
            {
                "gap": column,
                "mean (pp)": 100 * sum(values) / len(values),
                "95% CI lower (pp)": 100 * interval.low,
                "95% CI upper (pp)": 100 * interval.high,
                "p-value": float(test.pvalue),
                "seeds above 0": f"{sum(value > 0 for value in values)}/{len(values)}",
            }
        )
    return pl.DataFrame(rows)


def selection_redundancy(run: Path, label: str) -> pl.DataFrame:
    """Per seed and round: repeated word forms, O share, and words per sentence."""
    selections = pl.DataFrame(
        [row for row in json.loads((run / "selections.json").read_text())],
        infer_schema_length=None,
    )
    return (
        selections.with_columns(
            form=pl.col("token").str.to_lowercase(),
            selection=pl.when(pl.col("arm") == "random")
            .then(pl.lit("random"))
            .otherwise(pl.lit(label)),
        )
        .group_by("selection", "seed", "round")
        .agg(
            repeated_share=1 - pl.col("form").n_unique() / pl.len(),
            o_share=(pl.col("label") == "O").mean(),
            words_per_sentence=pl.len() / pl.col("pool_idx").n_unique(),
        )
        .sort("selection", "seed", "round")
    )


def round_jitter(curves: pl.DataFrame) -> pl.DataFrame:
    """Mean absolute round-to-round F1 change and share of drops beyond 2 points."""
    columns = ("random", "top_k", "variant")
    steps = curves.sort("seed", "round").with_columns(
        pl.col(column).diff().over("seed").alias(column) for column in columns
    )
    steps = steps.filter(pl.col("round") > 0)
    return pl.DataFrame(
        [
            {
                "arm": column,
                "mean |step| (pp)": 100 * steps[column].abs().mean(),
                "rounds dropping > 2 pp": (steps[column] < -0.02).mean(),
                "worst drop (pp)": 100 * steps[column].min(),
            }
            for column in columns
        ]
    )
