"""Load the paper rerun's saved results across datasets and models; nothing is trained.

Every loader reads `results/<stage>/<sweep>/` and keys rows by dataset and model, taken
from each sweep's plan, so sweep folder names never need parsing.
"""

import json
from pathlib import Path

import polars as pl

from utils.oracle_analysis import curve_auc

MODELS = {
    "distilbert-base-cased": "DistilBERT",
    "bert-base-cased": "BERT",
    "roberta-base": "RoBERTa",
    "microsoft/deberta-v3-base": "DeBERTa-v3",
    "answerdotai/ModernBERT-base": "ModernBERT",
}
MODEL_ORDER = list(MODELS.values())
DATASETS = {"pet": "PET", "quishpi": "Quishpi", "medical": "Medical"}
DATASET_ORDER = list(DATASETS.values())
STAGES = {
    "random_baseline_search": "Random-baseline tuning",
    "best_uq": "Best UQ",
    "supervised": "Supervised baseline",
    "random_search": "30-config random sweep",
}
UQ_METRICS = ["entropy", "least_confidence", "margin"]


def _sweeps(stage_root: Path):
    """Yield (sweep_dir, plan, summary or None, dataset, model) for each planned sweep."""
    for plan_path in sorted(stage_root.glob("*/plan.json")):
        plan = json.loads(plan_path.read_text())
        fixed = plan.get("fixed_config", {})
        summary_path = plan_path.with_name("summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        yield (
            plan_path.parent,
            plan,
            summary,
            DATASETS.get(fixed.get("dataset", "pet"), fixed.get("dataset", "pet")),
            MODELS.get(fixed["checkpoint"], fixed["checkpoint"]),
        )


def _acquisition(plan: dict) -> str:
    return "Gumbel noise" if plan["fixed_config"].get("gumbel_noise") else "Top-K"


def sweep_status(results_root: Path) -> pl.DataFrame:
    """One row per sweep: stage, dataset, model, and its completed, failed, pending items."""
    rows = []
    for stage, stage_name in STAGES.items():
        for sweep_dir, plan, summary, dataset, model in _sweeps(results_root / stage):
            items = (summary or {}).get("comparisons") or (summary or {}).get("trials") or []
            planned = len(items) or plan.get("num_configs", 0)
            counts = {
                status: sum(item["status"] == status for item in items)
                for status in ("complete", "failed")
            }
            rows.append(
                {
                    "stage": stage_name,
                    "dataset": dataset,
                    "model": model,
                    "variant": _acquisition(plan) if stage == "best_uq" else "",
                    "complete": counts["complete"],
                    "failed": counts["failed"],
                    "planned": planned,
                    "sweep": sweep_dir.name,
                }
            )
    return pl.DataFrame(rows).with_columns((pl.col("complete") == pl.col("planned")).alias("done"))


def best_uq_curves(results_root: Path) -> pl.DataFrame:
    """Every evaluation row of every completed best-UQ comparison, keyed by its setting."""
    frames = []
    for sweep_dir, plan, summary, dataset, model in _sweeps(results_root / "best_uq"):
        for comparison in (summary or {}).get("comparisons", []):
            if comparison["status"] != "complete":
                continue
            frames.append(
                pl.read_csv(sweep_dir / comparison["run_dir"] / "results.csv").with_columns(
                    dataset=pl.lit(dataset),
                    model=pl.lit(model),
                    acquisition=pl.lit(_acquisition(plan)),
                    uq_metric=pl.lit(comparison["uq_metric"]),
                )
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


SETTING = ["dataset", "model", "acquisition", "uq_metric"]


def seed_scores(curves: pl.DataFrame, field: str) -> pl.DataFrame:
    """Per setting and model seed: both arms' normalized AUC and final score, and their gaps.

    AUC is over acquired-pool percentage, as in the saved summaries, so the seed mean of
    `gap_auc` equals the summary's gap AUC.
    """
    rows = []
    for key, runs in curves.group_by(SETTING, maintain_order=True):
        aucs = curve_auc(runs, field)
        finals = {
            (seed, arm): score
            for seed, arm, score in runs.filter(
                pl.col("round") == pl.col("round").max().over("seed", "arm")
            )
            .select("seed", "arm", field)
            .iter_rows()
        }
        for seed in sorted({seed for seed, _ in aucs}):
            random_final = finals[seed, "random"]
            uq_final = finals[seed, "uncertainty"]
            rows.append(
                dict(zip(SETTING, key, strict=True))
                | {
                    "seed": seed,
                    "random_auc": aucs[seed, "random"],
                    "uq_auc": aucs[seed, "uncertainty"],
                    "gap_auc": aucs[seed, "uncertainty"] - aucs[seed, "random"],
                    "random_final": random_final,
                    "uq_final": uq_final,
                    "final_gap": uq_final - random_final,
                }
            )
    return pl.DataFrame(rows)


def pool_to_reach(curves: pl.DataFrame, field: str, fraction: float) -> pl.DataFrame:
    """Pool percentage at which each arm first reaches `fraction` of random's final score.

    The target is per seed: `fraction` times that seed's random-arm score after the
    whole pool is acquired. Crossings are observed rounds, not interpolated; an arm that
    never reaches the target gets a null.
    """
    final_random = curves.filter(
        (pl.col("arm") == "random")
        & (pl.col("round") == pl.col("round").max().over(*SETTING, "seed"))
    ).select(*SETTING, "seed", (fraction * pl.col(field)).alias("target"))
    return (
        curves.join(final_random, on=[*SETTING, "seed"])
        .filter(pl.col(field) >= pl.col("target"))
        .group_by(*SETTING, "seed", "arm")
        .agg(pl.col("percent_acquired").min())
        .pivot(on="arm", index=[*SETTING, "seed"], values="percent_acquired")
        .rename({"random": "random_pool", "uncertainty": "uq_pool"})
    )


def supervised_scores(results_root: Path) -> pl.DataFrame:
    """Test scores of every model seed in each supervised sweep, with its winning settings."""
    frames = []
    for sweep_dir, _, summary, dataset, model in _sweeps(results_root / "supervised"):
        if summary is None:
            continue
        best = json.loads((sweep_dir / "best_config.json").read_text())
        for marker in sorted(sweep_dir.glob("test/*/completed.json")):
            run_dir = sweep_dir / json.loads(marker.read_text())["run_dir"]
            frames.append(
                pl.read_csv(run_dir / "results.csv").with_columns(
                    dataset=pl.lit(dataset),
                    model=pl.lit(model),
                    epochs=pl.lit(best["epochs"]),
                    learning_rate=pl.lit(best["learning_rate"]),
                    batch_size=pl.lit(best["batch_size"]),
                    weight_decay=pl.lit(best["weight_decay"]),
                    validation_score=pl.lit(summary["selection"]["validation_score"]),
                )
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def tuning_trials(results_root: Path) -> pl.DataFrame:
    """Every completed random-baseline tuning trial with its validation score and winner flag.

    The winner is the highest score, ties broken by ascending configuration ID, among
    completed trials; it is final only when the sweep is complete.
    """
    rows = []
    for _, _, summary, dataset, model in _sweeps(results_root / "random_baseline_search"):
        trials = [
            trial for trial in (summary or {}).get("trials", []) if trial["status"] == "complete"
        ]
        if not trials:
            continue
        winner = min(trials, key=lambda trial: (-trial["score"], trial["config_id"]))["config_id"]
        for trial in trials:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "config_id": trial["config_id"],
                    "validation_auc": trial["score"],
                    "winner": trial["config_id"] == winner,
                    "sweep_complete": bool(summary.get("complete")),
                    **trial["parameters"],
                }
            )
    return pl.DataFrame(rows)


def best_uq_settings(results_root: Path) -> pl.DataFrame:
    """The frozen hyperparameters each best-UQ sweep ran with, one row per model and dataset."""
    rows = {}
    for _, plan, _, dataset, model in _sweeps(results_root / "best_uq"):
        rows[dataset, model] = {
            "dataset": dataset,
            "model": model,
            **plan["configurations"][0]["parameters"],
        }
    return pl.DataFrame(list(rows.values()))


def random_sweep_comparisons(results_root: Path, field: str) -> pl.DataFrame:
    """Per configuration and UQ metric of the 30-config sweeps: gap AUC and random-arm AUC.

    The random arm is shared by the UQ metrics of one configuration, so its AUC repeats.
    """
    rows = []
    for sweep_dir, _, summary, dataset, model in _sweeps(results_root / "random_search"):
        for comparison in (summary or {}).get("comparisons", []):
            if comparison["status"] != "complete":
                continue
            gap = comparison.get(f"mean_test_{field}_gap_auc")
            if gap is None:
                continue
            aucs = curve_auc(pl.read_csv(sweep_dir / comparison["run_dir"] / "results.csv"), field)
            random_aucs = [auc for (_, arm), auc in aucs.items() if arm == "random"]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "gap_auc": gap,
                    "random_auc": sum(random_aucs) / len(random_aucs),
                    **comparison["parameters"],
                }
            )
    return pl.DataFrame(rows)


def dataset_sizes(results_root: Path) -> pl.DataFrame:
    """Split sizes and scoreable pool words of each dataset with a best-UQ run."""
    rows = {}
    for sweep_dir, _, summary, dataset, _ in _sweeps(results_root / "best_uq"):
        comparisons = [
            c for c in (summary or {}).get("comparisons", []) if c["status"] == "complete"
        ]
        if dataset in rows or not comparisons:
            continue
        run_dir = sweep_dir / comparisons[0]["run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        results = pl.read_csv(run_dir / "results.csv")
        rows[dataset] = {
            "dataset": dataset,
            "bootstrap_sentences": config["seed_sentences"],
            "pool_sentences": config["pool_sentences"],
            "test_sentences": config["test_sentences"],
            "scoreable_pool_words": results["scoreable_pool_tokens"].max(),
        }
    return pl.DataFrame(list(rows.values()))


def gap_curves(curves: pl.DataFrame, field: str) -> pl.DataFrame:
    """UQ-minus-random `field` per setting, model seed, and round."""
    return (
        curves.select(*SETTING, "seed", "round", "percent_acquired", "arm", field)
        .pivot(on="arm", index=[*SETTING, "seed", "round", "percent_acquired"], values=field)
        .with_columns(gap=pl.col("uncertainty") - pl.col("random"))
    )


def config_gap_ttests(comparisons: pl.DataFrame) -> pl.DataFrame:
    """One-sample t-test of gap AUC against zero across a sweep's sampled configurations.

    One row per dataset, model, and UQ metric with at least two configurations; Holm and
    Bonferroni adjust across the three planned UQ metrics within each sweep.
    """
    from scipy import stats

    rows = []
    for (dataset, model, metric), group in comparisons.group_by(
        "dataset", "model", "uq_metric", maintain_order=True
    ):
        gaps = group["gap_auc"].to_list()
        if len(gaps) < 2:
            continue
        test = stats.ttest_1samp(gaps, 0)
        ci = test.confidence_interval(0.95)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "uq_metric": metric,
                "configurations": len(gaps),
                "mean_gap_pp": 100 * sum(gaps) / len(gaps),
                "ci_low_pp": 100 * ci.low,
                "ci_high_pp": 100 * ci.high,
                "configs_favouring_uq": sum(gap > 0 for gap in gaps),
                "p_value": float(test.pvalue),
            }
        )
    if not rows:
        return pl.DataFrame()
    planned = len(UQ_METRICS)
    return (
        pl.DataFrame(rows)
        .with_columns(
            rank=pl.col("p_value").rank("ordinal").over("dataset", "model").cast(pl.Int64),
            bonferroni_p=(planned * pl.col("p_value")).clip(upper_bound=1.0),
        )
        .sort("dataset", "model", "p_value")
        .with_columns(
            holm_p=((planned - pl.col("rank") + 1) * pl.col("p_value"))
            .cum_max()
            .over("dataset", "model")
            .clip(upper_bound=1.0)
        )
        .drop("rank")
    )
