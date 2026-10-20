"""Load the paper rerun's saved results across datasets and models; nothing is trained.

Every loader reads `results/<stage>/<sweep>/` and keys rows by dataset and model, taken
from each sweep's plan, so sweep folder names never need parsing. Loaders make the
dataset, model, and acquisition columns Enums, so sorting on them follows the paper's
order; derived tables keep their inputs' types.
"""

import json
import statistics
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from seqeval.metrics.sequence_labeling import get_entities

from uq_pet import data_prep
from uq_pet.active_learning import read_predictions
from uq_pet.token_model import UQ_METRICS, evaluate_predictions

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
ACQUISITIONS = ["Top-K", "Gumbel noise"]
STAGES = {
    "random_baseline_search": "Random-baseline tuning",
    "best_uq": "Best UQ",
    "supervised": "Supervised baseline",
    "random_search": "30-config random sweep",
}
SETTING = ["dataset", "model", "acquisition", "uq_metric"]
# One run: a setting's model seed in one arm.
RUN = [*SETTING, "seed", "arm"]
_ENUMS = {
    "dataset": pl.Enum(DATASET_ORDER),
    "model": pl.Enum(MODEL_ORDER),
    "acquisition": pl.Enum(ACQUISITIONS),
}


def _ordered(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast the dataset, model, and acquisition columns present to their Enums."""
    return frame.cast({column: enum for column, enum in _ENUMS.items() if column in frame.columns})


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
    """One row per sweep: stage, dataset, model, and its completed, failed, pending items.

    A supervised sweep is done only once its test runs have finished after tuning.
    """
    rows = []
    for stage, stage_name in STAGES.items():
        for sweep_dir, plan, summary, dataset, model in _sweeps(results_root / stage):
            items = (summary or {}).get("comparisons") or (summary or {}).get("trials") or []
            planned = len(items) or plan.get("num_configs", 0)
            counts = {
                status: sum(item["status"] == status for item in items)
                for status in ("complete", "failed")
            }
            test_runs = (summary or {}).get("test_runs") or []
            tested = stage != "supervised" or (
                bool(test_runs) and all(run["status"] == "complete" for run in test_runs)
            )
            rows.append(
                {
                    "stage": stage_name,
                    "dataset": dataset,
                    "model": model,
                    "variant": _acquisition(plan) if stage == "best_uq" else "",
                    "complete": counts["complete"],
                    "failed": counts["failed"],
                    "planned": planned,
                    "tested": tested,
                    "sweep": sweep_dir.name,
                }
            )
    return _ordered(pl.DataFrame(rows)).with_columns(
        ((pl.col("complete") == pl.col("planned")) & pl.col("tested")).alias("done")
    )


def comparison_curves(results_root: Path, stage: str = "best_uq") -> pl.DataFrame:
    """Every evaluation row of every completed comparison in a stage, keyed by its setting.

    Rows also carry the comparison's `config_id`, its sampled parameters, and its
    `run_dir`, so one read of the results serves every chapter and score.
    """
    frames = []
    for sweep_dir, plan, summary, dataset, model in _sweeps(results_root / stage):
        for comparison in (summary or {}).get("comparisons", []):
            if comparison["status"] != "complete":
                continue
            run_dir = sweep_dir / comparison["run_dir"]
            frames.append(
                pl.read_csv(run_dir / "results.csv").with_columns(
                    dataset=pl.lit(dataset),
                    model=pl.lit(model),
                    acquisition=pl.lit(_acquisition(plan)),
                    uq_metric=pl.lit(comparison["uq_metric"]),
                    config_id=pl.lit(comparison.get("config_id"), pl.Categorical),
                    run_dir=pl.lit(str(run_dir), pl.Categorical),
                    **{
                        name: pl.lit(value)
                        for name, value in comparison.get("parameters", {}).items()
                    },
                )
            )
    return _ordered(pl.concat(frames, how="diagonal_relaxed")) if frames else pl.DataFrame()


def arm_series(curves: pl.DataFrame) -> pl.DataFrame:
    """Label each row's `series`, "random" or its UQ metric, keeping one copy of the random arm.

    Each UQ metric's comparison saves the same random arm of its dataset, model, and
    acquisition, so those rows repeat once per metric; the first metric's copy stays.
    """
    shared = ["dataset", "model", "acquisition"]
    return curves.filter(
        (pl.col("arm") == "uncertainty")
        | (pl.col("uq_metric") == pl.col("uq_metric").min().over(shared))
    ).with_columns(
        series=pl.when(pl.col("arm") == "random")
        .then(pl.lit("random"))
        .otherwise(pl.col("uq_metric"))
    )


@cache
def _test_split(dataset: str) -> tuple[dict, ...]:
    (_, _, _, test), _ = data_prep.load_splits(dataset)
    return tuple(test)


def load_test_predictions(run_dir: Path) -> tuple[list[dict], pl.DataFrame, list]:
    """Return a run's test sentences with gold `tags`, results rows, and predicted tags.

    Predicted tags come one per results row. predictions.npz stores tags only, so the
    words come from rebuilding the run's test split from local data; its gold tags must
    equal the saved ones.
    """
    dataset = json.loads((run_dir / "config.json").read_text())["dataset"]
    test = _test_split(dataset)
    gold, predictions = read_predictions(run_dir)
    labels = data_prep.DATASET_TAGS[dataset]
    if gold != [[labels[tag] for tag in sentence["ner_tags"]] for sentence in test]:
        raise ValueError(f"{run_dir} was not scored on the rebuilt {dataset} test split")
    sentences = [sentence | {"tags": tags} for sentence, tags in zip(test, gold, strict=True)]
    return sentences, pl.read_csv(run_dir / "results.csv"), predictions


def match_entities(
    gold: list[str], predicted: list[str], matching: str
) -> tuple[list[tuple[int, int, str, bool]], list[tuple[int, int, str, bool]]]:
    """Return gold and predicted (first word, end word, type, matched) entities of a sentence.

    Entities are chunked and matched as the scores do: exact needs the same words and
    type; partial needs the same type and one shared word, each gold entity matching
    once, greedily in predicted order.
    """
    unmatched = [(start, end + 1, kind) for kind, start, end in get_entities(gold)]
    gold_entities = list(unmatched)
    predicted_entities = []
    for kind, start, end in get_entities(predicted):
        match = next(
            (
                entity
                for entity in unmatched
                if entity[2] == kind
                and (
                    entity[:2] == (start, end + 1)
                    if matching == "exact"
                    else entity[0] <= end and start < entity[1]
                )
            ),
            None,
        )
        if match is not None:
            unmatched.remove(match)
        predicted_entities.append((start, end + 1, kind, match is not None))
    return (
        [(*entity, entity not in unmatched) for entity in gold_entities],
        predicted_entities,
    )


def arm_scores(curves: pl.DataFrame, field: str, keys: list[str] = SETTING) -> pl.DataFrame:
    """Per `keys` group, model seed, and arm: normalized `field` AUC and the final score.

    The AUC is the trapezoid area over acquired-pool percentage divided by the span it
    covers, so it is in the same units as `field`.
    """
    x = pl.col("percent_acquired").sort_by("round")
    y = pl.col(field).sort_by("round")
    scores = curves.group_by(*keys, "seed", "arm").agg(
        auc=(x.diff() * (y + y.shift(1)) / 2).sum() / (x.last() - x.first()),
        final=y.last(),
        increasing=(pl.len() > 1) & (x.diff().drop_nulls() > 0).all(),
    )
    if not scores["increasing"].all():
        bad = scores.filter(~pl.col("increasing")).row(0, named=True)
        raise ValueError(f"{bad} needs increasing acquisition points")
    return scores.drop("increasing")


def seed_scores(curves: pl.DataFrame, field: str, keys: list[str] = SETTING) -> pl.DataFrame:
    """Per `keys` group and model seed: both arms' normalized AUC and final score, and their gaps.

    AUC is over acquired-pool percentage, as in the saved summaries, so the seed mean of
    `gap_auc` equals the summary's gap AUC. Groups without `field` in every row, such as
    runs saved before a score existed, are left out.
    """
    if field not in curves.columns:
        return pl.DataFrame()
    scores = arm_scores(curves.filter(pl.col(field).is_not_null().all().over(keys)), field, keys)

    def arm(name, prefix):
        return scores.filter(pl.col("arm") == name).select(
            *keys,
            "seed",
            pl.col("auc").alias(f"{prefix}_auc"),
            pl.col("final").alias(f"{prefix}_final"),
        )

    return (
        arm("random", "random")
        .join(arm("uncertainty", "uq"), on=[*keys, "seed"], nulls_equal=True)
        .with_columns(
            gap_auc=pl.col("uq_auc") - pl.col("random_auc"),
            final_gap=pl.col("uq_final") - pl.col("random_final"),
        )
        .sort(*keys, "seed")
    )


# The sampled parameters `comparison_curves` copies onto each row of the 30-config sweeps.
CONFIG_PARAMETERS = [
    "learning_rate",
    "k",
    "update_passes",
    "replay_ratio",
    "batch_size",
    "weight_decay",
]


def config_scores(curves: pl.DataFrame, field: str) -> pl.DataFrame:
    """Per sampled configuration and UQ metric: gaps and random-arm scores, seed means.

    `curves` comes from `comparison_curves(..., "random_search")`. Gaps are computed from
    the curves, as for best UQ, because the saved summaries hold micro F1 gaps only. The
    random arm is shared by the UQ metrics of one configuration, so its scores repeat.
    """
    parameters = [name for name in CONFIG_PARAMETERS if name in curves.columns]
    keys = [*SETTING, "config_id", *parameters]
    seeds = seed_scores(curves, field, keys)
    if seeds.is_empty():
        return seeds
    return seeds.group_by(keys, maintain_order=True).agg(
        pl.col("gap_auc", "final_gap", "random_auc", "random_final").mean()
    )


def gap_curves(curves: pl.DataFrame, field: str) -> pl.DataFrame:
    """UQ-minus-random `field` per setting, model seed, and round."""
    return (
        curves.select(*RUN, "round", "percent_acquired", field)
        .pivot(on="arm", index=[*SETTING, "seed", "round", "percent_acquired"], values=field)
        .with_columns(gap=pl.col("uncertainty") - pl.col("random"))
    )


def score_aggs(field: str) -> list[pl.Expr]:
    """Mean `f1` and `sd` of `field`, its precision and recall means, and the `runs` count.

    Precision and recall are saved for micro F1 only, so they are null for macro F1.
    """
    prefix = field.removesuffix("_f1")
    return [
        pl.col(field).mean().alias("f1"),
        pl.col(field).std().alias("sd"),
        *(
            pl.lit(None, pl.Float64).alias(measure)
            if "macro" in field
            else pl.col(f"{prefix}_{measure}").mean().alias(measure)
            for measure in ("precision", "recall")
        ),
        pl.len().cast(pl.UInt32).alias("runs"),
    ]


def final_scores(curves: pl.DataFrame, field: str) -> pl.DataFrame:
    """Final-round scores of each series over model seeds, per dataset, model, acquisition."""
    keys = ["dataset", "model", "acquisition", "series"]
    return (
        arm_series(curves)
        .filter(pl.col("round") == pl.col("round").max().over(RUN))
        .group_by(keys)
        .agg(*score_aggs(field))
        .sort(keys)
    )


def first_round_reaching(curves: pl.DataFrame, field: str, targets: pl.DataFrame) -> pl.DataFrame:
    """Pool percentage at which each run's `field` first reaches its target, else null.

    `targets` holds a `target` per its other columns, all of them curve columns. Crossings
    are observed rounds, not interpolated; a later round can dip below again. A `series`
    column from `arm_series` is kept.
    """
    keys = [column for column in targets.columns if column != "target"]
    runs = RUN + ["series"] * ("series" in curves.columns)
    joined = curves.join(targets, on=keys)
    reached = (
        joined.filter(pl.col(field) >= pl.col("target"))
        .group_by(runs)
        .agg(pl.col("percent_acquired").min().alias("pool"))
    )
    return joined.select(runs).unique().join(reached, on=runs, how="left").sort(runs)


def median_pool(reach: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Median `first_round_reaching` pool per group, and how many runs reach the target.

    Runs that never reach it rank beyond the pool, so the median is null once half or
    more of the group's runs miss.
    """
    return (
        reach.group_by(by)
        .agg(
            pl.col("pool").fill_null(float("inf")).median(),
            pl.col("pool").is_not_null().sum().alias("reached"),
            pl.len().alias("runs"),
        )
        .with_columns(pool=pl.when(pl.col("pool").is_finite()).then(pl.col("pool")))
        .sort(by)
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
    return _ordered(pl.concat(frames, how="diagonal_relaxed")) if frames else pl.DataFrame()


def type_name(kind: str) -> str:
    """A medical type's display name ("T03" -> "Activity"); other types stay as they are."""
    return data_prep.MEDICAL_TYPE_NAMES.get(kind, kind)


def tag_name(tag: str) -> str:
    """A tag with its type's display name ("B-T03" -> "B-Activity")."""
    prefix, _, kind = tag.partition("-")
    return f"{prefix}-{type_name(kind)}" if kind else tag


def read_llm_predictions(predictions_file: Path, dataset: str) -> list[list[str]]:
    """Predicted tags of an LLM run's test-predictions.jsonl, one list per test sentence.

    The file lists the test split's sentences in order; its words must match the rebuilt
    split. Mention types outside the dataset's tags are kept, so they count as wrong.
    """
    rows = [json.loads(line) for line in predictions_file.read_text().splitlines()]
    expected = [(s["document_name"], s["sentence_id"], s["tokens"]) for s in _test_split(dataset)]
    if [(r["document name"], r["sentence-ID"], r["tokens"]) for r in rows] != expected:
        raise ValueError(f"{predictions_file} does not follow the rebuilt {dataset} test split")
    return [row["ner-tags"] for row in rows]


def llm_test_scores(llm_root: Path, task: str = "md-test-sentences") -> pl.DataFrame:
    """Test scores of every finished llm-annotation-eval run of `task`, one row per run.

    Each run's exported test predictions are rescored with `evaluate_predictions`, so
    exact and partial, micro and macro F1 mean the same as for the token classifiers.
    """
    rows = []
    for manifest in sorted(llm_root.glob("*/run-*/run.json")):
        run = json.loads(manifest.read_text())
        settings = run["settings"]
        predictions_file = manifest.parent / task / "test-predictions.jsonl"
        if settings["task"] != task or not predictions_file.is_file():
            continue
        dataset = settings.get("dataset", "pet")
        labels = data_prep.DATASET_TAGS[dataset]
        gold = [[labels[tag] for tag in s["ner_tags"]] for s in _test_split(dataset)]
        effort = settings.get("reasoning_effort") or "default"
        name = run["config"].removeprefix(f"{dataset}-").split(f"-{task}")[0]
        rows.append(
            {
                "dataset": DATASETS.get(dataset, dataset),
                "config": name if effort == "none" else f"{name} · {effort}",
                "llm": name,
                "reasoning_effort": effort,
                "llm_model": settings["model"],
                "run": run["run"],
                "predictions_file": str(predictions_file),
                **evaluate_predictions(gold, read_llm_predictions(predictions_file, dataset)),
            }
        )
    return _ordered(pl.DataFrame(rows))


def tuning_winners(results_root: Path) -> pl.DataFrame:
    """The validation winner of each finished random-baseline tuning sweep and its tuned settings."""
    rows = []
    for sweep_dir, plan, summary, dataset, model in _sweeps(
        results_root / "random_baseline_search"
    ):
        if not (summary or {}).get("complete"):
            continue
        selection = json.loads((sweep_dir / "selection.json").read_text())
        best = json.loads((sweep_dir / "best_config.json").read_text())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "config_id": selection["config_id"],
                "validation_score": selection["validation_score"],
                **{name: best[name] for name in plan["search_space"]},
            }
        )
    frame = _ordered(pl.DataFrame(rows))
    return frame.sort("dataset", "model") if rows else frame


def dataset_sizes(curves: pl.DataFrame) -> pl.DataFrame:
    """Split sizes and scoreable pool words of each dataset in `comparison_curves`."""
    if curves.is_empty():
        return pl.DataFrame()
    rows = []
    for (dataset,), runs in curves.group_by("dataset"):
        config = json.loads((Path(runs["run_dir"][0]) / "config.json").read_text())
        rows.append(
            {
                "dataset": dataset,
                "bootstrap_sentences": config["seed_sentences"],
                "pool_sentences": config["pool_sentences"],
                "test_sentences": config["test_sentences"],
                "scoreable_pool_words": runs["scoreable_pool_tokens"].max(),
            }
        )
    return pl.DataFrame(rows).cast({"dataset": curves.schema["dataset"]}).sort("dataset")


def config_gap_ttests(comparisons: pl.DataFrame, gap: str = "gap_auc") -> pl.DataFrame:
    """One-sample t-test of a gap column against zero across a sweep's sampled configurations.

    One row per dataset, model, and UQ metric with at least two configurations; Holm and
    Bonferroni adjust across the three planned UQ metrics within each sweep.
    """
    rows = []
    for (dataset, model, metric), group in comparisons.group_by(
        "dataset", "model", "uq_metric", maintain_order=True
    ):
        gaps = group[gap].to_list()
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
                "mean_gap_pp": 100 * statistics.fmean(gaps),
                "ci_low_pp": 100 * ci.low,
                "ci_high_pp": 100 * ci.high,
                "configs_favouring_uq": sum(gap > 0 for gap in gaps),
                "p_value": float(test.pvalue),
            }
        )
    if not rows:
        return pl.DataFrame()
    planned = len(UQ_METRICS)
    keys = ["dataset", "model", "uq_metric"]
    return (
        pl.DataFrame(rows)
        .cast({column: comparisons.schema[column] for column in keys})
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


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and two-sided 95% t-interval of `values`; one value gives a zero-width interval."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = stats.t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / len(values) ** 0.5
    return mean, mean - half, mean + half


def floor_and_ceiling(curves: pl.DataFrame, supervised: pl.DataFrame, field: str) -> pl.DataFrame:
    """Per dataset and model: the round-0 test score (floor) and the supervised one (ceiling).

    Round 0 is the bootstrap model both arms start from, so the floor averages the random
    arm's round-0 rows over seeds and sweeps. The ceiling averages the supervised test
    runs' seeds.
    """
    floor = (
        curves.filter((pl.col("round") == 0) & (pl.col("arm") == "random"))
        .group_by("dataset", "model")
        .agg(pl.col(field).mean().alias("floor"))
    )
    ceiling = supervised.group_by("dataset", "model").agg(
        pl.col(field).mean().alias("ceiling"),
        pl.col(field).std().alias("ceiling_sd"),
        pl.len().alias("ceiling_seeds"),
    )
    return floor.join(ceiling, on=["dataset", "model"], how="full", coalesce=True).sort(
        "dataset", "model"
    )


def dataset_bounds(bounds: pl.DataFrame, supervised: pl.DataFrame, field: str) -> pl.DataFrame:
    """Per dataset: the mean floor over models and the ceiling with its 95% t-interval.

    `bounds` comes from `floor_and_ceiling`. The interval runs over every supervised test
    run of the dataset (models × seeds), so it covers model choice as well as seeds.
    """
    rows = []
    for (dataset,), group in bounds.group_by("dataset", maintain_order=True):
        scores = supervised.filter(pl.col("dataset") == dataset)[field].drop_nulls().to_list()
        ceiling, low, high = mean_interval(scores) if scores else (None, None, None)
        floor = group["floor"].mean()
        rows.append(
            {
                "dataset": dataset,
                "floor": floor,
                "ceiling": ceiling,
                "ceiling_low": low,
                "ceiling_high": high,
                "floor_to_ceiling": None if ceiling is None or floor is None else ceiling - floor,
                "models": group.drop_nulls("ceiling").height,
            }
        )
    schema = {"dataset": bounds.schema["dataset"], "models": pl.Int64} | {
        column: pl.Float64
        for column in ("floor", "ceiling", "ceiling_low", "ceiling_high", "floor_to_ceiling")
    }
    return pl.DataFrame(rows, schema=schema).sort("dataset")


def scores_at_budgets(curves: pl.DataFrame, field: str, budgets: list[float]) -> pl.DataFrame:
    """Each run's `field` at the given pool percentages, linearly interpolated between rounds.

    Models acquire different K per round, so their rounds land on different pool
    percentages; interpolating puts every run on the same budgets. One row per run and
    budget.
    """
    rows = []
    for key, run in curves.group_by(RUN, maintain_order=True):
        run = run.sort("round")
        scores = np.interp(budgets, run["percent_acquired"].to_numpy(), run[field].to_numpy())
        rows += [
            dict(zip(RUN, key, strict=True)) | {"budget": float(budget), "score": float(score)}
            for budget, score in zip(budgets, scores, strict=True)
        ]
    return pl.DataFrame(rows).cast({column: curves.schema[column] for column in RUN})


def headroom_summary(
    at_budgets: pl.DataFrame, ceilings: pl.DataFrame, by: list[str]
) -> pl.DataFrame:
    """How far each arm is from the ceiling at each budget, and how much of that UQ closes.

    `at_budgets` comes from `scores_at_budgets`; `ceilings` has a `ceiling` per dataset and
    model. Each (model, seed) pairs its UQ and random scores, so the gap interval is a
    95% t-interval over those pairs. Headroom is the mean ceiling minus the mean random
    score; the share closed is the mean gap over the headroom, left empty when nothing
    is left to gain.
    """
    paired = (
        at_budgets.pivot(on="arm", index=[*SETTING, "seed", "budget"], values="score")
        .rename({"uncertainty": "uq"})
        .join(ceilings.select("dataset", "model", "ceiling"), on=["dataset", "model"])
        .with_columns(gap=pl.col("uq") - pl.col("random"))
    )
    rows = []
    for key, group in paired.group_by([*by, "budget"], maintain_order=True):
        gap, low, high = mean_interval(group["gap"].to_list())
        random, ceiling = group["random"].mean(), group["ceiling"].mean()
        headroom = ceiling - random
        rows.append(
            dict(zip([*by, "budget"], key, strict=True))
            | {
                "random": random,
                "uq": group["uq"].mean(),
                "gap": gap,
                "gap_low": low,
                "gap_high": high,
                "ceiling": ceiling,
                "headroom": headroom,
                "share_closed": gap / headroom if headroom > 0 else None,
                "pairs": group.height,
            }
        )
    return (
        pl.DataFrame(rows)
        .cast({column: at_budgets.schema[column] for column in by})
        .sort([*by, "budget"])
    )
