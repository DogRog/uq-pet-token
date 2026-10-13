"""Load the paper rerun's saved results across datasets and models; nothing is trained.

Every loader reads `results/<stage>/<sweep>/` and keys rows by dataset and model, taken
from each sweep's plan, so sweep folder names never need parsing.
"""

import json
from functools import cache
from pathlib import Path

import polars as pl
from seqeval.metrics.sequence_labeling import get_entities

from uq_pet import data_prep
from uq_pet.active_learning import read_predictions
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
    return pl.DataFrame(rows).with_columns(
        ((pl.col("complete") == pl.col("planned")) & pl.col("tested")).alias("done")
    )


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


def best_uq_runs(results_root: Path) -> pl.DataFrame:
    """The run folder of every completed best-UQ comparison, keyed by its setting."""
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "acquisition": _acquisition(plan),
            "uq_metric": comparison["uq_metric"],
            "run_dir": str(sweep_dir / comparison["run_dir"]),
        }
        for sweep_dir, plan, summary, dataset, model in _sweeps(results_root / "best_uq")
        for comparison in (summary or {}).get("comparisons", [])
        if comparison["status"] == "complete"
    ]
    return pl.DataFrame(rows, schema=[*SETTING, "run_dir"])


@cache
def _test_split(dataset: str) -> tuple[dict, ...]:
    (_, _, _, test), _ = data_prep.load_splits(dataset)
    return tuple(test)


def load_test_predictions(run_dir: Path) -> tuple[list[dict], pl.DataFrame, list]:
    """Return a run's test sentences with gold `tags`, results rows, and predicted tags.

    Predicted tags come one per results row. predictions.npz stores tags only, so the words come from rebuilding the run's test
    split from local data; its gold tags must equal the saved ones.
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
    return pl.DataFrame(rows)


def random_sweep_comparisons(results_root: Path, field: str) -> pl.DataFrame:
    """Per configuration and UQ metric of the 30-config sweeps: gaps and random-arm scores.

    Each score comes as an AUC and a final-round value, averaged over model seeds. The
    random arm is shared by the UQ metrics of one configuration, so its scores repeat.
    """
    rows = []
    for sweep_dir, _, summary, dataset, model in _sweeps(results_root / "random_search"):
        for comparison in (summary or {}).get("comparisons", []):
            if comparison["status"] != "complete":
                continue
            gap = comparison.get(f"mean_test_{field}_gap_auc")
            if gap is None:
                continue
            results = pl.read_csv(sweep_dir / comparison["run_dir"] / "results.csv")
            aucs = curve_auc(results, field)
            random_aucs = [auc for (_, arm), auc in aucs.items() if arm == "random"]
            random_finals = results.filter(
                (pl.col("arm") == "random")
                & (pl.col("round") == pl.col("round").max().over("seed"))
            )[field]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "gap_auc": gap,
                    "final_gap": comparison[f"mean_final_test_{field}_gap"],
                    "random_auc": sum(random_aucs) / len(random_aucs),
                    "random_final": random_finals.mean(),
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


def config_gap_ttests(comparisons: pl.DataFrame, gap: str = "gap_auc") -> pl.DataFrame:
    """One-sample t-test of a gap column against zero across a sweep's sampled configurations.

    One row per dataset, model, and UQ metric with at least two configurations; Holm and
    Bonferroni adjust across the three planned UQ metrics within each sweep.
    """
    from scipy import stats

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


def dataset_words(datasets: list[str]) -> tuple[pl.DataFrame, dict[str, str]]:
    """Every word of each dataset in document order, with its gold tag and entity flag.

    All splits are included: bootstrap sentences, pool, and test split. Words are
    lowercased; tokens without a letter or digit (punctuation) are dropped. Datasets that
    cannot be loaded are reported in the returned errors instead of raising.
    """
    from uq_pet.data_prep import DATASET_TAGS, load_splits

    rows, errors = [], {}
    for name in datasets:
        try:
            (seed, pool_inputs, pool_gold, test), _ = load_splits(name)
        except Exception as error:  # noqa: BLE001 - a gated or missing dataset is reported
            errors[DATASETS.get(name, name)] = f"{type(error).__name__}: {error}"
            continue
        tags = DATASET_TAGS[name]
        sentences = [*seed, *test] + [
            {
                **pool,
                "ner_tags": [pool_gold[pool["pool_idx"], i] for i in range(len(pool["tokens"]))],
            }
            for pool in pool_inputs
        ]
        for sentence in sorted(sentences, key=lambda s: (s["document_name"], s["sentence_id"])):
            for word_idx, (word, tag) in enumerate(
                zip(sentence["tokens"], sentence["ner_tags"], strict=True)
            ):
                if not any(char.isalnum() for char in word):
                    continue
                rows.append(
                    {
                        "dataset": DATASETS.get(name, name),
                        "document": sentence["document_name"],
                        "sentence_id": sentence["sentence_id"],
                        "word_idx": word_idx,
                        "word": word.lower(),
                        "tag": tags[tag],
                    }
                )
    words = pl.DataFrame(rows).with_columns(entity=pl.col("tag") != "O")
    return words, errors


def sentence_ttr(words: pl.DataFrame) -> pl.DataFrame:
    """Type-token ratio and length of every sentence (punctuation dropped)."""
    return (
        words.group_by("dataset", "document", "sentence_id")
        .agg(pl.col("word").n_unique().alias("types"), pl.len().alias("words"))
        .with_columns(ttr=pl.col("types") / pl.col("words"))
    )


def window_ttr(words: pl.DataFrame, size: int, entity_only: bool = False) -> pl.DataFrame:
    """Type-token ratio of consecutive, non-overlapping windows of `size` words.

    Windows run over each dataset's text in document order; a final window shorter than
    `size` is dropped, so every value is comparable across datasets (mean segmental TTR).
    With `entity_only`, only words inside gold entities are kept before windowing.
    """
    kept = words.filter(pl.col("entity")) if entity_only else words
    return (
        kept.with_columns(window=(pl.int_range(pl.len()) // size).over("dataset"))
        .group_by("dataset", "window")
        .agg(pl.col("word").n_unique().alias("types"), pl.len().alias("words"))
        .filter(pl.col("words") == size)
        .with_columns(ttr=pl.col("types") / size)
    )
