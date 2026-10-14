"""Run sampled or fixed UQ comparisons, or tune random selection on one split.

Random tuning scores the validation holdout for the headline protocol, or the test split
for the appendix's test-tuned random oracle (ADR 0007).
"""

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from uq_pet import active_learning
from uq_pet.active_learning import run_metric_comparisons, write_run
from uq_pet.config import (
    ExperimentConfig,
    FixedComparisonConfig,
    RandomBaselineSearchConfig,
    RandomSearchConfig,
    omit_implicit_defaults,
)
from uq_pet.data_prep import DATASET_TAGS, load_splits, split_tuning_pool
from uq_pet.paths import PROJECT_ROOT
from uq_pet.sweeps import (
    completed_run,
    learning_rate_space,
    mark_complete,
    open_sweep,
    preserve_json,
    quiet_library_progress,
    record_failure,
    run_status,
    sample_learning_rate,
    write_json,
)
from uq_pet.token_model import UQ_METRICS, get_device, gold_tags, resolve_precision
from uq_pet.utils.wandb_logging import connect_wandb, wandb_comparison_logging
from uq_pet.utils.wandb_tuning import ensure_sweep, mark_winner, publish_trial

CONSOLE = Console()
WANDB_FIELDS = {"wandb_enabled", "wandb_project", "wandb_run_name"}
SEARCH_SPACE = {
    "k": (32, 64),
    "bootstrap_epochs": (10,),
    "update_passes": (1, 2, 4),
    "batch_size": (32, 64),
    "replay_ratio": (0, 1.0, 2.0),
    "weight_decay": (0.0, 0.01),
}


@contextmanager
def _quiet_search_output():
    """Suppress routine model output while retaining concise sweep progress."""
    quiet = active_learning.CONSOLE.quiet
    try:
        active_learning.CONSOLE.quiet = True
        with quiet_library_progress():
            yield
    finally:
        active_learning.CONSOLE.quiet = quiet


def mean_entity_f1_gap_auc(
    results: Sequence[Mapping[str, object]], field: str = "entity_f1"
) -> float:
    """Return mean normalized AUC of uncertainty-minus-random entity F1 across seeds."""
    rows_by_seed_round: dict[tuple[int, int], dict[str, Mapping[str, object]]] = {}
    for row in results:
        key = (int(row["seed"]), int(row["round"]))
        arm = str(row["arm"])
        if arm in rows_by_seed_round.setdefault(key, {}):
            raise ValueError(f"duplicate {arm} result for seed={key[0]}, round={key[1]}")
        rows_by_seed_round[key][arm] = row

    if not rows_by_seed_round:
        raise ValueError("cannot score an empty result set")

    points_by_seed: dict[int, list[tuple[float, float]]] = {}
    for (seed, round_idx), rows_by_arm in rows_by_seed_round.items():
        if set(rows_by_arm) != {"uncertainty", "random"}:
            raise ValueError(
                f"seed={seed}, round={round_idx} must contain uncertainty and random rows"
            )
        uncertainty = rows_by_arm["uncertainty"]
        random_row = rows_by_arm["random"]
        if uncertainty["percent_acquired"] != random_row["percent_acquired"]:
            raise ValueError(f"seed={seed}, round={round_idx} rows disagree on percent_acquired")
        points_by_seed.setdefault(seed, []).append(
            (
                float(uncertainty["percent_acquired"]),
                float(uncertainty[field]) - float(random_row[field]),
            )
        )

    seed_aucs = []
    for seed, points in points_by_seed.items():
        points.sort()
        if len(points) < 2 or points[-1][0] <= points[0][0]:
            raise ValueError(f"seed={seed} needs at least two increasing acquisition points")
        area = sum(
            (right_x - left_x) * (left_gap + right_gap) / 2
            for (left_x, left_gap), (right_x, right_gap) in zip(points, points[1:], strict=False)
        )
        seed_aucs.append(area / (points[-1][0] - points[0][0]))

    return sum(seed_aucs) / len(seed_aucs)


def sample_plan(config: RandomSearchConfig) -> dict:
    """Freeze supplied settings or sample a sweep before observing results."""
    if isinstance(config, FixedComparisonConfig):
        experiment = config.experiment_config().model_dump()
        return {
            "version": 1,
            "mode": "fixed_comparison",
            "shared_bootstrap_and_random": True,
            "evaluation_split": "test",
            "sampling": "none",
            "num_configs": 1,
            "fixed_config": omit_implicit_defaults(
                {
                    key: value
                    for key, value in experiment.items()
                    if key
                    not in {
                        "seed_workers",
                        "uq_metric",
                        "learning_rate",
                        *SEARCH_SPACE,
                        *WANDB_FIELDS,
                    }
                }
            ),
            "uq_metrics": config.uq_metrics,
            "configurations": [
                {
                    "config_id": "config_0000",
                    "parameters": {
                        key: experiment[key] for key in (*SEARCH_SPACE, "learning_rate")
                    },
                }
            ],
        }
    rng = random.Random(config.sampler_seed)
    sampled = []
    for _ in range(config.num_configs):
        parameters = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        parameters["learning_rate"] = sample_learning_rate(rng)
        sampled.append(parameters)
    # Worker scheduling can change on resume; scientific settings cannot.
    fixed = omit_implicit_defaults(
        config.experiment_config().model_dump(
            exclude={"seed_workers", "uq_metric", "learning_rate", *SEARCH_SPACE, *WANDB_FIELDS}
        )
    )
    return {
        "version": 4,
        "shared_bootstrap_and_random": True,
        "evaluation_split": "test",
        "sampling": "independent_categorical_and_log_uniform",
        "sampler_seed": config.sampler_seed,
        "num_configs": config.num_configs,
        "fixed_config": fixed,
        "search_space": {
            **{key: list(values) for key, values in SEARCH_SPACE.items()},
            "learning_rate": learning_rate_space(),
        },
        "uq_metrics": list(UQ_METRICS),
        "configurations": [
            {
                "config_id": f"config_{index:04d}",
                "parameters": parameters,
            }
            for index, parameters in enumerate(sampled)
        ],
    }


def best_trial_text(scores: Mapping[str, float]) -> str:
    """Describe the best completed tuning trial so far, with the winner's tie-break."""
    if not scores:
        return ""
    best = min(scores, key=lambda config_id: (-scores[config_id], config_id))
    return f" · best {best} {scores[best]:.4f}"


def tuning_plan(config: RandomBaselineSearchConfig) -> dict:
    """Lock random-only sampling, the scoring split, and objective before training."""
    sampled = sample_plan(config)
    plan = {
        "version": 3,
        "mode": "random_baseline_tuning",
        "sampling": sampled["sampling"],
        "sampler_seed": config.sampler_seed,
        "num_configs": config.num_configs,
        "search_space": sampled["search_space"],
        "configurations": sampled["configurations"],
        "fixed_config": sampled["fixed_config"],
    }
    if config.tuning_split == "validation":
        plan["validation_sentences"] = config.validation_sentences
        plan["validation_seed"] = config.validation_seed
    return {
        **plan,
        "objective": config.objective,
        "tie_break": "config_id_ascending",
        "tuning_evaluation_split": config.tuning_split,
    }


def winner_filename(plan: dict) -> str:
    """Name test-selected winners apart so they are never mistaken for headline winners."""
    if plan.get("tuning_evaluation_split") == "test":
        return "oracle_config.json"
    return "best_config.json"


def random_tuning_score(results: list[dict], objective: str) -> float:
    """Average random F1 across seeds, using final F1 or normalized curve AUC."""
    if not results or any(row["arm"] != "random" for row in results):
        raise ValueError("tuning requires nonempty random-only results")
    scores = []
    for seed in sorted({row["seed"] for row in results}):
        rows = sorted((row for row in results if row["seed"] == seed), key=lambda r: r["round"])
        if [row["round"] for row in rows] != list(range(rows[-1]["total_rounds"] + 1)):
            raise ValueError("tuning requires every round, including bootstrap and the final round")
        points = [(float(row["percent_acquired"]), float(row["entity_f1"])) for row in rows]
        if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
            raise ValueError("tuning scores must be finite")
        if len(points) < 2 or any(b[0] <= a[0] for a, b in zip(points, points[1:], strict=False)):
            raise ValueError("tuning requires increasing acquisition points")
        if objective in {"random_validation_final_entity_f1", "random_test_final_entity_f1"}:
            scores.append(points[-1][1])
        elif objective in {"random_validation_entity_f1_auc", "random_test_entity_f1_auc"}:
            area = sum(
                (b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:], strict=False)
            )
            scores.append(area / (points[-1][0] - points[0][0]))
        else:
            raise ValueError(f"unknown objective: {objective}")
    return sum(scores) / len(scores)


def scientific_plan(plan: dict) -> dict:
    """Ignore logging and last-bit rate roundoff when comparing saved scientific plans.

    The exact saved rates remain authoritative for execution.
    """
    return {
        **plan,
        "fixed_config": {
            key: value for key, value in plan["fixed_config"].items() if key not in WANDB_FIELDS
        },
        "configurations": [
            {
                **entry,
                "parameters": {
                    **entry["parameters"],
                    "learning_rate": float(format(entry["parameters"]["learning_rate"], ".14g")),
                },
            }
            for entry in plan["configurations"]
        ],
    }


def paired_summary(results: list[dict]) -> dict:
    """Describe a paired run, by exact and partial entity F1, without ranking it."""
    seeds = sorted({row["seed"] for row in results})
    summary = {}
    for field in ("entity_f1", "entity_partial_f1"):
        final_gaps = []
        for seed in seeds:
            rows = [row for row in results if row["seed"] == seed]
            last_round = max(row["round"] for row in rows)
            final = {row["arm"]: row[field] for row in rows if row["round"] == last_round}
            final_gaps.append(final["uncertainty"] - final["random"])
        summary[f"mean_test_{field}_gap_auc"] = mean_entity_f1_gap_auc(results, field)
        summary[f"mean_final_test_{field}_gap"] = sum(final_gaps) / len(seeds)
    return summary


def write_summary(root: Path, plan: dict) -> dict:
    """Include every planned comparison and expose incomplete sweep coverage."""
    if plan.get("mode") == "random_baseline_tuning":
        trials = [
            {**entry, **run_status(root, root / "tuning" / entry["config_id"])}
            for entry in plan["configurations"]
        ]
        summary = {
            "mode": plan["mode"],
            "objective": plan["objective"],
            "complete": all(row["status"] == "complete" for row in trials)
            and (root / winner_filename(plan)).is_file()
            and (root / "selection.json").is_file(),
            "trials": trials,
        }
        write_json(root / "summary.json", summary)
        return summary
    comparisons = []
    for entry in plan["configurations"]:
        for metric in plan["uq_metrics"]:
            slot = root / "runs" / entry["config_id"] / metric
            row = {
                "config_id": entry["config_id"],
                "uq_metric": metric,
                "parameters": entry["parameters"],
                **run_status(root, slot),
            }
            comparisons.append(row)
    by_metric = []
    for metric in plan["uq_metrics"]:
        rows = [
            row for row in comparisons if row["uq_metric"] == metric and row["status"] == "complete"
        ]
        gaps = [row["mean_test_entity_f1_gap_auc"] for row in rows]
        by_metric.append(
            {
                "uq_metric": metric,
                "planned_configurations": plan["num_configs"],
                "completed_configurations": len(rows),
                "mean_test_entity_f1_gap_auc": sum(gaps) / len(gaps) if gaps else None,
                "mean_final_test_entity_f1_gap": sum(
                    row["mean_final_test_entity_f1_gap"] for row in rows
                )
                / len(rows)
                if rows
                else None,
                "wins": sum(gap > 1e-12 for gap in gaps),
                "ties": sum(abs(gap) <= 1e-12 for gap in gaps),
                "losses": sum(gap < -1e-12 for gap in gaps),
                "win_fraction": sum(gap > 1e-12 for gap in gaps) / len(gaps) if gaps else None,
            }
        )
    summary = {
        "evaluation_split": "test",
        "complete": all(row["status"] == "complete" for row in comparisons),
        "by_metric": by_metric,
        "comparisons": comparisons,
    }
    write_json(root / "summary.json", summary)
    return summary


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("compare", "tune-random", "compare-fixed"),
        default=argparse.SUPPRESS,
        help="Compare sampled configurations on test (compare), compare one supplied configuration (compare-fixed), or tune random on its tuning_split, validation by default (tune-random).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without loading data or models."
    )
    parser.add_argument(
        "--publish-wandb-only",
        action="store_true",
        help="Create the W&B tuning sweep/chart and publish saved completed trials without training.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON file with sweep and experiment settings.")
    source.add_argument("--config-json", help="Inline JSON with sweep and experiment settings.")
    parser.add_argument(
        "--num-configs",
        type=int,
        default=argparse.SUPPRESS,
        help="Total fixed budget of distinct hyperparameter configurations.",
    )
    parser.add_argument("--sweep-name", default=argparse.SUPPRESS)
    parser.add_argument("--sweeps-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--sampler-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--uq-metrics", nargs="+", choices=UQ_METRICS, default=argparse.SUPPRESS)


def load_search_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> RandomSearchConfig:
    """Validate settings before creating output directories or loading models."""
    try:
        payload = args.config.read_text() if args.config is not None else args.config_json or "{}"
        settings = json.loads(payload)
        if not isinstance(settings, dict):
            raise ValueError("search configuration must be a JSON object")
        settings.update(
            {
                key: value
                for key, value in vars(args).items()
                if key not in {"config", "config_json", "dry_run", "publish_wandb_only"}
            }
        )
        config_class = (
            RandomBaselineSearchConfig
            if settings.get("mode") == "tune-random"
            else FixedComparisonConfig
            if settings.get("mode") == "compare-fixed"
            else RandomSearchConfig
        )
        config = config_class.model_validate(settings)
        sample_plan(config)
        return config
    except (OSError, ValueError) as error:
        parser.error(str(error))


def _same_scientific_plan(saved: dict, plan: dict) -> bool:
    return scientific_plan(saved) == scientific_plan(plan)


def _publish_tuning(config: RandomBaselineSearchConfig, plan: dict, root: Path) -> tuple:
    """Create the W&B tuning sweep and chart, then publish every saved trial and the winner."""
    publication = ensure_sweep(config, plan, root)
    for entry in plan["configurations"]:
        publish_trial(config, root, entry, publication, random_tuning_score)
    mark_winner(publication, root)
    CONSOLE.print(f"Tuning chart: {publication[1][publication[2]]['workspace_url']}")
    return publication


def _tuning_splits(
    config: RandomBaselineSearchConfig, root: Path, splits: tuple, identity: dict
) -> tuple:
    """Return the seed, pool, gold, and evaluation sentences that random tuning trains on."""
    seed_examples, pool_inputs, pool_gold, test_examples = splits
    if config.tuning_split == "test":
        # The appendix oracle tunes on the full pool and scores the test split (ADR 0007).
        preserve_json(
            root / "split.json",
            {
                **identity,
                "tuning_split": "test",
                "seed_sentences": len(seed_examples),
                "tuning_pool_sentences": len(pool_inputs),
                "test_sentences": len(test_examples),
            },
        )
        return splits
    tune_pool, tune_gold, tune_evaluation, manifest = split_tuning_pool(
        pool_inputs,
        pool_gold,
        validation_sentences=config.validation_sentences,
        validation_seed=config.validation_seed,
    )
    preserve_json(
        root / "split.json",
        {
            **manifest,
            **identity,
            "seed_sentences": len(seed_examples),
            "tuning_pool_sentences": len(tune_pool),
            "validation_sentences": len(tune_evaluation),
            "final_pool_sentences": len(pool_inputs),
            "test_sentences": len(test_examples),
        },
    )
    return seed_examples, tune_pool, tune_gold, tune_evaluation


class _SweepProgress:
    """The sweep's overall bar, labelled with the best tuning trial so far, and one per seed."""

    def __init__(
        self,
        display: Progress,
        label: str,
        seeds: Sequence[int],
        *,
        total: int,
        completed: int,
        trial_scores: dict[str, float],
    ):
        self.display = display
        self.label = label
        self.completed = completed
        self.trial_scores = trial_scores
        self.overall = display.add_task(
            f"{label}{best_trial_text(trial_scores)}", total=total, completed=completed
        )
        self.seeds = {
            seed: display.add_task(f"Seed {seed} · waiting / bootstrap", total=1) for seed in seeds
        }
        self.description = label
        self._base = completed
        self._fractions: dict[tuple[str, int], float] = {}

    def start(self, config_id: str, metrics: Sequence[str]) -> None:
        self.description = f"{self.label} · {config_id}{best_trial_text(self.trial_scores)}"
        self.display.update(self.overall, description=self.description)
        for seed, task in self.seeds.items():
            self.display.reset(task, total=1, description=f"Seed {seed} · waiting / bootstrap")
        self._fractions = {(metric, seed): 0.0 for metric in metrics for seed in self.seeds}
        self._base = self.completed

    def round_done(self, metric: str, latest: dict, arm: str) -> None:
        seed = latest["seed"]
        self.display.update(
            self.seeds[seed],
            total=latest["total_rounds"],
            completed=latest["round"],
            description=f"Seed {seed} · {arm} rounds",
        )
        self._fractions[metric, seed] = (latest["round"] + 1) / (latest["total_rounds"] + 1)
        self.display.update(
            self.overall,
            completed=self._base + sum(self._fractions.values()) / len(self.seeds),
        )

    def saving(self) -> None:
        self.display.update(self.overall, description=f"{self.description} · saving")

    def run_done(self, config_id: str, score: float | None) -> None:
        self.completed += 1
        if score is not None:
            self.trial_scores[config_id] = score
        self.display.update(
            self.overall,
            completed=self.completed,
            description=f"{self.label} · {config_id} done{best_trial_text(self.trial_scores)}",
        )


def _write_progress(slot: Path, rows: list[dict]) -> None:
    temporary = slot / "progress.tmp"
    pl.DataFrame(rows).write_csv(temporary)
    temporary.replace(slot / "progress.csv")


def _run_metadata(
    config: RandomSearchConfig,
    plan: dict,
    run_config: ExperimentConfig,
    config_id: str,
    splits: tuple,
    *,
    stage: str,
    precision: str,
) -> dict:
    """Describe one saved run: its resolved settings, sweep, and sentence counts."""
    tuning = config.mode == "tune-random"
    seed_examples, pool, _, evaluation = splits
    metadata = {
        **run_config.resolved_dict(),
        "effective_precision": precision,
        "shared_bootstrap_and_random": not tuning,
        "random_only": tuning,
        "evaluation_split": stage,
        "mode": "random_baseline_tuning" if tuning else plan.get("mode", "random_search"),
        "sweep_name": config.sweep_name,
        "config_id": config_id,
        "seed_sentences": len(seed_examples),
        "pool_sentences": len(pool),
        "evaluation_sentences": len(evaluation),
    }
    metadata[f"{stage}_sentences"] = len(evaluation)
    if tuning:
        metadata.pop("uq_metric", None)
        metadata["objective"] = config.objective
    return metadata


def _run_entry(
    config: RandomSearchConfig,
    plan: dict,
    root: Path,
    entry: dict,
    metrics: list[str],
    splits: tuple,
    *,
    stage: str,
    device,
    precision: str,
    bars: _SweepProgress,
    announce_project: bool,
) -> None:
    """Train one configuration for each pending metric and save every metric's run."""
    tuning = config.mode == "tune-random"
    config_id = entry["config_id"]
    slots = {
        metric: root / "tuning" / config_id if tuning else root / "runs" / config_id / metric
        for metric in metrics
    }
    for slot in slots.values():
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "progress.csv").unlink(missing_ok=True)
    experiment = ExperimentConfig.model_validate(
        {**config.experiment_config().model_dump(), **entry["parameters"]}
    )
    bars.start(config_id, metrics)

    def update_progress(metric, rows):
        _write_progress(slots[metric], rows)
        if log_evaluation is not None:
            log_evaluation(metric, rows)
        bars.round_done(metric, rows[-1], "random" if tuning else metric)

    try:
        with wandb_comparison_logging(
            config,
            experiment,
            config_id,
            precision,
            slots,
            console=CONSOLE,
            evaluation_split=stage,
            random_only=tuning,
            announce_project=announce_project,
        ) as log_evaluation:
            settings = experiment.active_learning_kwargs()
            settings.pop("uq_metric")
            comparisons = run_metric_comparisons(
                *splits,
                **{**settings, "precision": precision},
                uq_metrics=metrics,
                random_only=tuning,
                device=device,
                progress_callback=update_progress,
            )
            bars.saving()
            for metric in metrics:
                results, selections, predictions = comparisons[metric]
                stats = (
                    {"score": random_tuning_score(results, config.objective)}
                    if tuning
                    else paired_summary(results)
                )
                run_config = experiment.model_copy(update={"uq_metric": metric})
                run_dir = write_run(
                    _run_metadata(
                        config,
                        plan,
                        run_config,
                        config_id,
                        splits,
                        stage=stage,
                        precision=precision,
                    ),
                    results,
                    selections,
                    results_dir=slots[metric],
                    gold=gold_tags(splits[3], DATASET_TAGS[config.dataset]),
                    predictions=predictions,
                )
                mark_complete(root, slots[metric], run_dir, stats)
                bars.run_done(config_id, stats.get("score") if tuning else None)
    except BaseException as error:
        for slot in slots.values():
            if not (slot / "completed.json").exists():
                record_failure(slot, error)
        raise
    finally:
        write_summary(root, plan)


def _run_configurations(
    config: RandomSearchConfig,
    plan: dict,
    root: Path,
    summary: dict,
    splits: tuple,
    *,
    device,
    precision: str,
    publication: tuple | None,
) -> None:
    """Train every planned configuration that has not completed, in plan order."""
    tuning = config.mode == "tune-random"
    rows = summary["trials" if tuning else "comparisons"]
    stage = config.tuning_split if tuning else "test"
    announced = False
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as display:
        bars = _SweepProgress(
            display,
            f"{config.tuning_split.capitalize()} tuning" if tuning else "Comparisons",
            config.model_seeds,
            total=config.num_configs if tuning else config.num_configs * len(plan["uq_metrics"]),
            completed=sum(row["status"] == "complete" for row in rows),
            trial_scores={
                row["config_id"]: row["score"] for row in rows if row["status"] == "complete"
            }
            if tuning
            else {},
        )
        for entry in plan["configurations"]:
            if tuning:
                # The shared engine requires a routing key; random_only disables UQ scoring.
                slot = root / "tuning" / entry["config_id"]
                pending = [config.uq_metric] if completed_run(root, slot) is None else []
            else:
                pending = [
                    row["uq_metric"]
                    for row in rows
                    if row["config_id"] == entry["config_id"] and row["status"] != "complete"
                ]
            if not pending:
                continue
            _run_entry(
                config,
                plan,
                root,
                entry,
                pending,
                splits,
                stage=stage,
                device=device,
                precision=precision,
                bars=bars,
                announce_project=not announced,
            )
            announced = announced or config.wandb_enabled
            if tuning and publication is not None:
                publish_trial(config, root, entry, publication, random_tuning_score)


def _freeze_winner(
    config: RandomBaselineSearchConfig, root: Path, plan: dict, publication: tuple | None
) -> None:
    """Save the best tuning trial's settings once every trial has finished."""
    trials = write_summary(root, plan)["trials"]
    if any(row["status"] != "complete" for row in trials):
        raise ValueError("All tuning configurations must finish before selecting a winner")
    best = min(trials, key=lambda row: (-row["score"], row["config_id"]))
    frozen = {
        **omit_implicit_defaults(
            config.experiment_config().model_dump(
                exclude={"seed_workers", "uq_metric", *WANDB_FIELDS}
            )
        ),
        **best["parameters"],
    }
    winner_path = root / winner_filename(plan)
    if winner_path.exists():
        saved = {
            key: value
            for key, value in json.loads(winner_path.read_text()).items()
            if key not in {*WANDB_FIELDS, "uq_metric"}
        }
        if saved != frozen:
            raise ValueError(f"Saved {winner_path.name} differs; choose a new sweep_name")
    else:
        write_json(winner_path, frozen)
    preserve_json(
        root / "selection.json",
        {
            "config_id": best["config_id"],
            "objective": config.objective,
            f"{config.tuning_split}_score": best["score"],
            "selection_split": config.tuning_split,
        },
    )
    if publication is not None:
        mark_winner(publication, root)
    CONSOLE.print(
        f"Frozen winner: {best['config_id']} · {config.tuning_split} score "
        f"{best['score']:.6f} · {winner_path.name}"
    )
    write_summary(root, plan)


@_quiet_search_output()
def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = load_search_config(args, parser)
    tuning = config.mode == "tune-random"
    if args.publish_wandb_only and (not tuning or not config.wandb_enabled):
        parser.error("--publish-wandb-only requires tune-random and wandb_enabled=true")
    plan = tuning_plan(config) if tuning else sample_plan(config)
    device = get_device()
    precision = resolve_precision(config.precision, device)
    plan["effective_precision"] = precision
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root = config.sweeps_dir / config.sweep_name
    if args.publish_wandb_only and not (root / "plan.json").exists():
        parser.error("--publish-wandb-only requires an existing saved tuning plan")
    try:
        plan = open_sweep(root, plan, same_plan=_same_scientific_plan)
    except ValueError as error:
        parser.error(str(error))
    write_json(root / "search_config.json", config.model_dump(mode="json"))
    summary = write_summary(root, plan)
    if summary["complete"] and not (tuning and config.wandb_enabled):
        CONSOLE.print(f"Sweep already complete: {root}")
        return

    publication = None
    if config.wandb_enabled:
        if connect_wandb(PROJECT_ROOT / ".env", CONSOLE):
            CONSOLE.print(f"W&B enabled: {config.wandb_project} (project link at round 0).")
            if tuning:
                publication = _publish_tuning(config, plan, root)
        elif args.publish_wandb_only:
            parser.error("--publish-wandb-only requires online W&B mode")
    if args.publish_wandb_only:
        CONSOLE.print(f"Saved tuning results published; no training required: {root}")
        return
    if summary["complete"]:
        CONSOLE.print(f"Sweep already complete: {root}")
        return
    CONSOLE.print(
        f"[bold cyan]Compute precision:[/] [bold]{precision.upper()}[/]"
        f" · device: {device} · parameters/AdamW: FP32"
    )
    splits, identity = load_splits(config.dataset, config.dataset_percent)
    if tuning:
        splits = _tuning_splits(config, root, splits, identity)
    _run_configurations(
        config,
        plan,
        root,
        summary,
        splits,
        device=device,
        precision=precision,
        publication=publication,
    )
    if tuning:
        _freeze_winner(config, root, plan, publication)
    CONSOLE.print(f"Search complete. Results: {root / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
