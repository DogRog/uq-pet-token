#!/usr/bin/env python3
"""Run a paired hyperparameter sweep or tune random selection before testing UQ."""

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import polars as pl
from datasets.utils import logging as datasets_logging
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from transformers.utils import logging as transformers_logging

from uq_pet import active_learning
from uq_pet.active_learning import run_metric_comparisons, write_run
from uq_pet.experiment import (
    ExperimentConfig,
    RandomBaselineSearchConfig,
    RandomSearchConfig,
    configure_wandb_metrics,
    make_wandb_evaluation_log,
    require_wandb_credentials,
)
from uq_pet.pet_data import PROJECT_ROOT, download_pet_ner, load_pet_splits, split_tuning_pool
from uq_pet.token_model import UQ_METRICS, get_device, resolve_precision

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
LEARNING_RATE_BOUNDS = (1e-6, 1e-4)


@contextmanager
def _quiet_search_output():
    """Suppress routine model output while retaining concise sweep progress."""
    quiet = active_learning.CONSOLE.quiet
    model_bars = transformers_logging.is_progress_bar_enabled()
    data_bars = datasets_logging.is_progress_bar_enabled()
    try:
        active_learning.CONSOLE.quiet = True
        transformers_logging.disable_progress_bar()
        datasets_logging.disable_progress_bar()
        yield
    finally:
        active_learning.CONSOLE.quiet = quiet
        if model_bars:
            transformers_logging.enable_progress_bar()
        if data_bars:
            datasets_logging.enable_progress_bar()


def mean_entity_f1_gap_auc(results: Sequence[Mapping[str, object]]) -> float:
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
                float(uncertainty["entity_f1"]) - float(random_row["entity_f1"]),
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
    """Sample categorical settings and log-uniform rates before observing results."""
    rng = random.Random(config.sampler_seed)
    sampled = []
    for _ in range(config.num_configs):
        parameters = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        parameters["learning_rate"] = math.exp(
            rng.uniform(*(math.log(bound) for bound in LEARNING_RATE_BOUNDS))
        )
        sampled.append(parameters)
    # Worker scheduling can change on resume; scientific settings cannot.
    fixed = config.experiment_config().model_dump(
        exclude={"seed_workers", "uq_metric", "learning_rate", *SEARCH_SPACE, *WANDB_FIELDS}
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
            "learning_rate": {
                "distribution": "log_uniform",
                "low": LEARNING_RATE_BOUNDS[0],
                "high": LEARNING_RATE_BOUNDS[1],
            },
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


def write_json(path: Path, value: dict) -> None:
    """Publish metadata atomically so an interrupted write is never considered complete."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def tuning_plan(config: RandomBaselineSearchConfig) -> dict:
    """Lock sampling, holdout, objective, and final acquisition rule before training."""
    sampled = sample_plan(config)
    return {
        "version": 2,
        "mode": "random_baseline_tuning",
        "sampling": sampled["sampling"],
        "sampler_seed": config.sampler_seed,
        "num_configs": config.num_configs,
        "search_space": sampled["search_space"],
        "configurations": sampled["configurations"],
        "fixed_config": sampled["fixed_config"],
        "validation_sentences": config.validation_sentences,
        "validation_seed": config.validation_seed,
        "objective": config.objective,
        "tie_break": "config_id_ascending",
        "final_uq_metric": config.uq_metric,
        "tuning_evaluation_split": "validation",
        "final_evaluation_split": "test",
        "final_pool": "original_328_sentences",
    }


def random_validation_score(results: list[dict], objective: str) -> float:
    """Average random validation F1 across seeds, using final F1 or normalized curve AUC."""
    if not results or any(row["arm"] != "random" for row in results):
        raise ValueError("tuning requires nonempty random-only validation results")
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
        if objective == "random_validation_final_entity_f1":
            scores.append(points[-1][1])
        elif objective == "random_validation_entity_f1_auc":
            area = sum(
                (b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:], strict=False)
            )
            scores.append(area / (points[-1][0] - points[0][0]))
        else:
            raise ValueError(f"unknown objective: {objective}")
    return sum(scores) / len(scores)


def preserve_json(path: Path, value: dict) -> None:
    """Refuse to mix scientific plans, dataset splits, or frozen winners across resumes."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Saved {path.name} differs; choose a new sweep_name")
    else:
        write_json(path, value)


def completed_run(root: Path, slot: Path) -> dict | None:
    marker = slot / "completed.json"
    if not marker.exists():
        return None
    completed = json.loads(marker.read_text())
    for name in ("config.json", "results.csv", "selections.json"):
        if not (root / completed["run_dir"] / name).is_file():
            raise ValueError(f"Completed run is missing {name}: {marker}")
    return completed


def scientific_plan(plan: dict) -> dict:
    """Ignore logging fields in older plans without relaxing any experiment settings."""
    return {
        **plan,
        "fixed_config": {
            key: value for key, value in plan["fixed_config"].items() if key not in WANDB_FIELDS
        },
    }


@contextmanager
def wandb_comparison_logging(
    config,
    experiment,
    config_id,
    effective_precision,
    slots,
    *,
    evaluation_split="test",
    random_only=False,
):
    """Own one W&B run per metric/seed in the parent, with separate acquisition axes."""
    if not config.wandb_enabled:
        yield None
        return

    import wandb

    runs = {}
    success = False
    offline = os.environ.get("WANDB_MODE", "").strip().lower() == "offline"
    with ExitStack() as cleanup:

        def log_evaluation(metric, rows):
            pair = rows[-1:] if random_only else rows[-2:]
            payload = make_wandb_evaluation_log(pair, metric, random_only=random_only)
            seed = pair[0]["seed"]
            key = (metric, seed)
            if key not in runs:
                name = (
                    f"{config.wandb_run_name or config.sweep_name}-{config_id}-{metric}-seed{seed}"
                )
                if random_only:
                    name += "-validation-random"
                settings = experiment.model_copy(
                    update={"uq_metric": metric, "model_seeds": [seed]}
                )
                run = wandb.init(
                    project=config.wandb_project,
                    name=name,
                    group=config.sweep_name,
                    job_type="random_search",
                    reinit="create_new",
                    settings={"quiet": True, "console": "off"},
                    dir=str(slots[metric]),
                    config={
                        **settings.resolved_dict(),
                        "wandb_run_name": name,
                        "seed": seed,
                        "config_id": config_id,
                        "sweep_name": config.sweep_name,
                        "evaluation_split": evaluation_split,
                        "random_only": random_only,
                        "effective_precision": effective_precision,
                        "shared_bootstrap_and_random": not random_only,
                    },
                )
                cleanup.callback(lambda run=run: run.finish(exit_code=0 if success else 1))
                runs[key] = run
                configure_wandb_metrics(run, metric, random_only=random_only)
                if not offline:
                    if len(runs) == 1:
                        CONSOLE.print(f"W&B project: {run.get_project_url()}", markup=False)
                    CONSOLE.print(f"W&B run ({metric}, seed {seed}): {run.get_url()}", markup=False)
            runs[key].log(payload)

        yield log_evaluation
        success = True


def paired_summary(results: list[dict]) -> dict:
    """Describe a paired run without selecting or ranking configurations."""
    auc = mean_entity_f1_gap_auc(results)
    seeds = sorted({row["seed"] for row in results})
    final_gaps = []
    for seed in seeds:
        rows = [row for row in results if row["seed"] == seed]
        last_round = max(row["round"] for row in rows)
        final = {row["arm"]: row["entity_f1"] for row in rows if row["round"] == last_round}
        final_gaps.append(final["uncertainty"] - final["random"])
    return {
        "mean_test_entity_f1_gap_auc": auc,
        "mean_final_test_entity_f1_gap": sum(final_gaps) / len(seeds),
    }


def run_status(root: Path, slot: Path) -> dict:
    completed = completed_run(root, slot)
    if completed is not None:
        return {**completed, "status": "complete"}
    if (slot / "failure.json").exists():
        return {**json.loads((slot / "failure.json").read_text()), "status": "failed"}
    return {"status": "pending"}


def write_summary(root: Path, plan: dict) -> dict:
    """Include every planned comparison and expose incomplete sweep coverage."""
    if plan.get("mode") == "random_baseline_tuning":
        trials = [
            {**entry, **run_status(root, root / "tuning" / entry["config_id"])}
            for entry in plan["configurations"]
        ]
        final = run_status(root, root / "final")
        summary = {
            "mode": plan["mode"],
            "objective": plan["objective"],
            "complete": final["status"] == "complete"
            and all(row["status"] == "complete" for row in trials),
            "trials": trials,
            "final": final,
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
        choices=("compare", "tune-random"),
        default=argparse.SUPPRESS,
        help="Compare every configuration on test (default), or tune random on validation then test the winner.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without loading data or models."
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
                if key not in {"config", "config_json", "dry_run"}
            }
        )
        config_class = (
            RandomBaselineSearchConfig
            if settings.get("mode") == "tune-random"
            else RandomSearchConfig
        )
        config = config_class.model_validate(settings)
        sample_plan(config)
        return config
    except (OSError, ValueError) as error:
        parser.error(str(error))


@_quiet_search_output()
def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = load_search_config(args, parser)
    tuning = config.mode == "tune-random"
    plan = tuning_plan(config) if tuning else sample_plan(config)
    device = get_device()
    effective_precision = resolve_precision(config.precision, device)
    plan["effective_precision"] = effective_precision
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root = config.sweeps_dir / config.sweep_name
    plan_path = root / "plan.json"
    if plan_path.exists():
        if scientific_plan(json.loads(plan_path.read_text())) != scientific_plan(plan):
            parser.error(
                "Existing sweep has a different plan; choose a new --sweep-name. The budget is fixed, not additional runs."
            )
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("Sweep directory is nonempty but has no plan; choose a new --sweep-name.")
        root.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    write_json(root / "search_config.json", config.model_dump(mode="json"))
    summary = write_summary(root, plan)
    if summary["complete"]:
        CONSOLE.print(f"Sweep already complete: {root}")
        return

    if config.wandb_enabled:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        require_wandb_credentials(os.environ)
        if os.environ.get("WANDB_MODE", "").strip().lower() == "offline":
            CONSOLE.print("W&B offline: logging locally; no online project link is available.")
        else:
            import wandb

            wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
            CONSOLE.print(f"W&B enabled: {config.wandb_project} (links appear at round 0).")
    CONSOLE.print(
        f"[bold cyan]Compute precision:[/] [bold]{effective_precision.upper()}[/]"
        f" · device: {device} · parameters/AdamW: FP32"
    )
    data_path = download_pet_ner()
    seed_examples, pool_inputs, pool_gold, test_examples = load_pet_splits(data_path)
    if tuning:
        tune_pool, tune_gold, validation_examples, manifest = split_tuning_pool(
            pool_inputs,
            pool_gold,
            validation_sentences=config.validation_sentences,
            validation_seed=config.validation_seed,
        )
        preserve_json(
            root / "split.json",
            {
                **manifest,
                "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                "seed_sentences": len(seed_examples),
                "tuning_pool_sentences": len(tune_pool),
                "validation_sentences": len(validation_examples),
                "final_pool_sentences": len(pool_inputs),
                "test_sentences": len(test_examples),
            },
        )
    total = config.num_configs + 1 if tuning else config.num_configs * len(plan["uq_metrics"])
    completed_runs = sum(
        row["status"] == "complete" for row in summary["trials" if tuning else "comparisons"]
    )
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        overall_task = progress.add_task("Configurations", total=total, completed=completed_runs)
        seed_tasks = {
            seed: progress.add_task(f"Seed {seed} · waiting / bootstrap", total=1)
            for seed in config.model_seeds
        }

        def execute(entry, pending_metrics, *, stage, final=False):
            nonlocal completed_runs
            random_only = stage == "validation"
            inputs, gold, evaluation = (
                (tune_pool, tune_gold, validation_examples)
                if random_only
                else (pool_inputs, pool_gold, test_examples)
            )
            slots = {
                metric: (
                    root / "final"
                    if final
                    else root / "tuning" / entry["config_id"]
                    if random_only
                    else root / "runs" / entry["config_id"] / metric
                )
                for metric in pending_metrics
            }
            for slot in slots.values():
                slot.mkdir(parents=True, exist_ok=True)
                (slot / "progress.csv").unlink(missing_ok=True)
            experiment = ExperimentConfig.model_validate(
                {
                    **config.experiment_config().model_dump(),
                    **entry["parameters"],
                }
            )
            description = f"{'Final test' if final else stage.capitalize()} {entry['config_id']}"
            progress.update(overall_task, description=description)
            for seed, task in seed_tasks.items():
                progress.reset(task, total=1, description=f"Seed {seed} · waiting / bootstrap")
            seed_fractions = {
                (metric, seed): 0.0 for metric in pending_metrics for seed in seed_tasks
            }
            base_completed = completed_runs

            def update_progress(metric, rows):
                slot = slots[metric]
                temporary = slot / "progress.tmp"
                pl.DataFrame(rows).write_csv(temporary)
                temporary.replace(slot / "progress.csv")
                if log_evaluation is not None:
                    log_evaluation(metric, rows)
                latest = rows[-1]
                seed = latest["seed"]
                progress.update(
                    seed_tasks[seed],
                    total=latest["total_rounds"],
                    completed=latest["round"],
                    description=f"Seed {seed} · {'random' if random_only else metric} · round {latest['round']}/{latest['total_rounds']}",
                )
                seed_fractions[metric, seed] = (latest["round"] + 1) / (latest["total_rounds"] + 1)
                progress.update(
                    overall_task,
                    completed=base_completed + sum(seed_fractions.values()) / len(seed_tasks),
                )

            try:
                with wandb_comparison_logging(
                    config,
                    experiment,
                    entry["config_id"],
                    effective_precision,
                    slots,
                    evaluation_split=stage,
                    random_only=random_only,
                ) as log_evaluation:
                    settings = experiment.active_learning_kwargs()
                    settings.pop("uq_metric")
                    comparisons = run_metric_comparisons(
                        seed_examples,
                        inputs,
                        gold,
                        evaluation,
                        **{**settings, "precision": effective_precision},
                        uq_metrics=pending_metrics,
                        random_only=random_only,
                        device=device,
                        progress_callback=update_progress,
                    )
                    progress.update(overall_task, description=f"{description} · saving")
                    for metric in pending_metrics:
                        results, selections = comparisons[metric]
                        slot = slots[metric]
                        run_config = experiment.model_copy(update={"uq_metric": metric})
                        stats = (
                            {"score": random_validation_score(results, config.objective)}
                            if random_only
                            else paired_summary(results)
                        )
                        if final:
                            stats["per_seed"] = [
                                {
                                    "seed": seed,
                                    **paired_summary([r for r in results if r["seed"] == seed]),
                                }
                                for seed in config.model_seeds
                            ]
                        metadata = {
                            **run_config.resolved_dict(),
                            "effective_precision": effective_precision,
                            "shared_bootstrap_and_random": not random_only,
                            "random_only": random_only,
                            "evaluation_split": stage,
                            "mode": "random_baseline_tuning" if tuning else "random_search",
                            "sweep_name": config.sweep_name,
                            "config_id": entry["config_id"],
                            "seed_sentences": len(seed_examples),
                            "pool_sentences": len(inputs),
                            "evaluation_sentences": len(evaluation),
                        }
                        metadata["validation_sentences" if random_only else "test_sentences"] = len(
                            evaluation
                        )
                        if tuning:
                            metadata["objective"] = config.objective
                        run_dir = write_run(metadata, results, selections, results_dir=slot)
                        write_json(
                            slot / "completed.json",
                            {"run_dir": str(run_dir.relative_to(root)), **stats},
                        )
                        (slot / "failure.json").unlink(missing_ok=True)
                        completed_runs += 1
                        progress.update(
                            overall_task, completed=completed_runs, description=description
                        )
            except BaseException as error:
                for slot in slots.values():
                    if not (slot / "completed.json").exists():
                        write_json(
                            slot / "failure.json", {"error": f"{type(error).__name__}: {error}"}
                        )
                raise
            finally:
                write_summary(root, plan)

        for entry in plan["configurations"]:
            if tuning:
                pending_metrics = (
                    [config.uq_metric]
                    if completed_run(root, root / "tuning" / entry["config_id"]) is None
                    else []
                )
            else:
                pending_metrics = [
                    row["uq_metric"]
                    for row in summary["comparisons"]
                    if row["config_id"] == entry["config_id"] and row["status"] != "complete"
                ]
            if pending_metrics:
                execute(entry, pending_metrics, stage="validation" if tuning else "test")

        if tuning:
            trials = write_summary(root, plan)["trials"]
            if any(row["status"] != "complete" for row in trials):
                raise ValueError("All tuning configurations must finish before selecting a winner")
            best = min(trials, key=lambda row: (-row["score"], row["config_id"]))
            frozen = {
                **config.experiment_config().model_dump(exclude={"seed_workers", *WANDB_FIELDS}),
                **best["parameters"],
            }
            winner_path = root / "best_config.json"
            if winner_path.exists():
                saved = {
                    key: value
                    for key, value in json.loads(winner_path.read_text()).items()
                    if key not in WANDB_FIELDS
                }
                if saved != frozen:
                    raise ValueError("Saved best_config.json differs; choose a new sweep_name")
            else:
                write_json(winner_path, frozen)
            preserve_json(
                root / "selection.json",
                {
                    "config_id": best["config_id"],
                    "objective": config.objective,
                    "validation_score": best["score"],
                    "selection_split": "validation",
                },
            )
            CONSOLE.print(
                f"Frozen winner: {best['config_id']} · validation score {best['score']:.6f}"
            )
            # Use the persisted scientific configuration; current logging/worker settings may vary.
            execute(
                {"config_id": best["config_id"], "parameters": frozen},
                [config.uq_metric],
                stage="test",
                final=True,
            )
    CONSOLE.print(f"Search complete. Results: {root / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
