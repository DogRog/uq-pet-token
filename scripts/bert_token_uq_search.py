#!/usr/bin/env python3
"""Run a fixed random hyperparameter sweep with paired UQ/random test evaluation."""

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import polars as pl
from datasets.utils import logging as datasets_logging
from rich.console import Console
from transformers.utils import logging as transformers_logging

from uq_pet import active_learning
from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.experiment import ExperimentConfig, RandomSearchConfig
from uq_pet.pet_data import download_pet_ner, load_pet_splits
from uq_pet.token_model import UQ_METRICS, get_device

CONSOLE = Console()
SEARCH_SPACE = {
    "k": (8, 16, 32),
    "bootstrap_epochs": (10,),
    "update_passes": (1, 2, 4),
    "learning_rate": (2e-5, 5e-5),
    "batch_size": (8, 16),
    "replay_ratio": (0, 1.0, 2.0),
    "weight_decay": (0.0, 0.01),
}


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
    """Choose distinct combinations uniformly before observing any results."""
    combinations = list(product(*SEARCH_SPACE.values()))
    if config.num_configs > len(combinations):
        raise ValueError(f"num_configs cannot exceed the {len(combinations)} distinct combinations")
    sampled = random.Random(config.sampler_seed).sample(combinations, config.num_configs)
    # Worker scheduling can change on resume; scientific settings cannot.
    fixed = config.experiment_config().model_dump(
        exclude={"seed_workers", "uq_metric", *SEARCH_SPACE}
    )
    return {
        "version": 1,
        "evaluation_split": "test",
        "sampling": "uniform_without_replacement",
        "sampler_seed": config.sampler_seed,
        "num_configs": config.num_configs,
        "fixed_config": fixed,
        "search_space": {key: list(values) for key, values in SEARCH_SPACE.items()},
        "uq_metrics": list(UQ_METRICS),
        "configurations": [
            {
                "config_id": f"config_{index:04d}",
                "parameters": dict(zip(SEARCH_SPACE, values, strict=True)),
            }
            for index, values in enumerate(sampled)
        ],
    }


def write_json(path: Path, value: dict) -> None:
    """Publish metadata atomically so an interrupted write is never considered complete."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


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


def write_summary(root: Path, plan: dict) -> dict:
    """Include every planned comparison and expose incomplete sweep coverage."""
    comparisons = []
    for entry in plan["configurations"]:
        for metric in plan["uq_metrics"]:
            slot = root / "runs" / entry["config_id"] / metric
            marker = slot / "completed.json"
            row = {
                "config_id": entry["config_id"],
                "uq_metric": metric,
                "parameters": entry["parameters"],
            }
            if marker.exists():
                completed = json.loads(marker.read_text())
                for filename in ("results.csv", "config.json", "selections.json"):
                    if not (root / completed["run_dir"] / filename).is_file():
                        raise ValueError(f"Completed run is missing {filename}: {marker}")
                row.update(completed, status="complete")
            elif (slot / "failure.json").exists():
                row.update(
                    status="failed", error=json.loads((slot / "failure.json").read_text())["error"]
                )
            else:
                row["status"] = "pending"
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON file with sweep and experiment settings.")
    source.add_argument("--config-json", help="Inline JSON with sweep and experiment settings.")
    parser.add_argument(
        "--num-configs",
        type=int,
        default=argparse.SUPPRESS,
        help="Total fixed budget of distinct configurations, each run with every UQ metric.",
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
                if key not in {"config", "config_json"}
            }
        )
        config = RandomSearchConfig.model_validate(settings)
        sample_plan(config)
        return config
    except (OSError, ValueError) as error:
        parser.error(str(error))


@_quiet_search_output()
def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = load_search_config(args, parser)
    plan = sample_plan(config)
    root = config.sweeps_dir / config.sweep_name
    plan_path = root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
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

    # Use the original 5/328/84 split. Test labels never enter acquisition or training.
    seed_examples, pool_inputs, pool_gold, test_examples = load_pet_splits(download_pet_ner())
    device = get_device()
    total = config.num_configs * len(plan["uq_metrics"])
    for index, comparison in enumerate(summary["comparisons"]):
        if comparison["status"] == "complete":
            continue
        slot = root / "runs" / comparison["config_id"] / comparison["uq_metric"]
        slot.mkdir(parents=True, exist_ok=True)
        experiment = ExperimentConfig.model_validate(
            {
                **plan["fixed_config"],
                **comparison["parameters"],
                "uq_metric": comparison["uq_metric"],
                "seed_workers": config.seed_workers,
            }
        )
        CONSOLE.print(
            f"Run {index + 1}/{total}: {comparison['config_id']} / {comparison['uq_metric']}"
        )

        status = CONSOLE.status("Preparing test evaluation")

        def update_progress(rows, slot=slot, status=status):
            # The engine publishes only completed pairs, after bootstrap and each round.
            temporary = slot / "progress.tmp"
            pl.DataFrame(rows).write_csv(temporary)
            temporary.replace(slot / "progress.csv")
            latest = rows[-1]
            status.update(
                f"Seed {latest['seed']} · test round {latest['round']}/{latest['total_rounds']}"
            )

        try:
            with status:
                results, selections = run_active_learning(
                    seed_examples,
                    pool_inputs,
                    pool_gold,
                    test_examples,
                    **experiment.active_learning_kwargs(),
                    device=device,
                    progress_callback=update_progress,
                )
            stats = paired_summary(results)
            run_dir = write_run(
                {
                    **experiment.resolved_dict(),
                    "evaluation_split": "test",
                    "mode": "random_search",
                    "sweep_name": config.sweep_name,
                    "config_id": comparison["config_id"],
                    "seed_sentences": len(seed_examples),
                    "pool_sentences": len(pool_inputs),
                    "test_sentences": len(test_examples),
                },
                results,
                selections,
                results_dir=slot,
            )
            write_json(
                slot / "completed.json", {"run_dir": str(run_dir.relative_to(root)), **stats}
            )
            (slot / "failure.json").unlink(missing_ok=True)
        except BaseException as error:
            write_json(slot / "failure.json", {"error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            write_summary(root, plan)
    CONSOLE.print(f"Test sweep complete. All comparisons: {root / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
