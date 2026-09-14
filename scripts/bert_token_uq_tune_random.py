#!/usr/bin/env python3
"""Tune random acquisition on validation, freeze the winner, then compare UQ on test."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import polars as pl

from bert_token_uq_search import (
    CONSOLE,
    _quiet_search_output,
    configure_parser,
    paired_summary,
    sample_plan,
    write_json,
)
from uq_pet.active_learning import run_active_learning, run_random_selection, write_run
from uq_pet.experiment import ExperimentConfig, RandomBaselineSearchConfig
from uq_pet.pet_data import download_pet_ner, load_pet_splits, split_tuning_pool
from uq_pet.token_model import get_device, resolve_precision


def tuning_plan(config: RandomBaselineSearchConfig) -> dict:
    """Lock sampling, holdout, objective, and final acquisition rule before training."""
    sampled = sample_plan(config)
    return {
        "version": 1,
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


def write_summary(root: Path, plan: dict) -> dict:
    trials = []
    for entry in plan["configurations"]:
        slot = root / "tuning" / entry["config_id"]
        completed = completed_run(root, slot)
        row = {**entry, "status": "pending"}
        if completed is not None:
            row.update(completed, status="complete")
        elif (slot / "failure.json").exists():
            row.update(json.loads((slot / "failure.json").read_text()), status="failed")
        trials.append(row)
    final = completed_run(root, root / "final")
    final_status = {"status": "pending"}
    if final is not None:
        final_status.update(final, status="complete")
    elif (root / "final" / "failure.json").exists():
        final_status.update(
            json.loads((root / "final" / "failure.json").read_text()), status="failed"
        )
    summary = {
        "mode": plan["mode"],
        "objective": plan["objective"],
        "complete": final is not None and all(row["status"] == "complete" for row in trials),
        "trials": trials,
        "final": final_status,
    }
    write_json(root / "summary.json", summary)
    return summary


@_quiet_search_output()
def run(config: RandomBaselineSearchConfig, *, dry_run: bool = False) -> None:
    plan = tuning_plan(config)
    device = get_device()
    precision = resolve_precision(config.precision, device)
    plan["effective_precision"] = precision
    if dry_run:
        CONSOLE.print(json.dumps(plan, indent=2), markup=False)
        return
    root = config.sweeps_dir / config.sweep_name
    if root.exists() and any(root.iterdir()) and not (root / "plan.json").exists():
        raise ValueError("Nonempty sweep directory has no plan; choose a new sweep_name")
    root.mkdir(parents=True, exist_ok=True)
    preserve_json(root / "plan.json", plan)
    write_json(root / "search_config.json", config.model_dump(mode="json"))
    summary = write_summary(root, plan)
    if summary["complete"]:
        CONSOLE.print(f"Already complete: {root / 'summary.json'}")
        return

    data_path = download_pet_ner()
    seed, pool, gold, test = load_pet_splits(data_path)
    tune_pool, tune_gold, validation, manifest = split_tuning_pool(
        pool,
        gold,
        validation_sentences=config.validation_sentences,
        validation_seed=config.validation_seed,
    )
    preserve_json(
        root / "split.json",
        {
            **manifest,
            "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "seed_sentences": len(seed),
            "tuning_pool_sentences": len(tune_pool),
            "validation_sentences": len(validation),
            "final_pool_sentences": len(pool),
            "test_sentences": len(test),
        },
    )
    CONSOLE.print(
        f"Random baseline tuning: {config.num_configs} configurations × {len(config.model_seeds)} seeds"
        f" · {device} · {precision.upper()} · objective: {config.objective}"
    )

    def execute(slot, experiment, inputs, labels, evaluation, *, stage, config_id):
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "progress.csv").unlink(missing_ok=True)

        def publish(rows):
            temporary = slot / "progress.tmp"
            pl.DataFrame(rows).write_csv(temporary)
            temporary.replace(slot / "progress.csv")
            latest = rows[-1]
            CONSOLE.print(
                f"{stage} {config_id} · seed {latest['seed']} · "
                f"round {latest['round']}/{latest['total_rounds']} · "
                f"{latest['arm']} F1 {latest['entity_f1']:.4f}"
            )

        try:
            runner = run_random_selection if stage == "validation" else run_active_learning
            results, selections = runner(
                seed,
                inputs,
                labels,
                evaluation,
                **{**experiment.active_learning_kwargs(), "precision": precision},
                device=device,
                progress_callback=publish,
            )
            stats = (
                {"score": random_validation_score(results, config.objective)}
                if stage == "validation"
                else {
                    **paired_summary(results),
                    "per_seed": [
                        {
                            "seed": seed_id,
                            **paired_summary([r for r in results if r["seed"] == seed_id]),
                        }
                        for seed_id in config.model_seeds
                    ],
                }
            )
            run_dir = write_run(
                {
                    **experiment.resolved_dict(),
                    "effective_precision": precision,
                    "evaluation_split": stage,
                    "mode": plan["mode"],
                    "config_id": config_id,
                    "seed_sentences": len(seed),
                    "pool_sentences": len(inputs),
                    "evaluation_sentences": len(evaluation),
                    "objective": config.objective,
                },
                results,
                selections,
                results_dir=slot,
            )
            marker = {"run_dir": str(run_dir.relative_to(root)), **stats}
            write_json(slot / "completed.json", marker)
            (slot / "failure.json").unlink(missing_ok=True)
            return marker
        except BaseException as error:
            write_json(slot / "failure.json", {"error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            write_summary(root, plan)

    for entry in plan["configurations"]:
        slot = root / "tuning" / entry["config_id"]
        if completed_run(root, slot) is not None:
            continue
        experiment = ExperimentConfig.model_validate(
            {
                **config.experiment_config().model_dump(),
                **entry["parameters"],
            }
        )
        execute(
            slot,
            experiment,
            tune_pool,
            tune_gold,
            validation,
            stage="validation",
            config_id=entry["config_id"],
        )

    trials = write_summary(root, plan)["trials"]
    if any(row["status"] != "complete" for row in trials):
        raise ValueError("All tuning configurations must finish before selecting a winner")
    best = min(trials, key=lambda row: (-row["score"], row["config_id"]))
    frozen = {
        **config.experiment_config().model_dump(exclude={"seed_workers"}),
        **best["parameters"],
    }
    preserve_json(root / "best_config.json", frozen)
    preserve_json(
        root / "selection.json",
        {
            "config_id": best["config_id"],
            "objective": config.objective,
            "validation_score": best["score"],
            "selection_split": "validation",
        },
    )
    CONSOLE.print(f"Frozen winner: {best['config_id']} · validation score {best['score']:.6f}")
    experiment = ExperimentConfig.model_validate({**frozen, "seed_workers": config.seed_workers})
    # Fresh bootstrap; both arms share its weights/optimizer and use the original pool.
    execute(root / "final", experiment, pool, gold, test, stage="test", config_id=best["config_id"])
    CONSOLE.print(
        f"Completed validation tuning and held-out test comparison: {root / 'summary.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed plan without loading data or models.",
    )
    args = parser.parse_args()
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
        config = RandomBaselineSearchConfig.model_validate(settings)
        run(config, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
