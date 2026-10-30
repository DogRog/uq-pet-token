"""Check the uncertainty-minus-random gap at a follow-up pilot sweep's winner.

Trains both arms on the sweep's validation holdout at its frozen winner (default metric,
top-K), once without and once with averaged models, and saves each run with its gap AUC
under the sweep's gap_check/. The test split is never read. A finished variant is skipped
on a rerun.
"""

import argparse
import json
from pathlib import Path

from pilot_report import _sweep
from uq_pet.active_learning import run_metric_comparisons, write_run
from uq_pet.config import ExperimentConfig
from uq_pet.data_prep import DATASET_TAGS, load_splits, split_tuning_pool
from uq_pet.search import mean_entity_f1_gap_auc
from uq_pet.token_model import get_device, gold_tags, resolve_precision

VARIANTS = {"constant": False, "averaged": True}


def run_gap_check(config_path: Path) -> dict[str, float]:
    """Run each unfinished variant at the sweep's winner; return every variant's gap AUC."""
    config, root = _sweep(config_path)
    winner_path = root / "best_config.json"
    if not winner_path.is_file():
        raise ValueError(f"{config.sweep_name} has no frozen winner yet")
    winner = json.loads(winner_path.read_text())
    output = root / "gap_check"
    output.mkdir(exist_ok=True)
    device = get_device()
    precision = resolve_precision(config.precision, device)
    splits = None
    gaps = {}
    for name, averaging in VARIANTS.items():
        record = output / f"{name}.json"
        if record.is_file():
            gaps[name] = json.loads(record.read_text())["gap_auc"]
            continue
        if splits is None:
            (seed_examples, pool_inputs, pool_gold, _), _ = load_splits(
                config.dataset, config.dataset_percent
            )
            tune_pool, tune_gold, validation, _ = split_tuning_pool(
                pool_inputs,
                pool_gold,
                validation_sentences=config.validation_sentences,
                validation_seed=config.validation_seed,
            )
            splits = (seed_examples, tune_pool, tune_gold, validation)
        experiment = ExperimentConfig.model_validate(
            {
                **config.experiment_config().model_dump(),
                **winner,
                "weight_averaging": averaging,
            }
        )
        settings = experiment.active_learning_kwargs()
        metric = settings.pop("uq_metric")
        results, selections, predictions = run_metric_comparisons(
            *splits,
            **{**settings, "precision": precision},
            uq_metrics=[metric],
            device=device,
        )[metric]
        run_dir = write_run(
            {**experiment.model_dump(mode="json"), "evaluation_split": "validation"},
            results,
            selections,
            results_dir=output / name,
            gold=gold_tags(splits[3], DATASET_TAGS[config.dataset]),
            predictions=predictions,
        )
        gaps[name] = mean_entity_f1_gap_auc(results)
        record.write_text(
            json.dumps({"gap_auc": gaps[name], "run_dir": str(run_dir.relative_to(root))}, indent=2)
        )
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    args = parser.parse_args(argv)
    if args.dry_run:
        config, root = _sweep(args.config)
        print(f"Gap check of {config.sweep_name}: {', '.join(VARIANTS)} in {root / 'gap_check'}")
        return 0
    for name, gap in run_gap_check(args.config).items():
        print(f"{name}: validation entity-F1 gap AUC {gap:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
