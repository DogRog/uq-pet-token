#!/usr/bin/env python3
"""Tune PET token-UQ hyperparameters without using the held-out test split."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import optuna
from rich.console import Console
from rich.table import Table

from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.experiment import ExperimentConfig
from uq_pet.pet_data import RESULTS_DIR, download_pet_ner, load_pet_splits
from uq_pet.token_model import get_device
from uq_pet.tuning import (
    SEARCH_SPACE,
    make_tuning_split,
    mean_entity_f1_gap_auc,
    suggest_search_config,
)

DEFAULT_STUDIES_DIR = RESULTS_DIR / "optuna"
OBJECTIVE_NAME = "mean_validation_entity_f1_gap_auc"
CONSOLE = Console()


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _study_context(
    base_config: ExperimentConfig,
    *,
    validation_fraction: float,
    validation_seed: int,
    sampler_seed: int,
) -> dict:
    fixed_config = base_config.model_dump(exclude=set(SEARCH_SPACE))
    return {
        "objective": OBJECTIVE_NAME,
        "acquisition_schedule": "partial_final_round",
        "fixed_config": fixed_config,
        "search_space": {key: list(values) for key, values in SEARCH_SPACE.items()},
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "sampler_seed": sampler_seed,
    }


def _context_fingerprint(context: dict) -> str:
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_study_outputs(
    study: optuna.Study, study_root: Path, base_config: ExperimentConfig
) -> None:
    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "datetime_start": trial.datetime_start.isoformat()
            if trial.datetime_start is not None
            else None,
            "datetime_complete": trial.datetime_complete.isoformat()
            if trial.datetime_complete is not None
            else None,
        }
        for trial in study.trials
    ]
    (study_root / "trials.json").write_text(json.dumps(trials, indent=2))
    if not complete_trials:
        return

    best_config = ExperimentConfig.model_validate(
        {
            **base_config.model_dump(),
            **study.best_trial.params,
            "wandb_enabled": False,
            "wandb_run_name": "",
        }
    )
    (study_root / "best_config.json").write_text(json.dumps(best_config.model_dump(), indent=2))
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "objective": OBJECTIVE_NAME,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "completed_trials": len(complete_trials),
        "total_trials": len(study.trials),
        "updated_at": datetime.now().isoformat(),
    }
    (study_root / "summary.json").write_text(json.dumps(summary, indent=2))


def _make_objective(
    *,
    base_config: ExperimentConfig,
    seed_examples: list[dict],
    tuning_pool_inputs: list[dict],
    tuning_pool_gold: dict[tuple[int, int], int],
    validation_examples: list[dict],
    device,
    study_root: Path,
):
    def objective(trial: optuna.Trial) -> float:
        config = ExperimentConfig.model_validate(
            {
                **base_config.model_dump(),
                **suggest_search_config(trial),
                "wandb_enabled": False,
                "wandb_run_name": "",
            }
        )
        results, selections = run_active_learning(
            seed_examples,
            tuning_pool_inputs,
            tuning_pool_gold,
            validation_examples,
            **config.active_learning_kwargs(),
            device=device,
        )
        value = mean_entity_f1_gap_auc(results)
        trial_root = study_root / "trials" / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True)
        run_config = {
            **config.resolved_dict(),
            "mode": "optuna_tuning",
            "optuna_trial": trial.number,
            "objective": OBJECTIVE_NAME,
            "objective_value": value,
            "tuning_pool_sentences": len(tuning_pool_inputs),
            "validation_sentences": len(validation_examples),
        }
        run_dir = write_run(run_config, results, selections, results_dir=trial_root)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("validation_gap_auc", value)
        trial.set_user_attr("token_budget", results[0]["token_budget"])
        trial.set_user_attr("scoreable_pool_tokens", results[0]["scoreable_pool_tokens"])
        return value

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        required=True,
        help="Number of additional trials to run; required to avoid accidental long studies.",
    )
    parser.add_argument(
        "--config-json",
        default="{}",
        help="Fixed ExperimentConfig values as one JSON object; tuned fields are overwritten.",
    )
    parser.add_argument(
        "--study-name",
        default="bert-token-uq",
        help="Persistent Optuna study name (default: bert-token-uq).",
    )
    parser.add_argument(
        "--studies-dir",
        type=Path,
        default=DEFAULT_STUDIES_DIR,
        help="Directory containing study databases and trial artifacts.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of pool sentences withheld from acquisition for validation.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1729,
        help="Local RNG seed for the nested validation split.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=0,
        help="Seed for Optuna's TPE sampler.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional study timeout in seconds.",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    if not _slug(args.study_name):
        parser.error("--study-name must contain at least one letter or number")
    try:
        base_config = ExperimentConfig.model_validate_json(args.config_json)
    except ValueError as error:
        parser.error(str(error))
    base_config = ExperimentConfig.model_validate(
        {
            **base_config.model_dump(),
            "wandb_enabled": False,
            "wandb_run_name": "",
        }
    )

    study_root = args.studies_dir / _slug(args.study_name)
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "trials").mkdir(exist_ok=True)
    storage = f"sqlite:///{study_root / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )
    context = _study_context(
        base_config,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        sampler_seed=args.sampler_seed,
    )
    fingerprint = _context_fingerprint(context)
    existing_fingerprint = study.user_attrs.get("context_fingerprint")
    if existing_fingerprint is not None and existing_fingerprint != fingerprint:
        parser.error(
            "the existing study uses different fixed settings, search space, acquisition schedule, "
            "or validation split; choose a new --study-name"
        )
    study.set_user_attr("context", context)
    study.set_user_attr("context_fingerprint", fingerprint)

    data_path = download_pet_ner()
    seed_examples, pool_inputs, pool_gold, _test_examples = load_pet_splits(data_path)
    tuning_pool_inputs, tuning_pool_gold, validation_examples = make_tuning_split(
        pool_inputs,
        pool_gold,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
    device = get_device()

    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan", justify="right")
    overview.add_column()
    overview.add_row("Study", args.study_name)
    overview.add_row("Additional trials", str(args.trials))
    overview.add_row("Acquisition pool", f"{len(tuning_pool_inputs)} sentences")
    overview.add_row("Validation", f"{len(validation_examples)} sentences")
    overview.add_row("Test split", "untouched")
    overview.add_row("Device", str(device))
    overview.add_row("Output", str(study_root))
    CONSOLE.print(overview)

    objective = _make_objective(
        base_config=base_config,
        seed_examples=seed_examples,
        tuning_pool_inputs=tuning_pool_inputs,
        tuning_pool_gold=tuning_pool_gold,
        validation_examples=validation_examples,
        device=device,
        study_root=study_root,
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )
    _write_study_outputs(study, study_root, base_config)
    CONSOLE.print(
        f"[bold green]Best {OBJECTIVE_NAME}:[/] {study.best_value:.6f} "
        f"(trial {study.best_trial.number})"
    )
    CONSOLE.print(f"[bold green]Best config:[/] {study_root / 'best_config.json'}")


if __name__ == "__main__":
    main()
