#!/usr/bin/env python3
"""Search PET token-UQ configurations with Optuna TPE, grid, or random sampling."""

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import optuna
from datasets.utils import logging as datasets_logging
from pydantic import Field, field_validator
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from transformers.utils import logging as transformers_logging

from uq_pet import active_learning
from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.experiment import ExperimentConfig
from uq_pet.pet_data import RESULTS_DIR, TokenKey, download_pet_ner, load_pet_splits
from uq_pet.token_model import UQ_METRICS, get_device

DEFAULT_STUDIES_DIR = RESULTS_DIR / "optuna"
OBJECTIVE_NAME = "mean_validation_entity_f1_gap_auc"
CONSOLE = Console()


@contextmanager
def _quiet_search_output():
    """Keep routine logs off the progress line and restore settings on every exit."""
    quiet = active_learning.CONSOLE.quiet
    verbosity = optuna.logging.get_verbosity()
    model_bars = transformers_logging.is_progress_bar_enabled()
    data_bars = datasets_logging.is_progress_bar_enabled()
    try:
        active_learning.CONSOLE.quiet = True
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        transformers_logging.disable_progress_bar()
        datasets_logging.disable_progress_bar()
        yield
    finally:
        active_learning.CONSOLE.quiet = quiet
        optuna.logging.set_verbosity(verbosity)
        if model_bars:
            transformers_logging.enable_progress_bar()
        if data_bars:
            datasets_logging.enable_progress_bar()


SEARCH_SPACE = {
    "uq_metric": UQ_METRICS,
    "k": (8, 16, 32),
    "bootstrap_epochs": (10, 20, 30),
    "update_passes": (1, 2, 4),
    "learning_rate": (2e-5, 3e-5, 5e-5),
    "batch_size": (4, 8, 16),
    "replay_ratio": (0, 0.5, 1.0, 2.0),
    "weight_decay": (0.0, 0.01),
}


def make_tuning_split(
    pool_inputs: list[dict],
    pool_gold: Mapping[TokenKey, int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], dict[TokenKey, int], list[dict]]:
    """Hold out pool sentences for tuning and reindex the remaining acquisition pool."""
    if len(pool_inputs) < 2:
        raise ValueError("tuning requires at least two pool sentences")
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    validation_count = round(len(pool_inputs) * validation_fraction)
    validation_count = min(len(pool_inputs) - 1, max(1, validation_count))
    validation_indices = set(random.Random(seed).sample(range(len(pool_inputs)), validation_count))

    tuning_pool_inputs: list[dict] = []
    tuning_pool_gold: dict[TokenKey, int] = {}
    validation_examples: list[dict] = []
    for old_pool_idx, example in enumerate(pool_inputs):
        labels = [pool_gold[(old_pool_idx, word_idx)] for word_idx in range(len(example["tokens"]))]
        if old_pool_idx in validation_indices:
            validation_examples.append(
                {
                    "document_name": example["document_name"],
                    "sentence_id": example["sentence_id"],
                    "tokens": example["tokens"],
                    "ner_tags": labels,
                }
            )
            continue

        new_pool_idx = len(tuning_pool_inputs)
        tuning_pool_inputs.append(
            {
                **example,
                "pool_idx": new_pool_idx,
            }
        )
        tuning_pool_gold.update(
            {(new_pool_idx, word_idx): label for word_idx, label in enumerate(labels)}
        )

    return tuning_pool_inputs, tuning_pool_gold, validation_examples


def suggest_search_config(trial) -> dict:
    """Sample one configuration from the shared discrete search space."""
    return {
        name: trial.suggest_categorical(name, list(values)) for name, values in SEARCH_SPACE.items()
    }


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


class SearchConfig(ExperimentConfig):
    """All experiment and launch settings for one Optuna search invocation."""

    trials: int = Field(ge=1)
    sampler: Literal["tpe", "grid", "random"] = "tpe"
    study_name: str = "bert-token-uq"
    studies_dir: Path = DEFAULT_STUDIES_DIR
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    validation_seed: int = Field(default=1729, ge=0)
    sampler_seed: int = Field(default=0, ge=0)
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("study_name")
    @classmethod
    def validate_study_name(cls, value):
        if not _slug(value):
            raise ValueError("study_name must contain at least one letter or number")
        return value

    def experiment_config(self) -> ExperimentConfig:
        """Strip launch settings and disable W&B for every search trial."""
        return ExperimentConfig.model_validate(
            {
                **self.model_dump(include=set(ExperimentConfig.model_fields)),
                "wandb_enabled": False,
                "wandb_run_name": "",
            }
        )


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Use the same categorical search space for all three Optuna samplers."""
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "grid":
        return optuna.samplers.GridSampler(SEARCH_SPACE, seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler: {name}")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _study_context(
    base_config: ExperimentConfig,
    *,
    validation_fraction: float,
    validation_seed: int,
    sampler_seed: int,
    sampler: str = "tpe",
) -> dict:
    # Scheduling does not change the study's scientific settings or resume identity.
    fixed_config = base_config.model_dump(exclude={*SEARCH_SPACE, "seed_workers"})
    return {
        "objective": OBJECTIVE_NAME,
        "acquisition_schedule": "partial_final_round",
        "fixed_config": fixed_config,
        "search_space": {key: list(values) for key, values in SEARCH_SPACE.items()},
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "sampler_seed": sampler_seed,
        "sampler": sampler,
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
        {**base_config.model_dump(), **study.best_trial.params}
    )
    (study_root / "best_config.json").write_text(json.dumps(best_config.model_dump(), indent=2))
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "objective": OBJECTIVE_NAME,
        "context": study.user_attrs.get("context"),
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
    progress: Progress,
    task_id: int,
):
    def objective(trial: optuna.Trial) -> float:
        completed_trials = progress.tasks[task_id].completed
        trial_label = f"Trial {int(completed_trials) + 1}/{int(progress.tasks[task_id].total)}"
        progress.update(task_id, description=f"{trial_label} · loading / bootstrap")
        config = ExperimentConfig.model_validate(
            {**base_config.model_dump(), **suggest_search_config(trial)}
        )

        def update_progress(rows):
            row = rows[-1]
            # Two rows per complete round, including each seed's baseline.
            fraction = len(rows) / (2 * (row["total_rounds"] + 1) * len(config.model_seeds))
            progress.update(
                task_id,
                completed=completed_trials + fraction,
                description=(
                    f"{trial_label} · seed {row['seed']} · "
                    f"round {row['round']}/{row['total_rounds']}"
                ),
            )

        results, selections = run_active_learning(
            seed_examples,
            tuning_pool_inputs,
            tuning_pool_gold,
            validation_examples,
            **config.active_learning_kwargs(),
            device=device,
            progress_callback=update_progress,
        )
        progress.update(task_id, description=f"{trial_label} · saving results")
        value = mean_entity_f1_gap_auc(results)
        trial_root = study_root / "trials" / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True)
        run_config = {
            **config.resolved_dict(),
            "mode": "optuna_tuning",
            "optuna_trial": trial.number,
            "search_context": trial.study.user_attrs.get("context"),
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
        progress.update(task_id, completed=completed_trials + 1)
        return value

    return objective


def configure_parser(parser: argparse.ArgumentParser) -> None:
    config_source = parser.add_mutually_exclusive_group()
    config_source.add_argument(
        "--config", type=Path, help="JSON file containing all search and experiment settings."
    )
    config_source.add_argument(
        "--config-json",
        help="Inline JSON with the same settings as --config; tuned fields are overwritten.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=argparse.SUPPRESS,
        help="Override additional trials; trials must be supplied in JSON or with this flag.",
    )
    parser.add_argument(
        "--study-name",
        default=argparse.SUPPRESS,
        help="Persistent Optuna study name (default: bert-token-uq).",
    )
    parser.add_argument(
        "--studies-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Directory containing study databases and trial artifacts.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of pool sentences withheld from acquisition for validation.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Local RNG seed for the nested validation split.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Seed for the selected Optuna sampler.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional study timeout in seconds.",
    )


def load_search_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> SearchConfig:
    """Validate JSON and explicit CLI overrides before any experiment side effects."""
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
        return SearchConfig.model_validate(settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))


@_quiet_search_output()
def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    search_config = load_search_config(args, parser)
    base_config = search_config.experiment_config()

    study_root = search_config.studies_dir / _slug(search_config.study_name)
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "trials").mkdir(exist_ok=True)
    storage = f"sqlite:///{study_root / 'study.db'}"
    sampler = make_sampler(search_config.sampler, search_config.sampler_seed)
    study = optuna.create_study(
        study_name=search_config.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    context = _study_context(
        base_config,
        validation_fraction=search_config.validation_fraction,
        validation_seed=search_config.validation_seed,
        sampler_seed=search_config.sampler_seed,
        sampler=search_config.sampler,
    )
    fingerprint = _context_fingerprint(context)
    existing_fingerprint = study.user_attrs.get("context_fingerprint")
    # Legacy studies used TPE implicitly; verify their stored fingerprint before migration.
    existing_context = study.user_attrs.get("context")
    if (
        existing_context is not None
        and "sampler" not in existing_context
        and existing_fingerprint == _context_fingerprint(existing_context)
        and {**existing_context, "sampler": "tpe"} == context
    ):
        existing_fingerprint = fingerprint
    if (existing_fingerprint is not None and existing_fingerprint != fingerprint) or (
        existing_fingerprint is None and study.trials
    ):
        parser.error(
            "the existing study uses different fixed settings, search space, acquisition schedule, "
            "sampler, or validation split; choose a new --study-name"
        )
    study.set_user_attr("context", context)
    study.set_user_attr("context_fingerprint", fingerprint)
    (study_root / "search_config.json").write_text(search_config.model_dump_json(indent=2))

    if (
        isinstance(sampler, optuna.samplers.GridSampler)
        and sampler.is_exhausted(study)
        and not any(trial.state == optuna.trial.TrialState.WAITING for trial in study.trials)
    ):
        _write_study_outputs(study, study_root, base_config)
        CONSOLE.print("Grid search is already exhausted; no additional trials to run.")
        return

    data_path = download_pet_ner()
    seed_examples, pool_inputs, pool_gold, _test_examples = load_pet_splits(data_path)
    tuning_pool_inputs, tuning_pool_gold, validation_examples = make_tuning_split(
        pool_inputs,
        pool_gold,
        validation_fraction=search_config.validation_fraction,
        seed=search_config.validation_seed,
    )
    device = get_device()

    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan", justify="right")
    overview.add_column()
    overview.add_row("Study", search_config.study_name)
    overview.add_row("Sampler", search_config.sampler)
    overview.add_row("Additional trials", str(search_config.trials))
    overview.add_row("Acquisition pool", f"{len(tuning_pool_inputs)} sentences")
    overview.add_row("Validation", f"{len(validation_examples)} sentences")
    overview.add_row("Test split", "untouched")
    overview.add_row("Device", str(device))
    overview.add_row("Output", str(study_root))

    with Progress(console=CONSOLE, transient=True) as progress:
        task_id = progress.add_task("Preparing search", total=search_config.trials)
        objective = _make_objective(
            base_config=base_config,
            seed_examples=seed_examples,
            tuning_pool_inputs=tuning_pool_inputs,
            tuning_pool_gold=tuning_pool_gold,
            validation_examples=validation_examples,
            device=device,
            study_root=study_root,
            progress=progress,
            task_id=task_id,
        )

        try:
            study.optimize(
                objective,
                n_trials=search_config.trials,
                timeout=search_config.timeout,
                gc_after_trial=True,
            )
        finally:
            _write_study_outputs(study, study_root, base_config)
    overview.add_row(
        "Trials completed this invocation", str(int(progress.tasks[task_id].completed))
    )
    CONSOLE.print(overview)
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        CONSOLE.print("No completed trials; no best configuration is available yet.")
        return
    CONSOLE.print(
        f"[bold green]Best {OBJECTIVE_NAME}:[/] {study.best_value:.6f} "
        f"(trial {study.best_trial.number})"
    )
    CONSOLE.print(f"[bold green]Best config:[/] {study_root / 'best_config.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
