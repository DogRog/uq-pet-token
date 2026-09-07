"""Atomic round checkpoints and retry handling for interrupted searches."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import optuna
import torch

from uq_pet.pet_data import NER_TAGS


def checkpoint_signature(settings, seed, seed_examples, pool_inputs, evaluation_examples):
    payload = {
        "settings": settings,
        "seed": seed,
        "seed_examples": seed_examples,
        "pool_inputs": pool_inputs,
        "evaluation_examples": evaluation_examples,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_checkpoint(path, signature, pool_gold):
    if path is None or not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["version"] != 1 or saved["signature"] != signature:
        raise ValueError(f"checkpoint settings or data do not match: {path}")
    if any(
        NER_TAGS[pool_gold[(row["pool_idx"], row["word_idx"])]] != row["label"]
        for row in saved["selections"]
    ):
        raise ValueError(f"previously selected labels changed: {path}")
    return saved


def save_checkpoint(path: Path, state: dict) -> None:
    """Replace the last complete round only after its new checkpoint is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_round(path, signature, results, selections, models, optimizers):
    if path is None:
        return
    row = results[-1]
    complete = row["round"] == row["total_rounds"]
    save_checkpoint(
        path,
        {
            "version": 1,
            "signature": signature,
            "round": row["round"],
            "results": results,
            "selections": selections,
            "models": {}
            if complete
            else {arm: model.state_dict() for arm, model in models.items()},
            "optimizers": {}
            if complete
            else {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
        },
    )


def trial_checkpoint_dir(study_root: Path, trial) -> Path:
    number = trial.user_attrs.get("checkpoint_trial", trial.number)
    return study_root / "checkpoints" / f"trial_{number:04d}"


def resume_interrupted_trial(study: optuna.Study) -> None:
    """Retry the last interrupted trial with its original parameters and seed checkpoints."""
    if not study.trials:
        return
    trial = study.trials[-1]
    if trial.state not in {optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING}:
        return
    if "checkpoint_trial" not in trial.user_attrs:
        return  # Runs started before checkpoint support have no recoverable training state.
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.WAITING,
            params=trial.params,
            distributions=trial.distributions,
            system_attrs=trial.system_attrs,
            user_attrs={
                "checkpoint_trial": trial.user_attrs["checkpoint_trial"],
                "resumed_from_trial": trial.number,
            },
        )
    )


def prepare_trial_checkpoint(study_root: Path, trial: optuna.Trial) -> Path:
    previous = trial.user_attrs.get("resumed_from_trial")
    if (
        previous is not None
        and trial.study.trials[previous].state == optuna.trial.TrialState.RUNNING
    ):
        # GridSampler may stop the study here, so this runs inside optimize.
        trial.study.tell(previous, state=optuna.trial.TrialState.FAIL)
    trial.set_user_attr("checkpoint_trial", trial.user_attrs.get("checkpoint_trial", trial.number))
    return trial_checkpoint_dir(study_root, trial)


def release_trial_checkpoints(study_root: Path, study, trial) -> None:
    # Delete large tensors only after Optuna has committed the completed result.
    if trial.state == optuna.trial.TrialState.COMPLETE:
        shutil.rmtree(trial_checkpoint_dir(study_root, trial), ignore_errors=True)
