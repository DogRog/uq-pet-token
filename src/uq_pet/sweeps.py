"""What the search and supervised sweeps share.

Both sample learning rates from one log-uniform range and save the same files, so an
interrupted sweep resumes its saved plan and never mixes runs from two plans: `plan.json`
in the sweep directory, and `completed.json` or `failure.json` in each run's slot.
"""

import json
import math
import operator
import random
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from datasets.utils import logging as datasets_logging
from transformers.utils import logging as transformers_logging

LEARNING_RATE_BOUNDS = (1e-5, 5e-4)


def sample_learning_rate(rng: random.Random) -> float:
    """Draw a log-uniform learning rate within the shared bounds."""
    return math.exp(rng.uniform(*(math.log(bound) for bound in LEARNING_RATE_BOUNDS)))


def learning_rate_space() -> dict:
    """Describe the learning-rate distribution for a saved plan's search space."""
    return {
        "distribution": "log_uniform",
        "low": LEARNING_RATE_BOUNDS[0],
        "high": LEARNING_RATE_BOUNDS[1],
    }


def write_json(path: Path, value: dict) -> None:
    """Publish metadata atomically so an interrupted write is never considered complete."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def preserve_json(path: Path, value: dict) -> None:
    """Refuse to mix scientific plans, dataset splits, or frozen winners across resumes."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Saved {path.name} differs; choose a new sweep_name")
    else:
        write_json(path, value)


def open_sweep(
    root: Path,
    plan: dict,
    *,
    same_plan: Callable[[dict, dict], bool] = operator.eq,
) -> dict:
    """Return the saved plan when it matches `plan`, or save `plan` in a new sweep directory.

    The saved plan stays authoritative on resume. A sweep's budget is fixed, so a different
    plan needs a new sweep name rather than more runs in the same directory.
    """
    plan_path = root / "plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if not same_plan(saved, plan):
            raise ValueError("Existing sweep has a different plan; choose a new sweep_name")
        return saved
    if root.exists() and any(root.iterdir()):
        raise ValueError("Sweep directory is nonempty but has no plan; choose a new sweep_name")
    root.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    return plan


def completed_run(root: Path, slot: Path) -> dict | None:
    marker = slot / "completed.json"
    if not marker.exists():
        return None
    completed = json.loads(marker.read_text())
    for name in ("config.json", "results.csv", "selections.json"):
        if not (root / completed["run_dir"] / name).is_file():
            raise ValueError(f"Completed run is missing {name}: {marker}")
    return completed


def run_status(root: Path, slot: Path) -> dict:
    completed = completed_run(root, slot)
    if completed is not None:
        return {**completed, "status": "complete"}
    if (slot / "failure.json").exists():
        return {**json.loads((slot / "failure.json").read_text()), "status": "failed"}
    return {"status": "pending"}


def mark_complete(root: Path, slot: Path, run_dir: Path, stats: dict) -> None:
    """Point the slot at its saved run, written last so only finished runs count."""
    write_json(slot / "completed.json", {"run_dir": str(run_dir.relative_to(root)), **stats})
    (slot / "failure.json").unlink(missing_ok=True)


def record_failure(slot: Path, error: BaseException) -> None:
    write_json(slot / "failure.json", {"error": f"{type(error).__name__}: {error}"})


@contextmanager
def quiet_library_progress():
    """Hide transformers and datasets progress bars, restoring them afterwards."""
    model_bars = transformers_logging.is_progress_bar_enabled()
    data_bars = datasets_logging.is_progress_bar_enabled()
    try:
        transformers_logging.disable_progress_bar()
        datasets_logging.disable_progress_bar()
        yield
    finally:
        if model_bars:
            transformers_logging.enable_progress_bar()
        if data_bars:
            datasets_logging.enable_progress_bar()
