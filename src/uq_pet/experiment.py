"""Execute one experiment from the interactive notebook and persist its run."""

from collections.abc import Callable, Collection
from pathlib import Path

from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.config import ExperimentConfig
from uq_pet.data_prep import DATASET_TAGS, load_splits
from uq_pet.paths import RESULTS_DIR
from uq_pet.token_model import get_device, gold_tags, resolve_precision


def execute_experiment(
    config: ExperimentConfig,
    *,
    results_dir: Path = RESULTS_DIR,
    progress_callback: Callable[[list[dict]], None] | None = None,
    snapshot_rounds: Collection[int] = (),
) -> tuple[dict, dict, list[dict], list[dict], Path]:
    """Load the dataset, run both arms, persist the run, and return display-ready records.

    ``snapshot_rounds`` saves the uncertainty arm's model after those rounds under
    ``results_dir / "snapshots"``, next to the run.
    """
    device = get_device()
    effective_precision = resolve_precision(config.precision, device)
    (seed_examples, pool_inputs, pool_gold, test_examples), _ = load_splits(
        config.dataset, config.dataset_percent
    )
    results, selections, predictions = run_active_learning(
        seed_examples,
        pool_inputs,
        pool_gold,
        test_examples,
        **config.active_learning_kwargs(),
        device=device,
        snapshot_dir=results_dir / "snapshots",
        snapshot_rounds=snapshot_rounds,
        progress_callback=progress_callback,
    )

    first_result = results[0]
    derived_config = {
        "scoreable_pool_tokens": first_result["scoreable_pool_tokens"],
        "token_budget": first_result["token_budget"],
        "rounds": first_result["total_rounds"],
        "effective_precision": effective_precision,
        "effective_pool_percent": (
            100 * first_result["token_budget"] / first_result["scoreable_pool_tokens"]
        ),
    }
    run_config = {**config.resolved_dict(), **derived_config, "evaluation_split": "test"}
    if snapshot_rounds:
        run_config["snapshot_rounds"] = sorted(snapshot_rounds)
    dataset_summary = {
        "mode": "experiment",
        "dataset": config.dataset,
        "dataset_percent": config.dataset_percent,
        "seed_sentences": len(seed_examples),
        "pool_sentences": len(pool_inputs),
        "pool_tokens": len(pool_gold),
        "scoreable_pool_tokens": derived_config["scoreable_pool_tokens"],
        "acquisition_budget_tokens": derived_config["token_budget"],
        "acquisition_rounds": derived_config["rounds"],
        "effective_pool_percent": derived_config["effective_pool_percent"],
        "test_sentences": len(test_examples),
        "device": str(device),
    }
    run_dir = write_run(
        run_config,
        results,
        selections,
        results_dir=results_dir,
        gold=gold_tags(test_examples, DATASET_TAGS[config.dataset]),
        predictions=predictions,
    )
    return run_config, dataset_summary, results, selections, run_dir
