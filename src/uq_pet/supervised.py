"""Fully supervised sentence training: validation random search, then fixed-epoch test runs.

Following Vacareanu et al. (2024), every selected sentence is fully labelled and a
fresh pretrained model is trained for a fixed number of epochs, supervising first
subwords only. Hyperparameters are sampled like the other sweeps and chosen on the
tuning holdout; the test split is used only by the final runs. At 100% of pool
sentences this is the upper bound for active learning.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import torch
from datasets.utils import logging as datasets_logging
from dotenv import load_dotenv
from rich.console import Console
from transformers.utils import logging as transformers_logging

from uq_pet.active_learning import full_sentence_items, write_run
from uq_pet.config import SupervisedConfig, omit_default_dataset
from uq_pet.pet_data import DATASET_TAGS, PROJECT_ROOT, TokenKey, load_splits, split_tuning_pool
from uq_pet.search import (
    LEARNING_RATE_BOUNDS,
    preserve_json,
    run_status,
    sample_learning_rate,
    write_json,
)
from uq_pet.token_model import (
    evaluate_model,
    get_device,
    load_token_classifier,
    resolve_precision,
    scoreable_token_keys,
    set_seed,
    train_items,
)
from uq_pet.utils.wandb_logging import (
    login_quietly,
    require_wandb_credentials,
    wandb_supervised_logging,
)
from uq_pet.utils.wandb_tuning import ensure_supervised_sweep, publish_supervised_trial

CONSOLE = Console()
TUNED_FIELDS = ("learning_rate", "batch_size", "weight_decay")
SUPERVISED_ARM = "supervised"
OBJECTIVE = "validation_entity_f1"


def sample_plan(config: SupervisedConfig) -> dict:
    """Freeze sampled trials, validation split, and objective before training."""
    rng = random.Random(config.sampler_seed)
    sampled = []
    for _ in range(config.num_configs):
        parameters = {
            "batch_size": rng.choice(config.batch_sizes),
            "weight_decay": rng.choice(config.weight_decays),
        }
        parameters["learning_rate"] = sample_learning_rate(rng)
        sampled.append(parameters)
    return {
        "version": 2,
        "mode": "supervised_random",
        "sampling": "independent_categorical_and_log_uniform",
        "sampler_seed": config.sampler_seed,
        "num_configs": config.num_configs,
        "fixed_config": omit_default_dataset(
            config.model_dump(
                mode="json",
                include={
                    "checkpoint",
                    "dataset",
                    "dataset_percent",
                    "model_seeds",
                    "epochs",
                    "max_length",
                    "score_batch_size",
                },
            )
        ),
        "search_space": {
            "batch_size": config.batch_sizes,
            "weight_decay": config.weight_decays,
            "learning_rate": {
                "distribution": "log_uniform",
                "low": LEARNING_RATE_BOUNDS[0],
                "high": LEARNING_RATE_BOUNDS[1],
            },
        },
        "configurations": [
            {"config_id": f"config_{index:04d}", "parameters": parameters}
            for index, parameters in enumerate(sampled)
        ],
        "validation_sentences": config.validation_sentences,
        "validation_seed": config.validation_seed,
        "objective": OBJECTIVE,
        "tie_break": "config_id_ascending",
        "tuning_evaluation_split": "validation",
    }


def sentence_order(n_pool: int, seed: int) -> list[int]:
    """Order pool sentences without labels; each budget takes a prefix, so budgets nest."""
    order = list(range(n_pool))
    random.Random(seed).shuffle(order)
    return order


def pool_sentence_items(
    pool_indices: list[int],
    pool_inputs: list[dict],
    pool_gold: dict[TokenKey, int],
    scoreable: set[TokenKey],
) -> list[dict]:
    """Fully label selected pool sentences, excluding words lost to truncation."""
    items = []
    for pool_idx in pool_indices:
        tokens = pool_inputs[pool_idx]["tokens"]
        targets = {
            word_idx: pool_gold[(pool_idx, word_idx)]
            for word_idx in range(len(tokens))
            if (pool_idx, word_idx) in scoreable
        }
        if targets:
            items.append({"tokens": tokens, "targets": targets})
    return items


def train_and_evaluate(
    seed_examples: list[dict],
    pool_inputs: list[dict],
    pool_gold: dict[TokenKey, int],
    pool_indices: list[int],
    eval_examples: list[dict],
    *,
    checkpoint: str,
    labels: list[str],
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    score_batch_size: int,
    max_length: int,
    device: torch.device,
    precision: str,
) -> dict:
    """Train a fresh model on seed plus selected pool sentences, then evaluate once."""
    set_seed(seed)
    model, tokenizer = load_token_classifier(checkpoint, device, labels=labels)
    scoreable = scoreable_token_keys(
        tokenizer, pool_inputs, max_length=max_length, batch_size=score_batch_size
    )
    items = [
        *full_sentence_items(seed_examples),
        *pool_sentence_items(pool_indices, pool_inputs, pool_gold, scoreable),
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = train_items(
        model,
        optimizer,
        tokenizer,
        items,
        passes=epochs,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        seed=seed,
        tokenization_cache={},
        precision=precision,
    )
    metrics = evaluate_model(
        model,
        tokenizer,
        eval_examples,
        labels=labels,
        max_length=max_length,
        batch_size=score_batch_size,
        device=device,
        precision=precision,
    )
    selected = set(pool_indices)
    n_acquired = sum(pool_idx in selected for pool_idx, _ in scoreable)
    del model, optimizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "n_sentences": len(pool_indices),
        "n_acquired": n_acquired,
        "percent_acquired": 100 * n_acquired / len(scoreable),
        "scoreable_pool_tokens": len(scoreable),
        **metrics,
        "train_loss": loss,
    }


def _init_seed_worker(cpu_threads: int) -> None:
    torch.set_num_threads(cpu_threads)
    transformers_logging.disable_progress_bar()


def seed_executor(seed_workers: int) -> ProcessPoolExecutor | None:
    """Spawn persistent seed workers that share the device; one worker trains in-process."""
    if seed_workers <= 1:
        return None
    return ProcessPoolExecutor(
        seed_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_seed_worker,
        initargs=(max(1, torch.get_num_threads() // seed_workers),),
    )


def train_seeds(
    jobs: dict[int, tuple[tuple, dict]], executor: Executor | None
) -> Iterator[tuple[int, dict]]:
    """Yield each seed's metrics as it finishes; without an executor, train in seed order."""
    if executor is None:
        for seed, (args, kwargs) in jobs.items():
            yield seed, train_and_evaluate(*args, **kwargs)
        return
    futures = {
        executor.submit(train_and_evaluate, *args, **kwargs): seed
        for seed, (args, kwargs) in jobs.items()
    }
    try:
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        for future in futures:
            future.cancel()


def train_logged_seeds(
    jobs: dict[int, tuple[tuple, dict]], executor: Executor | None, log, label: str
) -> dict[int, dict]:
    """Train every seed, logging each as it finishes behind one transient status line."""
    finished = {}
    with CONSOLE.status(f"{label} · 0/{len(jobs)} seeds") as status:
        for seed, metrics in train_seeds(jobs, executor):
            finished[seed] = metrics
            log({"seed": seed, **metrics})
            status.update(f"{label} · {len(finished)}/{len(jobs)} seeds")
    return finished


def format_parameters(parameters: dict) -> str:
    return (
        f"lr {parameters['learning_rate']:.2e} · batch {parameters['batch_size']} · "
        f"wd {parameters['weight_decay']:g}"
    )


def seed_line(rows: list[dict], stats: dict) -> str:
    """Seed mean ± SD, then each seed's entity F1 in seed order."""
    seeds = " ".join(f"{row['entity_f1']:.3f}" for row in rows)
    return (
        f"[bold]{stats['mean_entity_f1']:.4f}[/] ± {stats['std_entity_f1']:.4f} [dim]({seeds})[/]"
    )


def training_settings(config: SupervisedConfig, device: torch.device, precision: str) -> dict:
    return {
        "checkpoint": config.checkpoint,
        "labels": DATASET_TAGS[config.dataset],
        "epochs": config.epochs,
        "score_batch_size": config.score_batch_size,
        "max_length": config.max_length,
        "device": device,
        "precision": precision,
    }


def seed_statistics(rows: list[dict], field: str = "entity_f1") -> dict:
    values = [float(row[field]) for row in rows]
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    return {f"mean_{field}": mean, f"std_{field}": std}


def percent_slot(root: Path, percent: float) -> Path:
    return root / "test" / f"pool_{percent:g}pct"


def write_summary(root: Path, plan: dict) -> dict:
    """Report every planned trial, the frozen winner, and every saved test budget."""
    trials = [
        {**entry, **run_status(root, root / "tuning" / entry["config_id"])}
        for entry in plan["configurations"]
    ]
    selection_path = root / "selection.json"
    test_runs = [
        {"slot": str(slot.relative_to(root)), **run_status(root, slot)}
        for slot in sorted((root / "test").glob("pool_*pct"))
    ]
    summary = {
        "mode": plan["mode"],
        "objective": plan["objective"],
        "tuning_complete": all(row["status"] == "complete" for row in trials)
        and selection_path.is_file(),
        "selection": json.loads(selection_path.read_text()) if selection_path.is_file() else None,
        "trials": trials,
        "test_runs": test_runs,
    }
    write_json(root / "summary.json", summary)
    return summary


def _record_failure(slot: Path, error: BaseException) -> None:
    write_json(slot / "failure.json", {"error": f"{type(error).__name__}: {error}"})


def tune(
    config: SupervisedConfig,
    root: Path,
    plan: dict,
    splits: tuple,
    *,
    device: torch.device,
    precision: str,
    executor: Executor | None = None,
    publication: tuple | None = None,
) -> dict:
    """Random-search on the validation holdout and freeze the best configuration."""
    seed_examples, pool_inputs, pool_gold, test_examples = splits
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
            "seed_sentences": len(seed_examples),
            "tuning_pool_sentences": len(tune_pool),
            "validation_sentences": len(validation_examples),
            "final_pool_sentences": len(pool_inputs),
            "test_sentences": len(test_examples),
        },
    )
    training = training_settings(config, device, precision)
    for position, entry in enumerate(plan["configurations"], start=1):
        slot = root / "tuning" / entry["config_id"]
        if run_status(root, slot)["status"] == "complete":
            continue
        slot.mkdir(parents=True, exist_ok=True)
        label = f"{entry['config_id']} ({position}/{len(plan['configurations'])})"
        metadata = {
            **plan["fixed_config"],
            **entry["parameters"],
            "effective_precision": precision,
            "mode": plan["mode"],
            "config_id": entry["config_id"],
            "evaluation_split": "validation",
            "seed_sentences": len(seed_examples),
            "pool_sentences": len(tune_pool),
            "validation_sentences": len(validation_examples),
        }
        try:
            with wandb_supervised_logging(
                config,
                f"{config.sweep_name}-{entry['config_id']}",
                slot,
                metadata,
                job_type="supervised_tuning",
            ) as log:
                arguments = (
                    seed_examples,
                    tune_pool,
                    tune_gold,
                    list(range(len(tune_pool))),
                    validation_examples,
                )
                jobs = {
                    seed: (arguments, {**training, **entry["parameters"], "seed": seed})
                    for seed in config.model_seeds
                }
                finished = train_logged_seeds(jobs, executor, log, f"tune {label}")
                rows = [
                    {"seed": seed, "arm": SUPERVISED_ARM, **finished[seed]}
                    for seed in config.model_seeds
                ]
                stats = seed_statistics(rows)
                log(stats)
            run_dir = write_run(metadata, rows, [], results_dir=slot)
            write_json(
                slot / "completed.json",
                {
                    "run_dir": str(run_dir.relative_to(root)),
                    "score": stats["mean_entity_f1"],
                    **stats,
                },
            )
            (slot / "failure.json").unlink(missing_ok=True)
            CONSOLE.print(
                f"[bold cyan]tune {label}[/] · {format_parameters(entry['parameters'])} · "
                f"validation entity F1 {seed_line(rows, stats)}"
            )
        except BaseException as error:
            _record_failure(slot, error)
            raise
        finally:
            write_summary(root, plan)
        if publication is not None:
            publish_supervised_trial(config, root, entry, publication)

    trials = write_summary(root, plan)["trials"]
    best = min(trials, key=lambda row: (-row["score"], row["config_id"]))
    frozen = {**plan["fixed_config"], **best["parameters"]}
    preserve_json(root / "best_config.json", frozen)
    preserve_json(
        root / "selection.json",
        {
            "config_id": best["config_id"],
            "objective": OBJECTIVE,
            "validation_score": best["score"],
            "selection_split": "validation",
        },
    )
    CONSOLE.print(
        f"[bold green]Frozen winner[/] {best['config_id']} · "
        f"{format_parameters(best['parameters'])} · validation entity F1 {best['score']:.4f}"
    )
    write_summary(root, plan)
    return frozen


def run_test(
    config: SupervisedConfig,
    root: Path,
    plan: dict,
    splits: tuple,
    *,
    device: torch.device,
    precision: str,
    executor: Executor | None = None,
) -> None:
    """Train the frozen winner on each pool-sentence budget and evaluate on test."""
    winner_path = root / "best_config.json"
    if not winner_path.is_file():
        raise ValueError("The test stage requires best_config.json; run the tune stage first")
    winner = json.loads(winner_path.read_text())
    parameters = {field: winner[field] for field in TUNED_FIELDS}
    seed_examples, pool_inputs, pool_gold, test_examples = splits
    training = training_settings(config, device, precision)
    for percent in config.sentence_percents:
        slot = percent_slot(root, percent)
        if run_status(root, slot)["status"] == "complete":
            continue
        slot.mkdir(parents=True, exist_ok=True)
        n_sentences = round(percent / 100 * len(pool_inputs))
        metadata = {
            **winner,
            "effective_precision": precision,
            "mode": "supervised_test",
            "evaluation_split": "test",
            "sentence_percent": percent,
            "selected_pool_sentences": n_sentences,
            "seed_sentences": len(seed_examples),
            "pool_sentences": len(pool_inputs),
            "test_sentences": len(test_examples),
            "selected_config_id": json.loads((root / "selection.json").read_text())["config_id"],
        }
        try:
            with wandb_supervised_logging(
                config,
                f"{config.sweep_name}-test-{percent:g}pct",
                slot,
                metadata,
                job_type="supervised_test",
            ) as log:
                selected = {
                    seed: sorted(sentence_order(len(pool_inputs), seed)[:n_sentences])
                    for seed in config.model_seeds
                }
                jobs = {
                    seed: (
                        (seed_examples, pool_inputs, pool_gold, indices, test_examples),
                        {**training, **parameters, "seed": seed},
                    )
                    for seed, indices in selected.items()
                }
                finished = train_logged_seeds(
                    jobs, executor, log, f"test {percent:g}% pool sentences"
                )
                rows = [
                    {
                        "seed": seed,
                        "arm": SUPERVISED_ARM,
                        "sentence_percent": percent,
                        **finished[seed],
                    }
                    for seed in config.model_seeds
                ]
                selections = [
                    {
                        "seed": seed,
                        "arm": SUPERVISED_ARM,
                        "sentence_percent": percent,
                        "pool_idx": pool_idx,
                        "document_name": pool_inputs[pool_idx]["document_name"],
                        "sentence_id": pool_inputs[pool_idx]["sentence_id"],
                    }
                    for seed in config.model_seeds
                    for pool_idx in selected[seed]
                ]
                stats = seed_statistics(rows)
                log(stats)
            run_dir = write_run(metadata, rows, selections, results_dir=slot)
            write_json(
                slot / "completed.json", {"run_dir": str(run_dir.relative_to(root)), **stats}
            )
            (slot / "failure.json").unlink(missing_ok=True)
            CONSOLE.print(
                f"[bold magenta]test {percent:g}% pool sentences[/] · "
                f"test entity F1 {seed_line(rows, stats)}"
            )
        except BaseException as error:
            _record_failure(slot, error)
            raise
        finally:
            write_summary(root, plan)


def load_test_results(
    sweeps_dir: Path, checkpoint: str, dataset: str = "pet", dataset_percent: float = 100.0
) -> tuple[str, pl.DataFrame] | None:
    """Return the first saved supervised sweep for a checkpoint and data pool, with test rows."""
    for sweep in sorted(path for path in Path(sweeps_dir).glob("*") if path.is_dir()):
        frames = []
        for marker in sorted(sweep.glob("test/*/completed.json")):
            run_dir = sweep / json.loads(marker.read_text())["run_dir"]
            if not (run_dir / "results.csv").is_file() or not (run_dir / "config.json").is_file():
                continue
            saved = json.loads((run_dir / "config.json").read_text())
            if (
                saved.get("checkpoint") == checkpoint
                and saved.get("dataset", "pet") == dataset
                and saved.get("dataset_percent", 100.0) == dataset_percent
            ):
                frames.append(pl.read_csv(run_dir / "results.csv"))
        if frames:
            return sweep.name, pl.concat(frames, how="vertical_relaxed")
    return None


def load_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> SupervisedConfig:
    try:
        payload = args.config.read_text() if args.config is not None else args.config_json
        settings = json.loads(payload)
        if not isinstance(settings, dict):
            raise ValueError("configuration must be a JSON object")
        if "seed_workers" in args:
            settings["seed_workers"] = args.seed_workers
        return SupervisedConfig.model_validate(settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = load_config(args, parser)
    device = get_device()
    plan = sample_plan(config)
    plan["effective_precision"] = resolve_precision(config.precision, device)
    if args.dry_run:
        print(
            json.dumps(
                {**plan, "sentence_percents": config.sentence_percents, "stage": args.stage},
                indent=2,
            )
        )
        return
    root = config.sweeps_dir / config.sweep_name
    plan_path = root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            parser.error("Existing supervised sweep has a different plan; choose a new sweep_name")
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("Sweep directory is nonempty but has no plan; choose a new sweep_name")
        root.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    write_json(root / "run_config.json", config.model_dump(mode="json"))
    write_summary(root, plan)
    if args.stage == "test" and not (root / "best_config.json").is_file():
        parser.error("The test stage requires best_config.json; run the tune stage first")
    publication = None
    if config.wandb_enabled:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        require_wandb_credentials(os.environ)
        if os.environ.get("WANDB_MODE", "").strip().lower() == "offline":
            CONSOLE.print("W&B offline: logging locally; no online sweep or chart is created.")
        else:
            login_quietly(os.environ["WANDB_API_KEY"])
            # Trials completed before the sweep existed are published on the next launch.
            publication = ensure_supervised_sweep(config, plan, root)
            for entry in plan["configurations"]:
                publish_supervised_trial(config, root, entry, publication)
            record = publication[1][publication[2]]
            CONSOLE.print(
                f"W&B {record['entity']}/{record['project']} · sweep {record['sweep_id']} · "
                f"chart {record['workspace_url']}",
                markup=False,
            )

    precision = plan["effective_precision"]
    seed_workers = min(config.seed_workers, len(config.model_seeds))
    CONSOLE.print(
        f"[bold cyan]Supervised {config.checkpoint}[/] · {len(plan['configurations'])} sampled "
        f"configurations × {len(config.model_seeds)} seeds ({seed_workers} concurrent) · "
        f"{config.epochs} epochs · {precision.upper()} on {device}"
    )
    splits, _ = load_splits(config.dataset, config.dataset_percent)
    model_bars = transformers_logging.is_progress_bar_enabled()
    data_bars = datasets_logging.is_progress_bar_enabled()
    transformers_logging.disable_progress_bar()
    datasets_logging.disable_progress_bar()
    executor = seed_executor(seed_workers)
    try:
        stage_settings = {"device": device, "precision": precision, "executor": executor}
        if args.stage in {"tune", "all"}:
            tune(config, root, plan, splits, **stage_settings, publication=publication)
        if args.stage in {"test", "all"}:
            run_test(config, root, plan, splits, **stage_settings)
    finally:
        if executor is not None:
            # Cancels queued seeds; running siblings finish their current seed.
            executor.shutdown(cancel_futures=True)
        if model_bars:
            transformers_logging.enable_progress_bar()
        if data_bars:
            datasets_logging.enable_progress_bar()
    CONSOLE.print(f"Supervised results: {root / 'summary.json'}")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON file with supervised settings.")
    source.add_argument("--config-json", help="Inline JSON with supervised settings.")
    parser.add_argument(
        "--stage",
        choices=("tune", "test", "all"),
        default="all",
        help="Random-search on validation (tune), evaluate the frozen winner on test (test), or both.",
    )
    parser.add_argument(
        "--seed-workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Concurrent seed processes on the device (default: config or 1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without loading data or models."
    )
