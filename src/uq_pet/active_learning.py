"""Sequential token acquisition, replay, orchestration, and run outputs."""

import copy
import csv
import json
import multiprocessing
import random
import traceback
from collections.abc import Callable
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from transformers.utils import logging as transformers_logging

from uq_pet.pet_data import NER_TAGS, RESULTS_DIR, TokenKey
from uq_pet.token_model import (
    UQ_METRICS,
    evaluate_model,
    load_token_classifier,
    prepare_inference_batches,
    resolve_precision,
    score_token_uncertainty,
    scoreable_token_keys,
    set_seed,
    train_items,
)

CONSOLE = Console()


def _round_progress_table(round_results: dict[str, dict], uq_metric: str) -> Table:
    """Render the named UQ method and random results as two equal-width columns."""
    arms = (
        ("uncertainty", uq_metric, "bold magenta"),
        ("random", "random", "bold blue"),
    )
    total_rounds = round_results["uncertainty"]["total_rounds"]
    token_budget = round_results["uncertainty"]["token_budget"]
    round_width = len(str(total_rounds))
    acquired_width = len(str(token_budget))

    table = Table.grid(expand=True, padding=(0, 3))
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    cells = []
    label_width = max(len(display_name) for _, display_name, _ in arms)
    for arm, display_name, arm_style in arms:
        row = round_results[arm]
        cells.append(
            f"[{arm_style}]{display_name:>{label_width}}[/]  "
            f"[dim]round[/] {row['round']:>{round_width}}/{total_rounds}  "
            f"[dim]acquired[/] {row['n_acquired']:>{acquired_width}}  "
            f"[dim]entity F1[/] [bold]{row['entity_f1']:.4f}[/]  "
            f"[dim]loss[/] {row['train_loss']:.4f}"
        )
    table.add_row(*cells)
    return table


def full_sentence_items(examples: list[dict]) -> list[dict]:
    """Create bootstrap items in which every original word is supervised."""
    return [
        {
            "tokens": example["tokens"],
            "targets": dict(enumerate(example["ner_tags"])),
        }
        for example in examples
    ]


def seed_replay_items(examples: list[dict]) -> list[dict]:
    """Represent seed supervision as equal-weight, one-word replay items."""
    return [
        {
            "key": ("seed", sentence_idx, word_idx),
            "tokens": example["tokens"],
            "targets": {word_idx: tag},
        }
        for sentence_idx, example in enumerate(examples)
        for word_idx, tag in enumerate(example["ner_tags"])
    ]


def reveal_pool_items(
    keys: list[TokenKey], pool_inputs: list[dict], pool_gold: dict[TokenKey, int]
) -> list[dict]:
    """Reveal selected labels and turn them into one-word masked training items."""
    return [
        {
            "key": ("pool", pool_idx, word_idx),
            "tokens": pool_inputs[pool_idx]["tokens"],
            "targets": {word_idx: pool_gold[(pool_idx, word_idx)]},
        }
        for pool_idx, word_idx in keys
    ]


def select_top_k(scores: dict[TokenKey, float], k: int) -> list[TokenKey]:
    if k < 1 or k > len(scores):
        raise ValueError(f"cannot select k={k} from {len(scores)} scored tokens")
    return sorted(scores, key=lambda key: (-scores[key], key))[:k]


def select_random(available: set[TokenKey], k: int, *, seed: int) -> list[TokenKey]:
    if k < 1 or k > len(available):
        raise ValueError(f"cannot select k={k} from {len(available)} available tokens")
    return random.Random(seed).sample(sorted(available), k)


def sample_replay(items: list[dict], k: int, ratio: float, *, seed: int) -> list[dict]:
    if ratio < 0:
        raise ValueError(f"replay ratio must be non-negative, got {ratio}")
    count = min(len(items), int(round(k * ratio)))
    if count == 0:
        return []
    indices = random.Random(seed).sample(range(len(items)), count)
    return [items[idx] for idx in indices]


def acquisition_schedule(scoreable_tokens: int, k: int, max_pool_percent: float) -> tuple[int, int]:
    """Return rounds and budget, allowing a smaller final acquisition round."""
    if scoreable_tokens < 1:
        raise ValueError("the scoreable pool must contain at least one token")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not 0 < max_pool_percent <= 100:
        raise ValueError(f"max_pool_percent must be in (0, 100], got {max_pool_percent}")

    requested_tokens = int(scoreable_tokens * max_pool_percent / 100)
    if requested_tokens < 1:
        raise ValueError(
            f"{max_pool_percent:g}% of {scoreable_tokens} scoreable tokens is fewer than one token"
        )
    rounds = (requested_tokens + k - 1) // k
    return rounds, requested_tokens


def _result_row(
    seed: int,
    arm: str,
    round_idx: int,
    acquired: int,
    metrics: dict,
    *,
    scoreable_tokens: int,
    token_budget: int,
    total_rounds: int,
) -> dict:
    return {
        "seed": seed,
        "arm": arm,
        "round": round_idx,
        "total_rounds": total_rounds,
        "n_acquired": acquired,
        "percent_acquired": 100 * acquired / scoreable_tokens,
        "scoreable_pool_tokens": scoreable_tokens,
        "token_budget": token_budget,
        **metrics,
    }


def run_active_learning(
    seed_examples: list[dict],
    pool_inputs: list[dict],
    pool_gold: dict[TokenKey, int],
    test_examples: list[dict],
    *,
    uq_metric: str,
    progress_callback: Callable[[list[dict]], None] | None = None,
    **settings,
) -> tuple[list[dict], list[dict]]:
    """Run one paired comparison using the same engine as all-metric sweeps."""
    comparisons = run_metric_comparisons(
        seed_examples,
        pool_inputs,
        pool_gold,
        test_examples,
        uq_metrics=[uq_metric],
        progress_callback=(
            (lambda metric, rows: progress_callback(rows))
            if progress_callback is not None
            else None
        ),
        **settings,
    )
    return comparisons[uq_metric]


def run_metric_comparisons(
    seed_examples: list[dict],
    pool_inputs: list[dict],
    pool_gold: dict[TokenKey, int],
    test_examples: list[dict],
    *,
    checkpoint: str,
    model_seeds: list[int],
    uq_metrics: list[str],
    k: int,
    max_pool_percent: float,
    bootstrap_epochs: int,
    update_passes: int,
    replay_ratio: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    score_batch_size: int,
    max_length: int,
    device: torch.device,
    seed_workers: int = 1,
    precision: str = "auto",
    progress_callback: Callable[[str, list[dict]], None] | None = None,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Share one bootstrap and random trajectory per seed across the requested metrics.

    Metrics run sequentially, keeping at most two trained models on the device.
    CPU bootstrap state and label/evaluation records are reused only within this call.
    """
    if not uq_metrics or len(set(uq_metrics)) != len(uq_metrics):
        raise ValueError("uq_metrics must be nonempty and unique")
    if any(metric not in UQ_METRICS for metric in uq_metrics):
        raise ValueError(f"uq_metrics must be drawn from {UQ_METRICS}")
    precision = resolve_precision(precision, device)
    if bootstrap_epochs < 0:
        raise ValueError(f"bootstrap_epochs must be non-negative, got {bootstrap_epochs}")
    if not model_seeds:
        raise ValueError("model_seeds must not be empty")
    if seed_workers < 1:
        raise ValueError("seed_workers must be positive")
    if len(model_seeds) != len(set(model_seeds)):
        raise ValueError("model seeds must be unique")
    settings = {
        "checkpoint": checkpoint,
        "uq_metrics": uq_metrics,
        "precision": precision,
        "k": k,
        "max_pool_percent": max_pool_percent,
        "bootstrap_epochs": bootstrap_epochs,
        "update_passes": update_passes,
        "replay_ratio": replay_ratio,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "score_batch_size": score_batch_size,
        "max_length": max_length,
    }
    if seed_workers > 1 and len(model_seeds) > 1:
        return _run_concurrent_seeds(
            (seed_examples, pool_inputs, pool_gold, test_examples),
            {**settings, "device": device},
            model_seeds=model_seeds,
            seed_workers=min(seed_workers, len(model_seeds)),
            progress_callback=progress_callback,
        )

    comparisons = {metric: ([], []) for metric in uq_metrics}

    for model_seed in model_seeds:
        CONSOLE.rule(f"[bold cyan]Seed {model_seed} · bootstrap[/]")
        set_seed(model_seed)
        base_model, tokenizer = load_token_classifier(checkpoint, device)
        tokenization_cache = {}
        pool_batches = prepare_inference_batches(
            tokenizer,
            pool_inputs,
            max_length=max_length,
            batch_size=score_batch_size,
            device=device,
        )
        test_batches = prepare_inference_batches(
            tokenizer,
            test_examples,
            max_length=max_length,
            batch_size=score_batch_size,
            evaluation=True,
            device=device,
        )
        scoreable = scoreable_token_keys(
            tokenizer,
            pool_inputs,
            max_length=max_length,
            batch_size=score_batch_size,
            prepared_batches=pool_batches,
        )
        rounds, token_budget = acquisition_schedule(len(scoreable), k, max_pool_percent)
        CONSOLE.print(
            f"[bold cyan]acquisition budget[/] [bold]{token_budget}[/] of "
            f"{len(scoreable)} scoreable tokens "
            f"([bold]{100 * token_budget / len(scoreable):.2f}%[/]) · "
            f"{rounds} rounds of up to K={k} "
            f"(final round: {token_budget - (rounds - 1) * k})"
        )

        bootstrap_optimizer = torch.optim.AdamW(
            base_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        bootstrap_loss = train_items(
            base_model,
            bootstrap_optimizer,
            tokenizer,
            full_sentence_items(seed_examples),
            passes=bootstrap_epochs,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            seed=model_seed,
            tokenization_cache=tokenization_cache,
            precision=precision,
        )
        baseline = evaluate_model(
            base_model,
            tokenizer,
            test_examples,
            max_length=max_length,
            batch_size=score_batch_size,
            device=device,
            prepared_batches=test_batches,
            precision=precision,
        )
        baseline["train_loss"] = bootstrap_loss
        baseline["n_new"] = 0
        baseline["n_replay"] = 0
        CONSOLE.print(
            "[bold green]baseline[/] "
            f"[dim]entity F1[/] [bold]{baseline['entity_f1']:.4f}[/]  "
            f"[dim]token accuracy[/] [bold]{baseline['token_accuracy']:.4f}[/]"
        )
        # Keep the common fitted state on CPU while each metric owns its live model.
        # AdamW load_state_dict moves each copied tensor to the new parameter device.
        base_model.to("cpu")
        bootstrap_optimizer_state = copy.deepcopy(bootstrap_optimizer.state_dict())
        for state in bootstrap_optimizer_state["state"].values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.cpu()
        del bootstrap_optimizer
        random_results = {}
        random_selections = {}

        for metric_index, uq_metric in enumerate(uq_metrics):
            results, selections = comparisons[uq_metric]
            for arm in ("uncertainty", "random"):
                results.append(
                    _result_row(
                        model_seed,
                        arm,
                        0,
                        0,
                        baseline,
                        scoreable_tokens=len(scoreable),
                        token_budget=token_budget,
                        total_rounds=rounds,
                    )
                )
            active_arms = ("uncertainty", "random") if metric_index == 0 else ("uncertainty",)
            models = {arm: copy.deepcopy(base_model).to(device) for arm in active_arms}
            optimizers = {}
            for arm, model in models.items():
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                optimizer.load_state_dict(copy.deepcopy(bootstrap_optimizer_state))
                optimizers[arm] = optimizer
            acquired = {arm: set() for arm in active_arms}
            replay_banks = {arm: seed_replay_items(seed_examples) for arm in active_arms}

            if progress_callback is not None:
                progress_callback(uq_metric, list(results))
            for round_idx in range(1, rounds + 1):
                round_k = min(k, token_budget - (round_idx - 1) * k)
                round_results = {}
                uq_scores = score_token_uncertainty(
                    models["uncertainty"],
                    tokenizer,
                    pool_inputs,
                    metric=uq_metric,
                    excluded=acquired["uncertainty"],
                    max_length=max_length,
                    batch_size=score_batch_size,
                    device=device,
                    prepared_batches=pool_batches,
                    precision=precision,
                )
                chosen = {"uncertainty": select_top_k(uq_scores, round_k)}
                if metric_index == 0:
                    chosen["random"] = select_random(
                        scoreable - acquired["random"],
                        round_k,
                        seed=model_seed * 10_000 + round_idx,
                    )

                for arm_idx, arm in enumerate(active_arms):
                    # Keep the original per-arm RNG seeds independent of metric/order.
                    update_seed = model_seed * 100_000 + round_idx * 10 + arm_idx
                    replay = sample_replay(
                        replay_banks[arm],
                        round_k,
                        replay_ratio,
                        seed=update_seed,
                    )
                    new_items = reveal_pool_items(chosen[arm], pool_inputs, pool_gold)
                    loss = train_items(
                        models[arm],
                        optimizers[arm],
                        tokenizer,
                        [*new_items, *replay],
                        passes=update_passes,
                        batch_size=batch_size,
                        max_length=max_length,
                        device=device,
                        seed=update_seed,
                        tokenization_cache=tokenization_cache,
                        precision=precision,
                    )
                    acquired[arm].update(chosen[arm])
                    replay_banks[arm].extend(new_items)
                    metrics = evaluate_model(
                        models[arm],
                        tokenizer,
                        test_examples,
                        max_length=max_length,
                        batch_size=score_batch_size,
                        device=device,
                        prepared_batches=test_batches,
                        precision=precision,
                    )
                    metrics.update(
                        train_loss=loss,
                        n_new=len(new_items),
                        n_replay=len(replay),
                    )
                    result = _result_row(
                        model_seed,
                        arm,
                        round_idx,
                        len(acquired[arm]),
                        metrics,
                        scoreable_tokens=len(scoreable),
                        token_budget=token_budget,
                        total_rounds=rounds,
                    )
                    results.append(result)
                    round_results[arm] = result
                    selected_rows = []
                    for pool_idx, word_idx in chosen[arm]:
                        example = pool_inputs[pool_idx]
                        selected_rows.append(
                            {
                                "seed": model_seed,
                                "arm": arm,
                                "round": round_idx,
                                "pool_idx": pool_idx,
                                "word_idx": word_idx,
                                "document_name": example["document_name"],
                                "sentence_id": example["sentence_id"],
                                "token": example["tokens"][word_idx],
                                "label": NER_TAGS[pool_gold[(pool_idx, word_idx)]],
                                "uq_metric": uq_metric if arm == "uncertainty" else None,
                                "uq_score": uq_scores[(pool_idx, word_idx)]
                                if arm == "uncertainty"
                                else None,
                            }
                        )
                    selections.extend(selected_rows)
                    if arm == "random":
                        random_results[round_idx] = result
                        random_selections[round_idx] = selected_rows
                if metric_index > 0:
                    # Preserve the ordinary paired export and complete-round snapshots.
                    result = dict(random_results[round_idx])
                    results.append(result)
                    round_results["random"] = result
                    selections.extend(dict(row) for row in random_selections[round_idx])
                if progress_callback is not None:
                    progress_callback(uq_metric, list(results))
                CONSOLE.print(_round_progress_table(round_results, uq_metric))
            del models, optimizers, model, optimizer

        del base_model, bootstrap_optimizer_state, tokenizer, pool_batches, test_batches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return comparisons


def _seed_worker(connection, args: tuple, kwargs: dict, cpu_threads: int) -> None:
    """Own a seed's models and RNGs in a fresh process; report only complete rounds."""
    try:
        torch.set_num_threads(cpu_threads)
        CONSOLE.quiet = True
        transformers_logging.disable_progress_bar()

        def publish(metric, rows):
            connection.send(("progress", (metric, rows[-2:])))

        result = run_metric_comparisons(*args, **kwargs, progress_callback=publish)
        connection.send(("result", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _run_concurrent_seeds(
    args: tuple,
    kwargs: dict,
    *,
    model_seeds: list[int],
    seed_workers: int,
    progress_callback: Callable[[str, list[dict]], None] | None,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Schedule isolated seeds and append paired progress in arrival order."""
    context = multiprocessing.get_context("spawn")
    pending = iter(model_seeds)
    active = {}
    completed = {}
    progress = {metric: [] for metric in kwargs["uq_metrics"]}
    cpu_threads = max(1, torch.get_num_threads() // seed_workers)

    def launch_next():
        seed = next(pending, None)
        if seed is None:
            return
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_seed_worker,
            args=(sender, args, {**kwargs, "model_seeds": [seed]}, cpu_threads),
            name=f"pet-seed-{seed}",
        )
        try:
            process.start()
        except BaseException:
            receiver.close()
            raise
        finally:
            sender.close()
        active[receiver] = (seed, process)
        CONSOLE.print(f"[bold cyan]Seed {seed} started[/]")

    try:
        for _ in range(seed_workers):
            launch_next()
        while active:
            for connection in wait(list(active)):
                seed, process = active[connection]
                try:
                    kind, payload = connection.recv()
                except EOFError as error:
                    process.join()
                    raise RuntimeError(
                        f"Seed {seed} exited without a result (exit code {process.exitcode})"
                    ) from error
                if kind == "error":
                    raise RuntimeError(f"Seed {seed} failed:\n{payload}")
                if kind == "progress":
                    metric, rows = payload
                    progress[metric].extend(rows)
                    CONSOLE.print(
                        f"[bold cyan]Seed {seed}[/]",
                        _round_progress_table({row["arm"]: row for row in rows}, metric),
                    )
                    if progress_callback is not None:
                        progress_callback(metric, list(progress[metric]))
                elif kind == "result":
                    completed[seed] = payload
                    process.join()
                    connection.close()
                    del active[connection]
                    if process.exitcode != 0:
                        raise RuntimeError(f"Seed {seed} exited with code {process.exitcode}")
                    launch_next()
    finally:
        # Also stop sibling workers on model errors, callback errors, or interruption.
        for _, process in active.values():
            if process.is_alive():
                process.terminate()
        for connection, (_, process) in active.items():
            process.join()
            connection.close()

    # Persist deterministic seed/round/arm order even when workers finish out of order.
    return {
        metric: (
            [row for seed in model_seeds for row in completed[seed][metric][0]],
            [row for seed in model_seeds for row in completed[seed][metric][1]],
        )
        for metric in kwargs["uq_metrics"]
    }


def write_run(
    config: dict,
    results: list[dict],
    selections: list[dict],
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write one compact, timestamped experiment record."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / f"bert_token_uq_{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "selections.json").write_text(json.dumps(selections, indent=2))
    if results:
        with (run_dir / "results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return run_dir
