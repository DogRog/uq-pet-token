"""Sequential token acquisition, replay, orchestration, and run outputs."""

import copy
import csv
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from uq_pet.pet_data import NER_TAGS, RESULTS_DIR, TokenKey
from uq_pet.token_model import (
    evaluate_model,
    load_token_classifier,
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
    """Return full acquisition rounds and their token budget."""
    if scoreable_tokens < 1:
        raise ValueError("the scoreable pool must contain at least one token")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not 0 < max_pool_percent <= 100:
        raise ValueError(f"max_pool_percent must be in (0, 100], got {max_pool_percent}")

    requested_tokens = int(scoreable_tokens * max_pool_percent / 100)
    rounds = requested_tokens // k
    if rounds < 1:
        raise ValueError(
            f"{max_pool_percent:g}% of {scoreable_tokens} scoreable tokens "
            f"is fewer than one full k={k} acquisition round"
        )
    return rounds, rounds * k


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
    checkpoint: str,
    model_seeds: list[int],
    uq_metric: str,
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
    progress_callback: Callable[[list[dict]], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run uncertainty and random online learners from the same bootstrap state."""
    if bootstrap_epochs < 0:
        raise ValueError(f"bootstrap_epochs must be non-negative, got {bootstrap_epochs}")
    if not model_seeds:
        raise ValueError("model_seeds must not be empty")

    results: list[dict] = []
    selections: list[dict] = []

    for model_seed in model_seeds:
        CONSOLE.rule(f"[bold cyan]Seed {model_seed} · bootstrap[/]")
        set_seed(model_seed)
        base_model, tokenizer = load_token_classifier(checkpoint, device)
        scoreable = scoreable_token_keys(
            tokenizer,
            pool_inputs,
            max_length=max_length,
            batch_size=score_batch_size,
        )
        rounds, token_budget = acquisition_schedule(len(scoreable), k, max_pool_percent)
        CONSOLE.print(
            f"[bold cyan]acquisition budget[/] [bold]{token_budget}[/] of "
            f"{len(scoreable)} scoreable tokens "
            f"([bold]{100 * token_budget / len(scoreable):.2f}%[/]) · "
            f"{rounds} full rounds of K={k}"
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
        )
        baseline = evaluate_model(
            base_model,
            tokenizer,
            test_examples,
            max_length=max_length,
            batch_size=score_batch_size,
            device=device,
        )
        baseline["train_loss"] = bootstrap_loss
        baseline["n_new"] = 0
        baseline["n_replay"] = 0
        CONSOLE.print(
            "[bold green]baseline[/] "
            f"[dim]entity F1[/] [bold]{baseline['entity_f1']:.4f}[/]  "
            f"[dim]token accuracy[/] [bold]{baseline['token_accuracy']:.4f}[/]"
        )
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
        if progress_callback is not None:
            progress_callback(list(results))

        models = {
            "uncertainty": copy.deepcopy(base_model),
            "random": copy.deepcopy(base_model),
        }
        bootstrap_optimizer_state = copy.deepcopy(bootstrap_optimizer.state_dict())
        del base_model, bootstrap_optimizer
        optimizers = {}
        for arm, model in models.items():
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            optimizer.load_state_dict(copy.deepcopy(bootstrap_optimizer_state))
            optimizers[arm] = optimizer
        del bootstrap_optimizer_state

        acquired = {"uncertainty": set(), "random": set()}
        replay_banks = {
            "uncertainty": seed_replay_items(seed_examples),
            "random": seed_replay_items(seed_examples),
        }

        for round_idx in range(1, rounds + 1):
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
            )
            chosen = {
                "uncertainty": select_top_k(uq_scores, k),
                "random": select_random(
                    scoreable - acquired["random"],
                    k,
                    seed=model_seed * 10_000 + round_idx,
                ),
            }

            for arm_idx, arm in enumerate(("uncertainty", "random")):
                replay = sample_replay(
                    replay_banks[arm],
                    k,
                    replay_ratio,
                    seed=model_seed * 100_000 + round_idx * 10 + arm_idx,
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
                    seed=model_seed * 100_000 + round_idx * 10 + arm_idx,
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
                )
                metrics.update(
                    {
                        "train_loss": loss,
                        "n_new": len(new_items),
                        "n_replay": len(replay),
                    }
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

                for pool_idx, word_idx in chosen[arm]:
                    example = pool_inputs[pool_idx]
                    selections.append(
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
            CONSOLE.print(_round_progress_table(round_results, uq_metric))
            if progress_callback is not None:
                progress_callback(list(results))

        del models, optimizers, tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, selections


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
