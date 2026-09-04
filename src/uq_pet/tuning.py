"""Search-space, validation-split, and objective helpers for tuning."""

import random
from collections.abc import Mapping, Sequence

from uq_pet.pet_data import TokenKey
from uq_pet.token_model import UQ_METRICS

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
    """Sample one configuration from the same discrete space as the grid runner."""
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
