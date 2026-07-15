"""Uncertainty metrics over repeated LLM samples, and the selection strategies
built on them.

Every metric returns a scalar where higher = more uncertain. Metrics register
themselves in METRICS with a box type:
- "black" metrics measure disagreement between the K parsed tag sequences and
  work with any backend;
- "white" metrics read the model's internal signal (`token_entropies`, only
  present in caches from a local backend).

Registered metrics uniformly take a full cache record (`register` adapts
black-box functions), so `compute_metric(name, record)` works for any metric.
Strategy strings are "random", "uncertainty:<metric_name>", or "full".
"""

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    fn: Callable[[dict], float]  # cache record -> score
    box: str  # "black" | "white"


METRICS: dict[str, Metric] = {}


def register(name: str, box: str = "black"):
    """Register a metric; black-box functions take `parsed_samples` and are
    wrapped so every registered fn takes the full cache record."""
    def decorator(fn):
        record_fn = fn if box == "white" else (lambda record: fn(record["parsed_samples"]))
        METRICS[name] = Metric(record_fn, box)
        return fn
    return decorator


class WhiteboxDataMissingError(ValueError):
    """A white-box metric was asked to score a record without model internals."""


def compute_metric(name: str, record: dict) -> float:
    """Score one cache record with a registered metric."""
    if name not in METRICS:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
    return METRICS[name].fn(record)


def shannon_entropy(samples: list) -> float:
    """Shannon entropy (bits) over the frequencies of a list of hashable items."""
    if not samples:
        return 0.0
    counts = Counter(samples)
    total = len(samples)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def token_entropies(parsed_samples: list[list[str]]) -> list[float]:
    """Per-position entropy of the tags the K samples assigned to each token."""
    if not parsed_samples:
        return []
    length = len(parsed_samples[0])
    return [
        shannon_entropy([sample[i] for sample in parsed_samples])
        for i in range(length)
    ]


def majority_vote(parsed_samples: list[list[str]]) -> list[str]:
    """Most frequent tag per position across the K samples."""
    if not parsed_samples:
        return []
    length = len(parsed_samples[0])
    return [
        Counter(sample[i] for sample in parsed_samples).most_common(1)[0][0]
        for i in range(length)
    ]


@register("sequence_entropy")
def sequence_entropy(parsed_samples: list[list[str]]) -> float:
    """Entropy over exact full-sequence matches. Saturates at log2(K)."""
    return shannon_entropy([tuple(sample) for sample in parsed_samples])


@register("mean_token_entropy")
def mean_token_entropy(parsed_samples: list[list[str]]) -> float:
    entropies = token_entropies(parsed_samples)
    return sum(entropies) / len(entropies) if entropies else 0.0


@register("max_token_entropy")
def max_token_entropy(parsed_samples: list[list[str]]) -> float:
    entropies = token_entropies(parsed_samples)
    return max(entropies) if entropies else 0.0


@register("variation_ratio")
def variation_ratio(parsed_samples: list[list[str]]) -> float:
    """Mean over tokens of (1 - majority tag fraction)."""
    if not parsed_samples or not parsed_samples[0]:
        return 0.0
    k = len(parsed_samples)
    length = len(parsed_samples[0])
    ratios = []
    for i in range(length):
        top_count = Counter(sample[i] for sample in parsed_samples).most_common(1)[0][1]
        ratios.append(1.0 - top_count / k)
    return sum(ratios) / length


@register("jaccard_distance")
def jaccard_distance(parsed_samples: list[list[str]]) -> float:
    """Average pairwise Jaccard distance over (position, tag) sets."""
    if not parsed_samples or len(parsed_samples) < 2:
        return 0.0
    distances = []
    for i in range(len(parsed_samples)):
        set_i = set(enumerate(parsed_samples[i]))
        for j in range(i + 1, len(parsed_samples)):
            set_j = set(enumerate(parsed_samples[j]))
            union = len(set_i | set_j)
            jaccard_index = len(set_i & set_j) / union if union > 0 else 1.0
            distances.append(1.0 - jaccard_index)
    return sum(distances) / len(distances)


@register("predictive_entropy", box="white")
def predictive_entropy(record: dict) -> float:
    """Mean token predictive entropy (bits) from the model's own per-step
    distributions: mean over generated tokens per sample, mean over K samples."""
    token_entropies = record.get("token_entropies")
    if not token_entropies:
        raise WhiteboxDataMissingError(
            f"Record '{record.get('key')}' has no token_entropies; white-box "
            "metrics need a cache produced by a local backend (llm.backend: mlx). "
            "Re-run score-pool with an mlx config."
        )
    per_sample = [sum(ents) / len(ents) for ents in token_entropies if ents]
    return sum(per_sample) / len(per_sample) if per_sample else 0.0


# --- selection strategies ---------------------------------------------------


def strategy_metric(strategy: str) -> str | None:
    """Metric name for an "uncertainty:<name>" strategy, else None."""
    return strategy.split(":", 1)[1] if strategy.startswith("uncertainty:") else None


def select_top_uncertainty(scores: dict[str, float], n: int, seed: int) -> list[str]:
    """Top-n keys by uncertainty, most uncertain first.

    A seeded shuffle before the stable sort breaks ties randomly but
    reproducibly (sequence entropy saturates at log2(K), so ties are common).
    """
    keys = list(scores)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: scores[k], reverse=True)
    return keys[:n]


def select_random(keys: list[str], n: int, seed: int) -> list[str]:
    return random.Random(seed).sample(list(keys), n)


def select(strategy: str, keys: list[str], scores: dict[str, float] | None,
           n: int, seed: int) -> list[str]:
    if strategy == "random":
        return select_random(keys, n, seed)
    if strategy_metric(strategy) is not None:
        if scores is None:
            raise ValueError(f"Strategy '{strategy}' needs uncertainty scores.")
        # Ties are broken with a fixed seed so the selected subset is identical
        # across repeats; only the training seed varies for uncertainty cells.
        return select_top_uncertainty(scores, n, seed=0)
    raise ValueError(f"Unknown strategy '{strategy}'.")
