"""BERT predictive-uncertainty metrics and the shared top-n selection rule.

Every scorer record contains one class-probability vector per scored word. Metrics
turn those vectors into one sentence score, where higher means less confident.
"""

import inspect
import math
import random
from collections.abc import Callable, Iterable
from statistics import fmean

RANDOM = "random"
METRICS: dict[str, Callable[..., dict[int, float]]] = {}


def register(name: str) -> Callable:
    """Register an uncertainty metric under ``name``."""

    def decorator(fn: Callable[..., dict[int, float]]) -> Callable[..., dict[int, float]]:
        if name in METRICS:
            raise ValueError(f"metric '{name}' is already registered")
        METRICS[name] = fn
        return fn

    return decorator


def metric_names() -> list[str]:
    return sorted(METRICS)


def metric_params(strategy: str) -> list[str]:
    """Keyword-only parameters accepted by a registered metric."""
    return [
        name
        for name, parameter in inspect.signature(METRICS[strategy]).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def validate_arm(strategy: str, params: dict) -> None:
    """Raise for an unknown strategy or unsupported strategy parameter."""
    if strategy not in METRICS:
        raise ValueError(f"Unknown strategy '{strategy}' (expected one of {metric_names()})")
    allowed = metric_params(strategy)
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        expected = f"accepts {allowed}" if allowed else "accepts no parameters"
        raise ValueError(f"Strategy '{strategy}' {expected}, got unknown {unknown}")


def score_arm(strategy: str, records: Iterable[dict], **params) -> dict[int, float]:
    """Dispatch one arm to its registered metric."""
    validate_arm(strategy, params)
    return METRICS[strategy](records, **params)


@register(RANDOM)
def random_scores(records: Iterable[dict], *, seed: int = 0) -> dict[int, float]:
    """Uniform scores, making top-n selection a uniform random sample."""
    rng = random.Random(seed)
    return {idx: rng.random() for idx in sorted(record["idx"] for record in records)}


def word_probabilities(record: dict) -> list[list[float]]:
    """The usable finite probability vectors in one scorer record."""
    usable = []
    for probabilities in record.get("probabilities", []):
        values = [float(value) for value in probabilities]
        total = sum(values)
        if values and total > 0 and all(math.isfinite(value) and value >= 0 for value in values):
            usable.append([value / total for value in values])
    return usable


def normalized_entropy(probabilities: list[float]) -> float:
    """Shannon entropy normalized to 0..1 for the vector's class count."""
    if len(probabilities) < 2:
        return 0.0
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    values = [value / total for value in probabilities]
    return -sum(value * math.log(value) for value in values if value > 0) / math.log(len(values))


def reduce_words(
    records: Iterable[dict], reduce: Callable[[list[list[float]]], float]
) -> dict[int, float]:
    """Apply a sentence-level reduction, omitting records with no scored words."""
    scores = {}
    for record in records:
        probabilities = word_probabilities(record)
        if probabilities:
            scores[record["idx"]] = reduce(probabilities)
    return scores


@register("mean_token_entropy")
def mean_token_entropy_scores(records: Iterable[dict]) -> dict[int, float]:
    """Mean normalized predictive entropy over words in each sentence."""
    return reduce_words(records, lambda words: fmean(normalized_entropy(p) for p in words))


@register("max_token_entropy")
def max_token_entropy_scores(records: Iterable[dict]) -> dict[int, float]:
    """Entropy of the least-certain word in each sentence."""
    return reduce_words(records, lambda words: max(normalized_entropy(p) for p in words))


@register("least_confident")
def least_confident_scores(records: Iterable[dict]) -> dict[int, float]:
    """Mean ``1 - max(class probability)`` over words."""
    return reduce_words(records, lambda words: fmean(1 - max(p) for p in words))


@register("margin")
def margin_scores(records: Iterable[dict]) -> dict[int, float]:
    """Mean inverse gap between the two most likely tag classes."""

    def inverse_margin(words: list[list[float]]) -> float:
        margins = []
        for probabilities in words:
            largest = sorted(probabilities, reverse=True)[:2]
            margins.append(1 - (largest[0] - largest[1] if len(largest) == 2 else largest[0]))
        return fmean(margins)

    return reduce_words(records, inverse_margin)


@register("length")
def length_scores(records: Iterable[dict]) -> dict[int, float]:
    """Sentence length control, independent of model confidence."""
    return {record["idx"]: float(record["n_tokens"]) for record in records}


@register("confident")
def confident_scores(
    records: Iterable[dict], *, metric: str = "mean_token_entropy"
) -> dict[int, float]:
    """Reverse another metric to select the most confident sentences first."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}' to reverse (expected one of {metric_names()})")
    if METRICS[metric] is confident_scores:
        raise ValueError("'confident' cannot reverse itself")
    return {idx: -score for idx, score in METRICS[metric](records).items()}


def rank_by_uncertainty(scores: dict[int, float], *, tie_seed: int = 0) -> list[int]:
    """Indices from most to least uncertain, with deterministic shuffled ties."""
    indices = list(scores)
    random.Random(tie_seed).shuffle(indices)
    return sorted(indices, key=lambda idx: scores[idx], reverse=True)


def n_from_percent(total: int, percentage: float) -> int:
    """Convert a percentage budget to a sentence count, with a minimum of one."""
    return max(1, int(total * percentage / 100))


def select(
    scores: dict[int, float], examples: list[dict], n: int, *, tie_seed: int = 0
) -> list[dict]:
    """Select the top-n scored pool examples."""
    if n > len(examples):
        raise ValueError(f"budget of {n} exceeds the {len(examples)} available sentences")
    if len(scores) < n:
        raise ValueError(
            f"budget of {n} exceeds the {len(scores)} scored sentences — scoring is incomplete"
        )
    return [examples[idx] for idx in rank_by_uncertainty(scores, tie_seed=tie_seed)[:n]]
