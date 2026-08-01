"""Uncertainty scoring, and the selection strategies it is compared against.

Scoring: each sentence was sampled K times with token logprobs, so its uncertainty is
the mean over samples of the average negative token logprob — higher means the model
was less confident. The `filtered` variant keeps only the tag-ID tokens, so that
punctuation and formatting tokens (which the model is always confident about, and
which there are more of in longer sentences) do not dilute the signal.

Selection: `random_sample_sentences` is the control arm. It lives here so both arms
sit side by side and share one notion of "a budget of n sentences".

Scores are always recomputed from the cache, never stored, so the score variant and
the budget can be changed without new LLM calls.
"""

import math
import random
from collections.abc import Iterable
from statistics import fmean


def is_tag_token(token: str) -> bool:
    """True for tokens that carry a tag ID, e.g. '1', ' 12', '[0', '3,'."""
    return token.strip().strip(",[]").isdigit()


def extract_logprobs(
    records: Iterable[dict], *, digits_only: bool = False
) -> dict[int, list[list[float]]]:
    """{idx: [[token logprob per token] per sample]}, skipping failed records."""
    return {
        rec["idx"]: [
            [
                t["logprob"]
                for t in choice.get("logprobs", [])
                if not digits_only or is_tag_token(t["token"])
            ]
            for choice in rec.get("choices", [])
        ]
        for rec in records
        if "error" not in rec
    }


def avg_neg_logprob(token_logprobs: list[float]) -> float:
    """-1/L * sum_j log(p_j); higher means lower model confidence. nan when empty."""
    return -fmean(token_logprobs) if token_logprobs else math.nan


def sentence_scores(logprobs: dict[int, list[list[float]]]) -> dict[int, float]:
    """Collapse the K samples per sentence into one score, dropping unusable samples.

    A sentence with no usable sample is omitted entirely rather than scored 0, so it
    can never be mistaken for a confidently-predicted sentence.
    """
    scores = {}
    for idx, samples in logprobs.items():
        values = [v for v in (avg_neg_logprob(lp) for lp in samples) if not math.isnan(v)]
        if values:
            scores[idx] = fmean(values)
    return scores


def score_records(records: Iterable[dict], *, digits_only: bool) -> dict[int, float]:
    """Uncertainty score per sentence index, straight from cache records."""
    return sentence_scores(extract_logprobs(records, digits_only=digits_only))


def rank_by_uncertainty(scores: dict[int, float], *, tie_seed: int = 0) -> list[int]:
    """Indices from most to least uncertain.

    Ties are broken by a seeded shuffle before a stable sort, so equal scores do not
    resolve by insertion order — otherwise the selection would silently depend on the
    order records happened to be written to the cache.
    """
    indices = list(scores)
    random.Random(tie_seed).shuffle(indices)
    return sorted(indices, key=lambda idx: scores[idx], reverse=True)


def n_from_percent(total: int, pct: float) -> int:
    """Budget in sentences from a percentage of the pool; at least 1."""
    return max(1, int(total * pct / 100))


def prepare_uncertain_sentences(
    scores: dict[int, float], examples: list[dict], n: int, *, tie_seed: int = 0
) -> list[dict]:
    """The n most uncertain sentences, most uncertain first."""
    if n > len(examples):
        raise ValueError(f"budget of {n} exceeds the {len(examples)} available sentences")
    if len(scores) < n:
        # Sentences whose samples all failed have no score. Selecting anyway would
        # hand this arm fewer sentences than the random arm and invalidate the
        # comparison, so refuse rather than quietly shrink the budget.
        raise ValueError(
            f"budget of {n} exceeds the {len(scores)} scored sentences — the LLM cache is "
            f"incomplete for this split"
        )
    return [examples[idx] for idx in rank_by_uncertainty(scores, tie_seed=tie_seed)[:n]]


def random_sample_sentences(examples: list[dict], n: int, *, seed: int = 42) -> list[dict]:
    """A uniform random sample of n sentences — the control arm.

    Uses a local Random so that selecting does not perturb the global RNG that
    model_training.set_seed relies on for training determinism.
    """
    if n > len(examples):
        raise ValueError(f"budget of {n} exceeds the {len(examples)} available sentences")
    return random.Random(seed).sample(examples, n)


def select(
    strategy: str,
    examples: list[dict],
    n: int,
    *,
    scores: dict[int, float] | None = None,
    seed: int = 42,
    tie_seed: int = 0,
) -> list[dict]:
    """Dispatch to a selection strategy by name."""
    if strategy == "uncertainty":
        if scores is None:
            raise ValueError("the 'uncertainty' strategy needs scores")
        return prepare_uncertain_sentences(scores, examples, n, tie_seed=tie_seed)
    if strategy == "random":
        return random_sample_sentences(examples, n, seed=seed)
    raise ValueError(f"Unknown selection strategy '{strategy}'")
