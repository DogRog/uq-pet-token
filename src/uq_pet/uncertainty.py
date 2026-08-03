"""The uncertainty metrics and the selection they drive.

Each sentence was sampled K times with token logprobs. A *metric* turns those cached
samples into one score per sentence, where higher means the model was less confident —
from the logprobs (`avg_neg_logprob`), from the response text alone
(`output_disagreement`), or from anything else a record holds. Metrics live in the
`METRICS` registry:

    @register("my_metric")
    def my_metric_scores(records, *, some_param: int = 3) -> dict[int, float]:
        ...

That is the whole contract — the config, the CLI and the reporting all pick metrics up
from the registry by name, and a metric's keyword-only parameters automatically become
the parameters its arm accepts in YAML. Nothing else needs to change to add one.

The `random` control is a metric like any other: it scores each sentence with a uniform
random number, so ranking by it and taking the top n *is* a uniform random sample of n.
That leaves exactly one selection rule — `select` takes the top n of a metric's ranking
— rather than one rule plus a special case for the control.

Scores are always recomputed from the cache, never stored, so metrics, their parameters
and the budget can all be swept without new LLM calls.
"""

import inspect
import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from statistics import fmean

from uq_pet.prompt import parse_tag_ids

# The control arm's name. Selection does not treat it specially; `main` and `plotting`
# use it to find the baseline the other arms are reported against.
RANDOM = "random"

METRICS: dict[str, Callable[..., dict[int, float]]] = {}


def register(name: str) -> Callable:
    """Register an uncertainty metric under `name`."""

    def decorator(fn: Callable[..., dict[int, float]]) -> Callable[..., dict[int, float]]:
        if name in METRICS:
            raise ValueError(f"metric '{name}' is already registered")
        METRICS[name] = fn
        return fn

    return decorator


def metric_names() -> list[str]:
    return sorted(METRICS)


def metric_params(strategy: str) -> list[str]:
    """The keyword-only parameter names a metric accepts, i.e. its YAML knobs."""
    signature = inspect.signature(METRICS[strategy])
    return [
        name
        for name, p in signature.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def validate_arm(strategy: str, params: dict) -> None:
    """Raise if the strategy is unknown or carries parameters it does not accept."""
    if strategy not in METRICS:
        raise ValueError(f"Unknown strategy '{strategy}' (expected one of {metric_names()})")
    allowed = metric_params(strategy)
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        expected = f"accepts {allowed}" if allowed else "accepts no parameters"
        raise ValueError(f"Strategy '{strategy}' {expected}, got unknown {unknown}")


def score_arm(strategy: str, records: Iterable[dict], **params) -> dict[int, float]:
    """Scores for one arm, from the metric its strategy names."""
    validate_arm(strategy, params)
    return METRICS[strategy](records, **params)


# --- metrics ------------------------------------------------------------------


@register(RANDOM)
def random_scores(records: Iterable[dict], *, seed: int = 42) -> dict[int, float]:
    """A uniform random score per sentence — the control arm.

    Ranking by iid uniform scores and taking the top n is a uniform random sample of n,
    so the control needs no selection rule of its own. It reads nothing from a record
    but its index, so the control never inherits the LLM's blind spots: a sentence whose
    samples all failed is exactly as selectable as any other.

    The indices are sorted first, so the draw depends on `seed` alone and not on the
    order the records happen to arrive in.
    """
    rng = random.Random(seed)
    return {idx: rng.random() for idx in sorted(rec["idx"] for rec in records)}


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


TOKEN_SELECTIONS = ("filtered", "pure")


@register("avg_neg_logprob")
def avg_neg_logprob_scores(
    records: Iterable[dict], *, tokens: str = "filtered"
) -> dict[int, float]:
    """Mean over the K samples of the average negative token logprob.

    `tokens` picks which tokens count:
      "filtered" — only the tag-ID tokens, so brackets and commas (which the model is
                   always confident about, and which there are more of in long
                   sentences) don't dilute the signal;
      "pure"     — every token in the response.
    """
    if tokens not in TOKEN_SELECTIONS:
        raise ValueError(f"tokens must be one of {TOKEN_SELECTIONS}, got '{tokens}'")
    return sentence_scores(extract_logprobs(records, digits_only=(tokens == "filtered")))


def extract_tag_ids(records: Iterable[dict]) -> dict[int, list[list[int]]]:
    """{idx: [parsed tag array per sample]}, reading the response text and nothing else.

    Skips failed records and samples that parse to nothing, so a sentence can end up
    with fewer arrays than there were samples — or none at all.
    """
    return {
        rec["idx"]: [
            tags
            for tags in (parse_tag_ids(choice.get("text")) for choice in rec.get("choices", []))
            if tags is not None
        ]
        for rec in records
        if "error" not in rec
    }


def votes_by_position(samples: list[list[int]]) -> Iterator[list[int | None]]:
    """The samples' votes at each token position, None where a sample has ended.

    Positions run to the *longest* sample: the samples disagreeing about how many tokens
    the sentence has is itself variation, and the None sentinel makes it count as such
    without a special case (it is not a tag ID, so it can never collide with one).
    """
    for i in range(max(len(s) for s in samples)):
        yield [s[i] if i < len(s) else None for s in samples]


def normalized_vote_entropy(votes: list[int | None]) -> float:
    """Shannon entropy of the votes over log(k), so it is 0..1 whatever k is.

    Dividing by log(k) — the entropy of k samples that all differ — keeps sentences
    comparable when some of them lost a sample to a parse failure.
    """
    k = len(votes)
    if k < 2:
        return 0.0
    counts = Counter(votes).values()
    return -sum((c / k) * math.log(c / k) for c in counts) / math.log(k)


def plurality_disagreement(votes: list[int | None]) -> float:
    """Fraction of samples that did not vote for the most popular tag; 0..1-1/k."""
    return 1 - max(Counter(votes).values()) / len(votes)


MEASURES: dict[str, Callable[[list[int | None]], float]] = {
    "vote_entropy": normalized_vote_entropy,
    "disagreement": plurality_disagreement,
}


@register("output_disagreement")
def output_disagreement_scores(
    records: Iterable[dict], *, measure: str = "vote_entropy"
) -> dict[int, float]:
    """How much the K sampled tag arrays disagree, averaged over token positions.

    Reads only the generated text — no logprobs — so it also works for a gateway that
    does not return them. The K temperature samples act as a committee: the more they
    vary, the less settled the model is about the sentence.

    `measure` picks the per-position disagreement:
      "vote_entropy"  — normalized Shannon entropy of the votes (0..1);
      "disagreement"  — the fraction of samples off the plurality tag (0..1-1/k),
                        coarser but easier to state.

    A sentence with fewer than two parsable samples has nothing to disagree with and is
    omitted rather than scored 0, so it can never be mistaken for a confident sentence.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {tuple(MEASURES)}, got '{measure}'")
    disagreement = MEASURES[measure]
    return {
        idx: fmean(disagreement(votes) for votes in votes_by_position(samples))
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) > 1
    }


# --- selection ----------------------------------------------------------------


def rank_by_uncertainty(scores: dict[int, float], *, tie_seed: int = 0) -> list[int]:
    """Indices from most to least uncertain.

    Ties are broken by a seeded shuffle before a stable sort, so equal scores do not
    resolve by insertion order — otherwise the selection would silently depend on the
    order records happened to be written to the cache.
    """
    indices = list(scores)
    random.Random(tie_seed).shuffle(indices)
    return sorted(indices, key=lambda idx: scores[idx], reverse=True)


def n_from_percent(total: int, percentage: float) -> int:
    """Budget in sentences from a percentage of the pool; at least 1."""
    return max(1, int(total * percentage / 100))


def select(
    scores: dict[int, float], examples: list[dict], n: int, *, tie_seed: int = 0
) -> list[dict]:
    """The n sentences an arm trains on: the top n of its metric's ranking.

    The one selection rule, control arm included — under `random`'s uniform scores the
    top n is a uniform random sample of n.
    """
    if n > len(examples):
        raise ValueError(f"budget of {n} exceeds the {len(examples)} available sentences")
    if len(scores) < n:
        # Sentences whose samples all failed have no score. Selecting anyway would
        # hand this arm fewer sentences than the others and invalidate the
        # comparison, so refuse rather than quietly shrink the budget.
        raise ValueError(
            f"budget of {n} exceeds the {len(scores)} scored sentences — the LLM cache is "
            f"incomplete for this split"
        )
    return [examples[idx] for idx in rank_by_uncertainty(scores, tie_seed=tie_seed)[:n]]
