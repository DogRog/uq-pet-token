"""The uncertainty metrics and the selection they drive.

Each sentence was sampled K times with token logprobs. A *metric* turns those cached
samples into one score per sentence, where **higher means less confident**. The two
families read different halves of a cached record — the token logprobs, or the response
text alone — and share one spine each, `logprob_scores` and `committee_scores`.

Metrics live in the `METRICS` registry; `@register` is the whole extension contract, and
a metric's keyword-only parameters automatically become the parameters its arm accepts
in YAML. See README.md for what each metric measures and why, and for how to add one.

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
from itertools import combinations
from statistics import fmean, median, pstdev

from uq_pet.config import NER_TAGS
from uq_pet.prompt import parse_tag_ids

Span = tuple[int, int, str]

# The control arm's name. Selection does not treat it specially; `main` and `plotting`
# use it to find the baseline the other arms are reported against.
RANDOM = "random"

METRICS: dict[str, Callable[..., dict[int, float]]] = {}


# --- the registry -------------------------------------------------------------


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
        name for name, p in signature.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
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


# --- the control --------------------------------------------------------------

# `random` is registered alongside the metrics rather than special-cased, so selection
# has one rule. `length` and `confident` are controls too, but they only make sense read
# next to an uncertainty arm, so they live at the bottom.


@register(RANDOM)
def random_scores(records: Iterable[dict], *, seed: int = 0) -> dict[int, float]:
    """A uniform random score per sentence — the control arm.

    Reads nothing from a record but its index, so the control never inherits the LLM's
    blind spots: a sentence whose samples all failed is as selectable as any other. The
    indices are sorted first, so the draw depends on `seed` alone and not on the order
    the records happen to arrive in.
    """
    rng = random.Random(seed)
    return {idx: rng.random() for idx in sorted(rec["idx"] for rec in records)}


# --- metrics from the logprobs ------------------------------------------------


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


def worst_neg_logprob(token_logprobs: list[float], n_worst: int) -> float:
    """Mean of the n_worst largest -log(p_j), i.e. the sample's least confident tokens.

    Fewer than n_worst tokens means every token counts; nan when there are none.
    """
    if not token_logprobs:
        return math.nan
    return fmean(sorted(-lp for lp in token_logprobs)[-n_worst:])


def logprob_scores(
    logprobs: dict[int, list[list[float]]],
    *,
    reduce: Callable[[list[float]], float] = avg_neg_logprob,
) -> dict[int, float]:
    """Collapse the K samples per sentence into one score, dropping unusable samples.

    `reduce` turns one sample's token logprobs into that sample's score; the samples are
    then averaged. A sentence with no usable sample is omitted entirely rather than
    scored 0, so it can never be mistaken for a confidently-predicted sentence.
    """
    scores = {}
    for idx, samples in logprobs.items():
        values = [v for v in (reduce(lp) for lp in samples) if not math.isnan(v)]
        if values:
            scores[idx] = fmean(values)
    return scores


@register("avg_neg_logprob_filtered")
def avg_neg_logprob_filtered_scores(records: Iterable[dict]) -> dict[int, float]:
    """Mean over the K samples of the average negative logprob of the tag-ID tokens.

    Only the tag-ID tokens count, so brackets and commas — which the model is always
    confident about — don't dilute the signal.
    """
    return logprob_scores(extract_logprobs(records, digits_only=True))


@register("avg_neg_logprob_pure")
def avg_neg_logprob_pure_scores(records: Iterable[dict]) -> dict[int, float]:
    """The same, over every token in the response rather than the tag-ID tokens alone."""
    return logprob_scores(extract_logprobs(records, digits_only=False))


@register("least_confident")
def least_confident_scores(records: Iterable[dict], *, n_worst: int = 1) -> dict[int, float]:
    """Mean over the K samples of the `n_worst` largest -log(p) among tag-ID tokens.

    Scores a sentence by its hardest decisions rather than its average one, so unlike
    the two averaging metrics it carries no implicit length normalization. `n_worst: 1`
    is the single least confident tag.
    """
    if n_worst < 1:
        raise ValueError(f"n_worst must be at least 1, got {n_worst}")
    return logprob_scores(
        extract_logprobs(records, digits_only=True),
        reduce=lambda token_logprobs: worst_neg_logprob(token_logprobs, n_worst),
    )


# --- metrics from the sampled outputs -----------------------------------------

# These read the generated text and no logprobs, so they also work against a gateway
# that does not return them. The K temperature samples act as a committee: the more they
# vary, the less settled the model is about the sentence.


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


def committee_scores(
    records: Iterable[dict],
    reduce: Callable[[list[list[int]]], float],
    *,
    min_samples: int = 2,
) -> dict[int, float]:
    """One score per sentence from its parsed tag arrays, via `reduce`.

    A sentence with fewer than `min_samples` parsable samples has nothing to disagree
    with and is omitted entirely rather than scored 0, so it can never be mistaken for a
    confident sentence. Only `length`, which counts tokens rather than disagreement,
    settles for one sample.
    """
    return {
        idx: reduce(samples)
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) >= min_samples
    }


def votes_by_position(samples: list[list[int]]) -> Iterator[list[int | None]]:
    """The samples' votes at each token position, None where a sample has ended.

    Positions run to the *longest* sample: the samples disagreeing about how many tokens
    the sentence has is itself variation, and the None sentinel makes it count as such
    without a special case (it is not a tag ID, so it can never collide with one).
    """
    for i in range(max(len(s) for s in samples)):
        yield [s[i] if i < len(s) else None for s in samples]


def mean_over_positions(
    samples: list[list[int]], vote_score: Callable[[list[int | None]], float]
) -> float:
    """Average a per-position reading of the votes over the sentence's positions."""
    return fmean(vote_score(votes) for votes in votes_by_position(samples))


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


@register("vote_entropy")
def vote_entropy_scores(records: Iterable[dict]) -> dict[int, float]:
    """Normalized entropy of the K votes, averaged over token positions (0..1)."""
    return committee_scores(records, lambda s: mean_over_positions(s, normalized_vote_entropy))


@register("disagreement")
def disagreement_scores(records: Iterable[dict]) -> dict[int, float]:
    """The same committee reading, scored off the plurality instead of the entropy.

    Coarser than `vote_entropy`, but easier to state: the fraction of samples off the
    most popular tag, averaged over token positions.
    """
    return committee_scores(records, lambda s: mean_over_positions(s, plurality_disagreement))


def tag_spans(tag_ids: list[int]) -> set[Span]:
    """The entity spans a tag-ID array encodes, as {(start, end, type)}, IOB2.

    Spans rather than seqeval, for two reasons: `model_training` sits *below* this module
    in the dependency order, and seqeval refuses two sequences of different lengths —
    which sampled arrays routinely are. A trailing `I-` with no `B-` opens a span, and an
    `I-` of a different type from the open one starts a new span, matching what
    `seqeval.metrics.f1_score` does with no explicit `scheme`
    (`tests/test_uncertainty.py::test_span_f1_matches_seqeval` pins that).

    IDs outside the tag set — only reachable from a truncated response the parser
    salvaged — are read as O rather than raising.
    """
    spans: set[Span] = set()
    start, open_type = 0, None
    for i, tag_id in enumerate([*tag_ids, 0]):
        label = NER_TAGS[tag_id] if 0 <= tag_id < len(NER_TAGS) else "O"
        prefix, _, entity_type = label.partition("-")
        continues = prefix == "I" and entity_type == open_type
        if open_type is not None and not continues:
            spans.add((start, i, open_type))
            open_type = None
        if prefix in ("B", "I") and not continues:
            start, open_type = i, entity_type
    return spans


def span_f1(a: set[Span], b: set[Span]) -> float:
    """Entity-level F1 between two span sets, symmetric in its arguments.

    Two samples that both found no entity agree perfectly, so the empty-empty case is
    1.0 — where seqeval, which cannot tell "nothing to find" from "found nothing", would
    report 0.0 under `zero_division=0`.
    """
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def mean_pairwise_span_f1(samples: list[list[int]]) -> float:
    """Mean entity-level F1 over all K(K-1)/2 pairs, each sample the other's gold."""
    spans = [tag_spans(s) for s in samples]
    return fmean(span_f1(a, b) for a, b in combinations(spans, 2))


@register("pairwise_f1_disagreement")
def pairwise_f1_disagreement_scores(records: Iterable[dict]) -> dict[int, float]:
    """1 - the mean entity-level F1 over every pair of the K samples.

    The same committee as `vote_entropy`, read in the unit the experiment is scored in:
    two samples that agree on 9 of 10 tokens have still named different entities if the
    odd one moved a boundary, and a long entity they all got wrong counts once here but
    once per token in the position-wise vote.
    """
    return committee_scores(records, lambda s: 1 - mean_pairwise_span_f1(s))


@register("entity_count_std")
def entity_count_std_scores(records: Iterable[dict]) -> dict[int, float]:
    """Population standard deviation of how many entities each of the K samples found.

    The coarsest committee reading there is — it ignores where the entities are — and
    orthogonal enough to the position-wise metrics to be worth an arm: samples can agree
    on the count while disagreeing everywhere about the boundaries, and vice versa.
    """
    return committee_scores(records, lambda s: pstdev(len(tag_spans(t)) for t in s))


# --- controls -----------------------------------------------------------------


@register("length")
def length_scores(records: Iterable[dict]) -> dict[int, float]:
    """The sentence's token count — the confounder arm, not an uncertainty measure.

    If this arm matches the best metric, the finding is about sentence length rather
    than uncertainty. The count is the median over the K sampled arrays, because a
    metric sees cache records and nothing else: the prompt demands one tag per token, so
    a sample's length *is* the token count whenever the model complied, and the median
    shrugs off the samples that did not.
    """
    return committee_scores(records, lambda s: float(median(len(t) for t in s)), min_samples=1)


@register("confident")
def confident_scores(
    records: Iterable[dict], *, metric: str = "avg_neg_logprob_filtered"
) -> dict[int, float]:
    """Another metric's ranking, reversed — most confident sentences first.

    A metric that beats `random` while its reverse also beats `random` was measuring
    something other than uncertainty. The wrapped metric runs with its own defaults, and
    sentences it omits stay omitted.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}' to reverse (expected one of {metric_names()})")
    if METRICS[metric] is confident_scores:
        raise ValueError("'confident' cannot reverse itself")
    return {idx: -score for idx, score in METRICS[metric](records).items()}


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
