"""The uncertainty metrics and the selection they drive.

Each sentence was sampled K times with token logprobs. A *metric* turns those cached
samples into one score per sentence, where higher means the model was less confident —
from the logprobs (`avg_neg_logprob_filtered`, `avg_neg_logprob_pure`,
`least_confident`), from the response text alone (`vote_entropy`, `disagreement`,
`pairwise_f1_disagreement`, `entity_count_std`), or from anything else a record holds.
Each variant is its own registry entry rather than a parameter of a shared one, so an
arm is just a name. Metrics live in the `METRICS` registry:

    @register("my_metric")
    def my_metric_scores(records, *, some_param: int = 3) -> dict[int, float]:
        ...

That is the whole contract — the config, the CLI and the reporting all pick metrics up
from the registry by name, and a metric's keyword-only parameters automatically become
the parameters its arm accepts in YAML. Nothing else needs to change to add one.

The `random` control is a metric like any other: it scores each sentence with a uniform
random number, so ranking by it and taking the top n *is* a uniform random sample of n.
That leaves exactly one selection rule — `select` takes the top n of a metric's ranking
— rather than one rule plus a special case for the control. `length` and `confident`
are controls in the same sense: they answer "is this uncertainty, or is it sentence
length?" and "does the signal reverse when you rank it backwards?", which the
random baseline alone cannot.

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


def worst_neg_logprob(token_logprobs: list[float], n_worst: int) -> float:
    """Mean of the n_worst largest -log(p_j), i.e. the sample's least confident tokens.

    Fewer than n_worst tokens means every token counts; nan when there are none.
    """
    if not token_logprobs:
        return math.nan
    return fmean(sorted(-lp for lp in token_logprobs)[-n_worst:])


def sentence_scores(
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
    confident about, and which there are more of in long sentences — don't dilute the
    signal.
    """
    return sentence_scores(extract_logprobs(records, digits_only=True))


@register("avg_neg_logprob_pure")
def avg_neg_logprob_pure_scores(records: Iterable[dict]) -> dict[int, float]:
    """The same, over every token in the response rather than the tag-ID tokens alone."""
    return sentence_scores(extract_logprobs(records, digits_only=False))


@register("least_confident")
def least_confident_scores(records: Iterable[dict], *, n_worst: int = 1) -> dict[int, float]:
    """The `n_worst` least confident tag-ID tokens per sample, averaged over the K samples.

    Classic least-confidence, and the counterweight to the two averaging metrics: a mean
    over the whole response dilutes one genuinely hard tag in an otherwise easy sentence
    — the longer the sentence, the more it dilutes. Taking the worst tokens instead
    scores a sentence by its hardest decisions, and drops the implicit length
    normalization along with it. `n_worst: 1` is the single least confident tag.
    """
    if n_worst < 1:
        raise ValueError(f"n_worst must be at least 1, got {n_worst}")
    return sentence_scores(
        extract_logprobs(records, digits_only=True),
        reduce=lambda token_logprobs: worst_neg_logprob(token_logprobs, n_worst),
    )


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


@register("vote_entropy")
def vote_entropy_scores(records: Iterable[dict]) -> dict[int, float]:
    """Disagreement among the K sampled tag arrays, averaged over token positions.

    Reads only the generated text — no logprobs — so it also works for a gateway that
    does not return them. The K temperature samples act as a committee: the more they
    vary, the less settled the model is about the sentence. Here that variation is the
    normalized Shannon entropy of each position's votes (0..1).

    A sentence with fewer than two parsable samples has nothing to disagree with and is
    omitted rather than scored 0, so it can never be mistaken for a confident sentence.
    """
    return {
        idx: fmean(normalized_vote_entropy(votes) for votes in votes_by_position(samples))
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) > 1
    }


@register("disagreement")
def disagreement_scores(records: Iterable[dict]) -> dict[int, float]:
    """The same committee reading as `vote_entropy`, scored off the plurality instead.

    Each position contributes the fraction of samples that did not vote for the most
    popular tag (0..1-1/k) — coarser than the entropy, but easier to state. Sentences
    with fewer than two parsable samples are omitted for the same reason.
    """
    return {
        idx: fmean(plurality_disagreement(votes) for votes in votes_by_position(samples))
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) > 1
    }


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


@register("pairwise_f1_disagreement")
def pairwise_f1_disagreement_scores(records: Iterable[dict]) -> dict[int, float]:
    """1 - the mean entity-level F1 over every pair of the K samples.

    The same committee as `vote_entropy`, read at the granularity the experiment is
    scored at. Position-aligned voting weights a sentence by how many *tokens* the
    samples put different tags on, which is not the quantity
    `model_training.evaluate` reports: samples that agree on 9 of 10 tokens have still
    named two different entities if the odd one moved a boundary, and a long entity all
    the samples got wrong counts once at the entity level but once per token in the
    vote. Comparing span sets scores the disagreement in entities. Taking each sample as
    the other's gold in turn is why the pairing is over all K(K-1)/2 pairs rather than
    against a single reference.

    Sentences with fewer than two parsable samples are omitted, as in `vote_entropy`.
    """
    return {
        idx: 1 - fmean(span_f1(a, b) for a, b in combinations([tag_spans(s) for s in samples], 2))
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) > 1
    }


@register("entity_count_std")
def entity_count_std_scores(records: Iterable[dict]) -> dict[int, float]:
    """Population standard deviation of how many entities each of the K samples found.

    The coarsest committee reading there is: it ignores where the entities are and asks
    only whether the samples agree on how much is going on in the sentence. Orthogonal
    enough to the position-wise metrics to be worth an arm — samples can agree on the
    count while disagreeing everywhere about the boundaries, and vice versa.
    """
    return {
        idx: pstdev([len(tag_spans(s)) for s in samples])
        for idx, samples in extract_tag_ids(records).items()
        if len(samples) > 1
    }


# --- controls -----------------------------------------------------------------

# `random` is registered with the metrics above, since it is the one the results are
# reported against. These two are controls in the same sense but answer narrower
# questions, and only make sense read next to an uncertainty arm.


@register("length")
def length_scores(records: Iterable[dict]) -> dict[int, float]:
    """The sentence's token count — the confounder arm, not an uncertainty measure.

    The budget is in sentences but the task is token-level NER, so any metric that
    correlates with length quietly buys its arm more labelled tokens for the same
    budget. Running "longest first" as its own arm is what tells an uncertainty result
    apart from a length result: if this arm matches the best metric, the finding is
    about sentence length.

    The count is the median over the K sampled arrays rather than the pool sentence,
    because a metric sees cache records and nothing else. The prompt demands one tag per
    token, so a sample's length *is* the token count whenever the model complied, and
    the median shrugs off the samples that did not.
    """
    return {
        idx: float(median(len(s) for s in samples))
        for idx, samples in extract_tag_ids(records).items()
        if samples
    }


@register("confident")
def confident_scores(
    records: Iterable[dict], *, metric: str = "avg_neg_logprob_filtered"
) -> dict[int, float]:
    """Another metric's ranking, reversed — most confident sentences first.

    The sharper test of whether an uncertainty signal is real. Against `random` a metric
    has to beat seed noise from a ~32-sentence fine-tune to show anything; against its
    own reverse the two arms move in opposite directions, so the same signal shows up as
    twice the gap. A metric that beats random while its reverse also beats random was
    measuring something other than uncertainty.

    The wrapped metric runs with its own defaults, and sentences it omits stay omitted.
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
