"""Scores predicted mentions against the gold annotations.

A predicted mention matches a gold mention of the same type if their tokens overlap at all
(or, with exact=True, are the same tokens). Matching is greedy in prediction order, and every
gold mention can be matched only once.
"""
import collections
import typing

import numpy as np

from autoannotate import data
from eval.metrics import Stats


def _tokens(document: data.Document, mention_index: int) -> typing.Set[data.Token]:
    return set(document.tokens[i] for i in document.mentions[mention_index].token_document_indices)


def _match(gold: typing.Set[data.Token], predicted: typing.Set[data.Token], exact: bool) -> bool:
    return gold == predicted if exact else bool(gold & predicted)


def _stats_by_type(
        predicted_types: typing.List[str], gold_types: typing.List[str], num_ok: typing.Counter[str]
) -> typing.Dict[str, Stats]:
    num_pred = collections.Counter(predicted_types)
    num_gold = collections.Counter(gold_types)
    return {
        t: Stats(num_pred=num_pred[t], num_gold=num_gold[t], num_ok=num_ok[t])
        for t in sorted(num_pred.keys() | num_gold.keys())
    }


def mention_stats(
        gold: data.Document, predicted: data.Document, exact: bool = False
) -> typing.Dict[str, Stats]:
    """:return: stats per mention type"""
    unmatched = list(range(len(gold.mentions)))
    num_ok: typing.Counter[str] = collections.Counter()
    for p, p_mention in enumerate(predicted.mentions):
        p_tokens = _tokens(predicted, p)
        for g in unmatched:
            g_mention = gold.mentions[g]
            if g_mention.type == p_mention.type and _match(_tokens(gold, g), p_tokens, exact):
                num_ok[p_mention.type] += 1
                unmatched.remove(g)
                break
    return _stats_by_type([m.type for m in predicted.mentions], [m.type for m in gold.mentions], num_ok)


def total(stats_by_type: typing.Dict[str, Stats]) -> Stats:
    return sum(stats_by_type.values(), Stats(num_pred=0, num_gold=0, num_ok=0))


def _scores(stats: Stats) -> typing.Dict[str, float]:
    return {"precision": stats.precision, "recall": stats.recall, "f1": stats.f1}


def summarize(per_document: typing.List[typing.Dict[str, Stats]]) -> typing.Dict[str, typing.Any]:
    """Aggregates per-document stats into macro, micro and per-type scores.

    Args:
        per_document: stats per type for every scored document

    Returns:
        "macro": mean and std (over documents) of the per-document scores, as the results
        notebook shows them; "micro": scores of the counts summed over all documents;
        "per_type": micro scores and counts per type
    """
    totals = [total(stats) for stats in per_document]
    macro = {}
    for name in ("precision", "recall", "f1"):
        values = [getattr(stats, name) for stats in totals]
        macro[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    by_type: typing.Dict[str, Stats] = collections.defaultdict(lambda: Stats(num_pred=0, num_gold=0, num_ok=0))
    for stats in per_document:
        for t, s in stats.items():
            by_type[t] = by_type[t] + s
    return {
        "macro": macro,
        "micro": _scores(total(by_type)),
        "per_type": {
            t: {**_scores(s), "num_gold": s.num_gold, "num_pred": s.num_pred, "num_ok": s.num_ok}
            for t, s in sorted(by_type.items())
        },
    }


def score(
        gold_documents: typing.List[data.Document], predicted_documents: typing.List[data.Document],
        exact: bool = False,
) -> typing.Dict[str, typing.Any]:
    """Scores the mentions of the predicted documents, under "md".

    The predicted documents must all be in the gold data. Mentions match if their tokens
    overlap, or with exact=True only if they are the same tokens.
    """
    gold_by_id = {d.id: d for d in gold_documents}
    unknown = [d.id for d in predicted_documents if d.id not in gold_by_id]
    assert not unknown, f"Predicted documents not in the gold data: {unknown}"
    pairs = [(gold_by_id[d.id], d) for d in predicted_documents]
    assert pairs, "No predicted documents to score."
    return {
        "documents": len(pairs),
        "md": summarize([mention_stats(g, p, exact) for g, p in pairs]),
    }
