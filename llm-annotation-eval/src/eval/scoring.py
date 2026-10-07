"""
Scores predicted mentions and relations against the gold annotations.

A predicted mention matches a gold mention of the same type if their tokens overlap at all.
A predicted relation matches a gold relation of the same type if both its head and its tail
overlap the gold head and tail. Matching is greedy in prediction order, and every gold
annotation can be matched only once.
"""
import collections
import typing

import numpy as np

from autoannotate import data
from eval.metrics import Stats


def _tokens(document: data.Document, mention_index: int) -> typing.Set[data.Token]:
    return set(document.tokens[i] for i in document.mentions[mention_index].token_document_indices)


def _stats_by_type(
        predicted_types: typing.List[str], gold_types: typing.List[str], num_ok: typing.Counter[str]
) -> typing.Dict[str, Stats]:
    num_pred = collections.Counter(predicted_types)
    num_gold = collections.Counter(gold_types)
    return {
        t: Stats(num_pred=num_pred[t], num_gold=num_gold[t], num_ok=num_ok[t])
        for t in sorted(num_pred.keys() | num_gold.keys())
    }


def mention_stats(gold: data.Document, predicted: data.Document) -> typing.Dict[str, Stats]:
    """:return: stats per mention type"""
    unmatched = list(range(len(gold.mentions)))
    num_ok: typing.Counter[str] = collections.Counter()
    for p, p_mention in enumerate(predicted.mentions):
        p_tokens = _tokens(predicted, p)
        for g in unmatched:
            if gold.mentions[g].type == p_mention.type and _tokens(gold, g) & p_tokens:
                num_ok[p_mention.type] += 1
                unmatched.remove(g)
                break
    return _stats_by_type([m.type for m in predicted.mentions], [m.type for m in gold.mentions], num_ok)


def relation_stats(gold: data.Document, predicted: data.Document) -> typing.Dict[str, Stats]:
    """:return: stats per relation type"""
    unmatched = list(gold.relations)
    num_ok: typing.Counter[str] = collections.Counter()
    for p_relation in predicted.relations:
        p_head = _tokens(predicted, p_relation.head_mention_index)
        p_tail = _tokens(predicted, p_relation.tail_mention_index)
        for g_relation in unmatched:
            if g_relation.type != p_relation.type:
                continue
            if not _tokens(gold, g_relation.head_mention_index) & p_head:
                continue
            if not _tokens(gold, g_relation.tail_mention_index) & p_tail:
                continue
            num_ok[p_relation.type] += 1
            unmatched.remove(g_relation)
            break
    return _stats_by_type([r.type for r in predicted.relations], [r.type for r in gold.relations], num_ok)


def total(stats_by_type: typing.Dict[str, Stats]) -> Stats:
    return sum(stats_by_type.values(), Stats(num_pred=0, num_gold=0, num_ok=0))


def _scores(stats: Stats) -> typing.Dict[str, float]:
    return {"precision": stats.precision, "recall": stats.recall, "f1": stats.f1}


def summarize(per_document: typing.List[typing.Dict[str, Stats]]) -> typing.Dict[str, typing.Any]:
    """
    :param per_document: stats per type for every scored document
    :return: "macro": mean and std (over documents) of the per-document scores, as printed by
             llm-metrics.py; "micro": scores of the counts summed over all documents;
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
        tasks: typing.Sequence[str] = ("md", "re"),
) -> typing.Dict[str, typing.Any]:
    """
    Scores mentions ("md") and/or relations ("re") of the predicted documents, which must be in the gold data.
    """
    gold_by_id = {d.id: d for d in gold_documents}
    unknown = [d.id for d in predicted_documents if d.id not in gold_by_id]
    assert not unknown, f"Predicted documents not in the gold data: {unknown}"
    pairs = [(gold_by_id[d.id], d) for d in predicted_documents]
    assert pairs, "No predicted documents to score."
    stats_by_type = {"md": mention_stats, "re": relation_stats}
    return {
        "documents": len(pairs),
        **{task: summarize([stats_by_type[task](g, p) for g, p in pairs]) for task in tasks},
    }
