"""LLM-alone evaluation: how good is the LLM as a NER tagger by itself, and
does its confidence track its correctness (or does it confidently fail)?

Works entirely from a score cache (see llm_scoring): the majority vote over
the K parsed samples is the LLM's prediction; the per-token agreement fraction
(majority tag count / K) is its black-box confidence. `train.evaluate` puts
the LLM on the same seqeval footing as the fine-tuned model. With K=5 the
confidence only takes values {0.2, 0.4, 0.6, 0.8, 1.0}, so calibration is
binned per unique agreement level rather than into arbitrary bins.
"""

from collections import Counter

import numpy as np
from scipy.stats import rankdata

from .train import evaluate
from .uncertainty import majority_vote


def llm_baseline_metrics(
    cache: dict[str, dict], exclude_keys: frozenset[str] = frozenset()
) -> dict:
    """seqeval metrics of the LLM's majority-vote prediction against gt_tags.

    Adds `n_sentences` and the mean±std entity F1 of the K individual samples
    (what self-consistency buys over a single greedy-ish draw)."""
    keys = [k for k in sorted(cache) if k not in exclude_keys]
    if not keys:
        return {"n_sentences": 0}
    predictions = [majority_vote(cache[k]["parsed_samples"]) for k in keys]
    gold = [cache[k]["gt_tags"] for k in keys]
    metrics = evaluate(predictions, gold)
    metrics["n_sentences"] = len(keys)

    num_samples = min(len(cache[k]["parsed_samples"]) for k in keys)
    single_f1s = [
        evaluate([cache[k]["parsed_samples"][i] for k in keys], gold)["entity_f1"]
        for i in range(num_samples)
    ]
    metrics["single_sample_f1_mean"] = float(np.mean(single_f1s))
    metrics["single_sample_f1_std"] = float(np.std(single_f1s))
    return metrics


def token_agreement_correctness(
    cache: dict[str, dict],
    exclude_keys: frozenset[str] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Flat per-token (confidence, correct) arrays over the whole cache:
    confidence = fraction of the K samples agreeing with the majority tag,
    correct = majority tag matches the ground-truth tag."""
    confidence, correct = [], []
    for key in sorted(cache):
        if key in exclude_keys:
            continue
        record = cache[key]
        samples = record["parsed_samples"]
        for i, gt_tag in enumerate(record["gt_tags"]):
            top_tag, top_count = Counter(s[i] for s in samples).most_common(1)[0]
            confidence.append(top_count / len(samples))
            correct.append(top_tag == gt_tag)
    return np.array(confidence), np.array(correct, dtype=bool)


def auroc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """P(confidence on a correct token > confidence on a wrong one), i.e.
    Mann-Whitney AUROC of confidence as a correctness classifier."""
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(confidence)
    u = ranks[correct].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def calibration_bins(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    """One bin per unique confidence level (the natural binning for K-sample
    agreement): [{"confidence", "accuracy", "count"}, ...] sorted by level."""
    return [
        {
            "confidence": float(level),
            "accuracy": float(correct[confidence == level].mean()),
            "count": int((confidence == level).sum()),
        }
        for level in np.unique(confidence)
    ]


def expected_calibration_error(bins: list[dict]) -> float:
    """Count-weighted mean |accuracy - confidence| over the bins."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    return sum(b["count"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins)


def confident_failure_stats(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Does the LLM confidently fail? Rates around unanimous (agreement 1.0)
    tokens: error rate given unanimity, and how many of all errors happen at
    full confidence (invisible to any agreement-based uncertainty metric)."""
    unanimous = confidence == 1.0
    errors = ~correct
    return {
        "unanimous_fraction": float(unanimous.mean()) if len(unanimous) else float("nan"),
        "confident_error_rate": (
            float(errors[unanimous].mean()) if unanimous.any() else float("nan")
        ),
        "share_of_errors_confident": (
            float((errors & unanimous).sum() / errors.sum()) if errors.any() else float("nan")
        ),
    }


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: answer the most confident fraction
    `coverage` of sentences, abstain on the rest; `risk` is the mean error over
    the answered ones. A useful uncertainty metric keeps risk low at low
    coverage; risk at coverage 1.0 equals the overall mean error."""
    order = np.argsort(uncertainty, kind="stable")
    n = len(order)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(error[order]) / np.arange(1, n + 1)
    return coverage, risk


def aurc(risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean risk over the uniform coverage
    grid); lower is better, equals overall error rate for a useless ranking."""
    return float(risk.mean()) if len(risk) else float("nan")
