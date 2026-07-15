"""Figures and tables that answer the research question.

Outputs (to results/<run_id>/figures/):
- learning_curves.png     entity-F1 vs budget, one line per strategy, ±std band,
                          dashed full-pool reference
- summary.md (also stdout)  mean±std F1 per cell + ΔF1 vs random, plus the
                          LLM-alone baseline (majority-vote F1 on pool/test and
                          confident-failure stats)
- uncertainty_vs_error.png  Spearman correlation between each metric's score and
                          per-sentence LLM error (does UQ track difficulty at all?)
- llm_calibration.png     reliability diagram of token-level sample agreement +
                          sentence-level risk-coverage curves per UQ metric
Also reports selected-subset mean sentence length per cell (length confound).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .config import FEW_SHOT_EXAMPLE_INDEX, ExperimentConfig, load_config
from .data import sentence_key, split_pool_test, tag_ids_to_labels
from .llm_eval import (
    aurc,
    auroc,
    calibration_bins,
    confident_failure_stats,
    expected_calibration_error,
    llm_baseline_metrics,
    risk_coverage,
    token_agreement_correctness,
)
from .llm_scoring import load_cache
from .uncertainty import METRICS, compute_metric, majority_vote, strategy_metric


def _strategy_label(strategy: str) -> str:
    """Human label with the metric's box type, e.g. 'UQ[white]: predictive_entropy'."""
    name = strategy_metric(strategy)
    if name is None:
        return strategy
    if name in METRICS:
        return f"UQ[{METRICS[name].box}]: {name}"
    return f"UQ: {name}"


def plot_ner_heatmap(ner, out_path=None):
    """Heatmap of entity-tag counts (B-/I- collapsed) per document."""
    tag_names = ner.features["ner-tags"].feature.names
    entity_types = sorted({t.removeprefix("B-").removeprefix("I-") for t in tag_names if t != "O"})
    docs = sorted(set(ner["document name"]))
    counts = np.zeros((len(docs), len(entity_types)))
    doc_idx = {d: i for i, d in enumerate(docs)}
    type_idx = {t: i for i, t in enumerate(entity_types)}
    for doc, tags in zip(ner["document name"], ner["ner-tags"], strict=True):
        for tid in tags:
            name = tag_names[tid]
            if name != "O":
                counts[doc_idx[doc], type_idx[name.removeprefix("B-").removeprefix("I-")]] += 1

    fig, ax = plt.subplots(figsize=(9, 0.25 * len(docs) + 2))
    im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(entity_types)), entity_types, rotation=45, ha="right")
    ax.set_yticks(range(len(docs)), docs, fontsize=6)
    ax.set_title("NER tag counts per document")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    return fig


def plot_selection_bias(by_key: dict[str, dict], selections: dict[str, list[str]], out_path=None):
    """What kind of sentences does each selection pick? Three panels comparing
    the named key subsets (e.g. pool vs random vs uncertainty): sentence-length
    distribution, entity density, and entity-type mix. Answers whether an
    uncertainty metric merely favors long / entity-dense sentences."""

    def subset_tags(keys):
        return [tag_ids_to_labels(by_key[k]["ner-tags"]) for k in keys]

    fig, (ax_len, ax_density, ax_types) = plt.subplots(1, 3, figsize=(16, 4.5))

    max_len = max(len(by_key[k]["tokens"]) for keys in selections.values() for k in keys)
    bins = np.linspace(0, max_len, 25)
    for name, keys in selections.items():
        lengths = [len(by_key[k]["tokens"]) for k in keys]
        ax_len.hist(
            lengths,
            bins=bins,
            density=True,
            alpha=0.45,
            label=f"{name} (mean {np.mean(lengths):.1f})",
        )
    ax_len.set_xlabel("tokens per sentence")
    ax_len.set_ylabel("density")
    ax_len.set_title("Sentence length")
    ax_len.legend(fontsize=8)

    densities = [
        np.mean([t != "O" for tags in subset_tags(keys) for t in tags])
        for keys in selections.values()
    ]
    ax_density.bar(range(len(selections)), densities, color="steelblue")
    ax_density.set_xticks(range(len(selections)), selections, rotation=20, ha="right", fontsize=8)
    ax_density.set_ylabel("fraction of non-O tokens")
    ax_density.set_title("Entity density")

    entity_types = sorted(
        {
            t.removeprefix("B-")
            for tags in subset_tags(selections[next(iter(selections))])
            for t in tags
            if t.startswith("B-")
        }
    )
    width = 0.8 / len(selections)
    for i, (name, keys) in enumerate(selections.items()):
        counts = Counter(
            t.removeprefix("B-") for tags in subset_tags(keys) for t in tags if t.startswith("B-")
        )
        total = sum(counts.values())
        shares = [counts.get(t, 0) / total if total else 0.0 for t in entity_types]
        ax_types.bar(np.arange(len(entity_types)) + i * width, shares, width, label=name)
    ax_types.set_xticks(
        np.arange(len(entity_types)) + 0.4 - width / 2,
        entity_types,
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax_types.set_ylabel("share of entities")
    ax_types.set_title("Entity-type mix")
    ax_types.legend(fontsize=8)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    return fig


def score_length_correlations(
    cache: dict[str, dict], scores_by_metric: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Spearman correlation between each metric's scores and sentence length.

    A high ρ means the metric's top-N selection is largely a longest-sentences
    selection — the length confound the summary table warns about."""
    correlations = {}
    for name, scores in scores_by_metric.items():
        keys = sorted(scores)
        lengths = [len(cache[k]["tokens"]) for k in keys]
        rho, _ = spearmanr([scores[k] for k in keys], lengths)
        correlations[name] = float(rho)
    return correlations


def load_runs(records_path: Path) -> list[dict]:
    with open(records_path) as f:
        return [json.loads(line) for line in f]


def _group_runs(runs: list[dict]) -> dict[tuple[int, str], list[dict]]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["budget_pct"], r["strategy"])].append(r)
    return groups


def plot_learning_curves(runs: list[dict], out_path) -> None:
    groups = _group_runs(runs)
    strategies = sorted({s for (_, s) in groups if s != "full"})
    budgets = sorted({b for (b, s) in groups if s != "full"})

    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy in strategies:
        means, stds = [], []
        for budget in budgets:
            f1s = [r["metrics"]["entity_f1"] for r in groups.get((budget, strategy), [])]
            means.append(np.mean(f1s) if f1s else np.nan)
            stds.append(np.std(f1s) if f1s else np.nan)
        means, stds = np.array(means), np.array(stds)
        ax.plot(budgets, means, marker="o", label=_strategy_label(strategy))
        ax.fill_between(budgets, means - stds, means + stds, alpha=0.15)

    full_runs = [r for (b, s), rs in groups.items() if s == "full" for r in rs]
    if full_runs:
        full_mean = np.mean([r["metrics"]["entity_f1"] for r in full_runs])
        ax.axhline(full_mean, linestyle="--", color="gray", label=f"full pool ({full_mean:.3f})")

    ax.set_xlabel("Training budget (% of pool)")
    ax.set_ylabel("Entity-level micro F1 (test)")
    ax.set_title("Uncertainty-based selection vs random — PET NER")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summary_table(runs: list[dict]) -> str:
    groups = _group_runs(runs)
    budgets = sorted({b for (b, s) in groups if s != "full"})
    strategies = sorted({s for (_, s) in groups if s != "full"}, key=lambda s: (s != "random", s))

    lines = [
        "| Budget | Strategy | Entity F1 (mean±std) | ΔF1 vs random | Token acc | Mean sent. len |",
        "|--------|----------|----------------------|---------------|-----------|----------------|",
    ]
    for budget in budgets:
        random_f1 = (
            np.mean([r["metrics"]["entity_f1"] for r in groups.get((budget, "random"), [])])
            if (budget, "random") in groups
            else np.nan
        )
        for strategy in strategies:
            cell = groups.get((budget, strategy))
            if not cell:
                continue
            f1s = [r["metrics"]["entity_f1"] for r in cell]
            accs = [r["metrics"]["token_accuracy"] for r in cell]
            lens = [r["selected_mean_tokens"] for r in cell]
            delta = np.mean(f1s) - random_f1
            delta_str = "—" if strategy == "random" else f"{delta:+.4f}"
            lines.append(
                f"| {budget}% | {strategy} | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
                f"| {delta_str} | {np.mean(accs):.4f} | {np.mean(lens):.1f} |"
            )
    full = [r for (b, s), rs in _group_runs(runs).items() if s == "full" for r in rs]
    if full:
        f1s = [r["metrics"]["entity_f1"] for r in full]
        accs = [r["metrics"]["token_accuracy"] for r in full]
        lines.append(
            f"| 100% | full pool | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | — "
            f"| {np.mean(accs):.4f} | {full[0]['selected_mean_tokens']:.1f} |"
        )
    return "\n".join(lines)


def _sentence_error_rates(cache: dict[str, dict], keys: list[str]) -> list[float]:
    """Per-sentence LLM error rate: fraction of tokens where the majority-vote
    tag disagrees with the ground truth."""
    error_rates = []
    for k in keys:
        record = cache[k]
        prediction = majority_vote(record["parsed_samples"])
        gt = record["gt_tags"]
        error_rates.append(sum(p != g for p, g in zip(prediction, gt, strict=True)) / len(gt))
    return error_rates


def _applicable_metric_names(cache: dict[str, dict], keys: list[str]) -> list[str]:
    """Registered metric names computable on this cache: white-box metrics need
    token_entropies (local backends only)."""
    has_entropies = all(cache[k].get("token_entropies") for k in keys)
    names = [
        name for name, metric in sorted(METRICS.items()) if metric.box != "white" or has_entropies
    ]
    skipped = sorted(set(METRICS) - set(names))
    if skipped:
        print(f"Skipping white-box metrics (no token_entropies in cache): {skipped}")
    return names


def plot_uncertainty_vs_error(cache: dict[str, dict], out_path) -> dict[str, float]:
    """Spearman correlation between each UQ metric and per-sentence LLM error rate."""
    keys = sorted(cache)
    error_rates = _sentence_error_rates(cache, keys)
    names = _applicable_metric_names(cache, keys)

    correlations = {}
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        scores = [compute_metric(name, cache[k]) for k in keys]
        rho, _ = spearmanr(scores, error_rates)
        correlations[name] = float(rho)
        ax.scatter(scores, error_rates, s=8, alpha=0.4)
        ax.set_title(f"{name} [{METRICS[name].box}]\nSpearman ρ={rho:.3f}")
        ax.set_xlabel("uncertainty")
    np.atleast_1d(axes)[0].set_ylabel("LLM majority-vote error rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return correlations


def _llm_metrics_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['entity_f1']:.4f} | {m['entity_precision']:.4f} "
        f"| {m['entity_recall']:.4f} | {m['token_accuracy']:.4f} | {m['n_sentences']} |"
    )


def llm_baseline_section(
    pool_cache: dict[str, dict],
    test_cache: dict[str, dict],
    exclude_keys: frozenset[str],
    model: str,
) -> str:
    """Markdown section: the LLM's own NER quality (same seqeval metrics as the
    fine-tuned model) plus how well its sample agreement tracks correctness."""
    pool_metrics = llm_baseline_metrics(pool_cache, exclude_keys)
    lines = [
        f"## LLM baseline: {model} (majority vote over K samples)",
        "",
        "| Predictor | Entity F1 | Precision | Recall | Token acc | Sentences |",
        "|-----------|-----------|-----------|--------|-----------|-----------|",
    ]
    if test_cache:
        lines.append(_llm_metrics_row("majority vote (test)", llm_baseline_metrics(test_cache)))
    lines.append(_llm_metrics_row("majority vote (pool)", pool_metrics))
    lines.append(
        f"| single sample (pool) | {pool_metrics['single_sample_f1_mean']:.4f} "
        f"± {pool_metrics['single_sample_f1_std']:.4f} | — | — | — "
        f"| {pool_metrics['n_sentences']} |"
    )

    confidence, correct = token_agreement_correctness(pool_cache, exclude_keys)
    bins = calibration_bins(confidence, correct)
    stats = confident_failure_stats(confidence, correct)
    lines += [
        "",
        f"- Token confidence (sample agreement, pool): "
        f"AUROC {auroc(confidence, correct):.3f}, ECE {expected_calibration_error(bins):.3f}",
        f"- {stats['unanimous_fraction']:.1%} of tokens are unanimous across samples; "
        f"{stats['confident_error_rate']:.1%} of those are wrong (confident failures)",
        f"- {stats['share_of_errors_confident']:.1%} of all LLM token errors happen at "
        f"full agreement — invisible to agreement-based uncertainty",
        "",
    ]
    caveat = (
        "Test rows share the eval set with the fine-tuned model above; pool rows "
        "cover the scored pool minus the few-shot prompt example."
    )
    if not test_cache:
        caveat += (
            " No test-split cache found — run `uq-pet score-pool --split test` "
            "for a direct comparison with the fine-tuned model."
        )
    lines.append(f"*{caveat}*")
    return "\n".join(lines)


def plot_llm_calibration(
    cache: dict[str, dict], out_path, exclude_keys: frozenset[str] = frozenset()
) -> None:
    """Left: reliability diagram of token-level sample agreement (is the LLM
    right when its samples agree?). Right: sentence-level risk-coverage curves
    — mean LLM error over the most-confident fraction of sentences, one curve
    per applicable UQ metric (lower AURC = better selective prediction)."""
    confidence, correct = token_agreement_correctness(cache, exclude_keys)
    bins = calibration_bins(confidence, correct)

    fig, (ax_rel, ax_rc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_rel.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax_rel.plot(
        [b["confidence"] for b in bins],
        [b["accuracy"] for b in bins],
        marker="o",
        color="steelblue",
        label="token accuracy",
    )
    for b in bins:
        ax_rel.annotate(
            f"n={b['count']}",
            (b["confidence"], b["accuracy"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=7,
        )
    ax_rel.set_xlabel("sample agreement (majority fraction)")
    ax_rel.set_ylabel("token accuracy")
    ax_rel.set_title(
        f"Reliability — AUROC {auroc(confidence, correct):.3f}, "
        f"ECE {expected_calibration_error(bins):.3f}"
    )
    ax_rel.set_xlim(0, 1.05)
    ax_rel.set_ylim(0, 1.05)
    ax_rel.legend(fontsize=8)
    ax_rel.grid(alpha=0.3)

    keys = [k for k in sorted(cache) if k not in exclude_keys]
    error_rates = np.array(_sentence_error_rates(cache, keys))
    for name in _applicable_metric_names(cache, keys):
        scores = np.array([compute_metric(name, cache[k]) for k in keys])
        coverage, risk = risk_coverage(scores, error_rates)
        ax_rc.plot(coverage, risk, label=f"{name} [{METRICS[name].box}] (AURC {aurc(risk):.3f})")
    ax_rc.set_xlabel("coverage (fraction answered, most confident first)")
    ax_rc.set_ylabel("risk (mean sentence error rate)")
    ax_rc.set_title("Risk–coverage (sentence level)")
    ax_rc.legend(fontsize=7)
    ax_rc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_report(run_dir: Path, cfg: ExperimentConfig | None = None) -> None:
    if cfg is None:
        snapshot = run_dir / "config.yaml"
        cfg = load_config(snapshot) if snapshot.exists() else ExperimentConfig()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(run_dir / "records.jsonl")
    print(f"Loaded {len(runs)} runs from {run_dir / 'records.jsonl'}\n")

    plot_learning_curves(runs, figures_dir / "learning_curves.png")
    print(f"Wrote {figures_dir / 'learning_curves.png'}")

    table = summary_table(runs)
    summary_sections = [table]
    print(table)

    cache = load_cache(cfg.llm.cache_path())
    if cache:
        correlations = plot_uncertainty_vs_error(cache, figures_dir / "uncertainty_vs_error.png")
        print(f"\nWrote {figures_dir / 'uncertainty_vs_error.png'}")
        print(
            "Spearman(uncertainty, LLM error):", {k: round(v, 3) for k, v in correlations.items()}
        )

        # The few-shot example appears verbatim in every prompt, so it is
        # excluded from the LLM's own evaluation.
        try:
            pool, _ = split_pool_test(seed=cfg.llm.seed)
            exclude = frozenset({sentence_key(pool[FEW_SHOT_EXAMPLE_INDEX])})
        except FileNotFoundError:
            print("Raw dataset not found — few-shot example not excluded from LLM baseline.")
            exclude = frozenset()
        if any(k not in exclude for k in cache):
            test_cache = load_cache(cfg.llm.cache_path("test"))
            section = llm_baseline_section(cache, test_cache, exclude, cfg.llm.model)
            summary_sections.append(section)
            print("\n" + section)
            plot_llm_calibration(cache, figures_dir / "llm_calibration.png", exclude)
            print(f"\nWrote {figures_dir / 'llm_calibration.png'}")

    (figures_dir / "summary.md").write_text("\n\n".join(summary_sections) + "\n")
    print(f"\nWrote {figures_dir / 'summary.md'}")
