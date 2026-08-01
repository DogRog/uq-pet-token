"""Figures for a finished run — presentation only, never a source of numbers.

Everything here takes the results DataFrame `main.stage_train` produced and writes a
PNG. Nothing in this module feeds back into the pipeline, so a change to a color, a
label or a layout rule cannot move a reported score.

matplotlib is imported inside the plotting functions on purpose: selecting the Agg
backend is a global side effect, and importing this module must not have one.
"""

import logging
from pathlib import Path

import pandas as pd

from uq_pet.uncertainty import RANDOM

logger = logging.getLogger("uq_pet.plotting")

# Validated categorical palette (light surface). The control is pinned to orange so it
# reads as the baseline across runs; uncertainty arms take the remaining slots in order.
CONTROL_COLOR = "#eb6834"
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def arm_colors(arms: list[str]) -> dict[str, str]:
    """Assign a categorical hue per arm, with the random control pinned to orange.

    Colors follow the arm, not its rank in the results, so a run with more arms does
    not repaint the ones it shares with an earlier run.
    """
    palette = iter(SERIES_COLORS)
    colors = {}
    for arm in arms:
        if arm == RANDOM:
            colors[arm] = CONTROL_COLOR
            continue
        colors[arm] = next(palette, None) or "#7a7a75"
    if len(arms) > len(SERIES_COLORS) + 1:
        logger.warning("More arms than distinct hues; some share grey — consider faceting")
    return colors


def declutter(values: dict[str, float], min_gap: float) -> dict[str, float]:
    """Push overlapping label positions apart, keeping their relative order.

    Arms converge as the budget grows, so their end-of-line labels would otherwise
    print on top of each other and none of them would be readable.
    """
    placed: dict[str, float] = {}
    previous = None
    for name, value in sorted(values.items(), key=lambda kv: kv[1]):
        y = value if previous is None else max(value, previous + min_gap)
        placed[name] = y
        previous = y
    return placed


def _style_axes(ax) -> None:
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d9d4")


def plot_arms(results: pd.DataFrame, out_path: Path) -> None:
    """Single-budget view — bar = mean over seeds, dots = the individual seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(results["arm"]))
    colors = arm_colors(arms)
    sizes = results.groupby("arm")["n_train"].first()
    budget = results["budget_pct"].iloc[0]

    fig, ax = plt.subplots(figsize=(1.9 * len(arms) + 2.2, 4.4))
    for x, arm in enumerate(arms):
        seed_scores = results.loc[results["arm"] == arm, "entity_f1"]
        ax.bar(x, seed_scores.mean(), width=0.55, color=colors[arm], zorder=2)
        ax.scatter(
            [x] * len(seed_scores),
            seed_scores,
            color="white",
            edgecolor="#2b2b2b",
            linewidth=1.2,
            s=34,
            zorder=3,
        )
        # Direct label above the bar in ink, not on the colored fill.
        ax.text(
            x,
            seed_scores.mean() + 0.012,
            f"{seed_scores.mean():.3f}",
            ha="center",
            va="bottom",
            color=INK_PRIMARY,
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xticks(range(len(arms)), [_wrap_label(a) for a in arms], color=INK_PRIMARY)
    ax.set_ylabel("entity-level micro F1 (test split)", color=INK_SECONDARY)
    ax.set_title(
        f"Selection strategy vs. test F1 — {budget:.3g}% budget "
        f"({int(sizes.iloc[0])} sentences)",
        pad=14,
        color=INK_PRIMARY,
    )
    ax.set_ylim(0, max(results["entity_f1"].max() * 1.3, 0.05))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curve(results: pd.DataFrame, out_path: Path) -> None:
    """Budget sweep — one line per arm, mean over seeds, +/-1 std as a band and error bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = list(dict.fromkeys(results["arm"]))
    colors = arm_colors(arms)
    budgets = sorted(results["budget_pct"].unique())
    # Budgets sit on a true linear axis, so the distance between 25% and 30% is drawn
    # five times shorter than the distance between 30% and 55%. The slope now reads as
    # real return-per-percent-of-pool. The cost is the reverse of the old categorical
    # layout: closely spaced conditions crowd together, which is the honest picture of
    # how close they actually are.
    x = np.asarray(budgets, dtype=float)
    span = float(x[-1] - x[0]) or 1.0

    stats = results.groupby(["arm", "budget_pct"])["entity_f1"].agg(["mean", "std"])
    # A single-seed run has no spread; a zero-width band says exactly that.
    stats["std"] = stats["std"].fillna(0.0)

    top = max((stats["mean"] + stats["std"]).max() * 1.2, 0.05)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ends = {}
    for arm in arms:
        at_arm = stats.loc[arm].reindex(budgets)
        mean, std = at_arm["mean"].to_numpy(), at_arm["std"].to_numpy()
        # Band and error bar carry the same number: the spread over training seeds, not
        # over selections. Every seed trains on the same selected sentences, so this is
        # training noise alone. The band makes overlap between arms readable at a glance;
        # the caps keep each condition's own interval legible where the bands collide.
        ax.fill_between(x, 
                        mean - std, 
                        mean + std, 
                        color=colors[arm], 
                        alpha=0.18, 
                        linewidth=0, 
                        zorder=2)
        ax.errorbar(
            x, 
            mean, 
            yerr=std,
            color=colors[arm], 
            linewidth=2, 
            elinewidth=1.5, 
            capsize=4, 
            capthick=1.5, 
            zorder=3,
            label=arm,
        )
        ends[arm] = mean[-1]

    # Identity never rests on color alone: a legend is always present, and the lines are
    # additionally labelled at their ends *only* when those ends are far enough apart to
    # point at unambiguously. Arms converge as the budget grows, and a label nudged clear
    # of its neighbours no longer marks the line it belongs to.
    min_gap = top * 0.055
    spread = max(ends.values()) - min(ends.values())
    if len(arms) <= 4 and spread >= min_gap * (len(arms) - 1):
        label_x = x[-1] + span * 0.03
        for arm, y in declutter(ends, min_gap).items():
            ax.annotate(
                arm, (label_x, y), va="center", fontsize=9,
                color=INK_PRIMARY, annotation_clip=False,
            )
        right_pad = 0.34
    else:
        right_pad = 0.04

    ax.set_xticks(x, [f"{b:.3g}%" for b in budgets], color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("training budget (% of the 328-sentence pool)", color=INK_SECONDARY)
    ax.set_ylabel("entity-level micro F1 (test split)", color=INK_SECONDARY)
    ax.set_title(
        "Selection strategy vs. test F1 across budgets  (mean +/- 1 std over seeds)",
        pad=14,
        color=INK_PRIMARY,
    )
    ax.set_ylim(0, top)
    ax.set_xlim(x[0] - span * 0.04, x[-1] + span * (right_pad + 0.04))
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    """Bars for a single budget, a learning curve for a sweep."""
    if results["budget_pct"].nunique() > 1:
        plot_learning_curve(results, out_path)
    else:
        plot_arms(results, out_path)


def _wrap_label(label: str) -> str:
    return label.replace(":", ":\n", 1)
