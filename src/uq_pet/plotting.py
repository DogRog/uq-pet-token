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

# Matplotlib's classic categorical sequence, matching the experiment figures used in
# the paper draft. The random control is always blue; uncertainty arms then receive
# orange, green, red, and so on in config order.
CONTROL_COLOR = "#1f77b4"
SERIES_COLORS = [
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SURFACE = "#ffffff"
INK_PRIMARY = "#111111"
INK_SECONDARY = "#111111"


def arm_colors(arms: list[str]) -> dict[str, str]:
    """Assign a categorical hue per arm, with the random control pinned to blue.

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
    ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, direction="out", length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK_PRIMARY)
        spine.set_linewidth(1.0)


def _show_figure(fig) -> None:
    """Display a standalone Figure in Jupyter without importing pyplot.

    ``Figure.show()`` only works for figures created through pyplot because it needs a
    GUI manager. These figures deliberately bypass pyplot to avoid its global backend
    and figure-registry side effects, so notebook display must go through IPython.
    """
    from IPython.display import display

    display(fig)


def plot_arms(results: pd.DataFrame, out_path: Path, show: bool = False) -> None:
    """Single-budget view — bar = mean over seeds, dots = the individual seeds."""
    from matplotlib.figure import Figure

    arms = list(dict.fromkeys(results["arm"]))
    colors = arm_colors(arms)
    sizes = results.groupby("arm")["n_train"].first()
    budget = results["budget_pct"].iloc[0]

    fig = Figure(figsize=(1.9 * len(arms) + 2.2, 4.4), facecolor=SURFACE)
    ax = fig.subplots()
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
        f"Selection strategy vs. test F1 — {budget:.3g}% budget ({int(sizes.iloc[0])} sentences)",
        pad=14,
        color=INK_PRIMARY,
    )
    ax.set_ylim(0, max(results["entity_f1"].max() * 1.3, 0.05))
    _style_axes(ax)
    fig.tight_layout()
    if show:
        _show_figure(fig)
        return
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)


def plot_learning_curve(results: pd.DataFrame, out_path: Path, show: bool = False) -> None:
    """Budget sweep — one marked line per arm, mean over seeds, +/-1 std as a band."""
    import numpy as np
    from matplotlib.figure import Figure

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

    band_low = float((stats["mean"] - stats["std"]).min())
    band_high = float((stats["mean"] + stats["std"]).max())
    y_span = max(band_high - band_low, 0.05)
    # The legend gets its own panel rather than a corner of the axes. Inside the axes it
    # has to sit somewhere, and wherever that is, a run with enough arms eventually draws
    # a line through it — arms fan out at the smallest budget, which is exactly where an
    # "upper left" legend lives. The panel is sized from the longest arm name so a config
    # with `confident:mean_token_entropy` in it doesn't get its labels clipped.
    legend_width = 0.55 + 0.058 * max(len(arm) for arm in arms)
    fig = Figure(figsize=(7.6 + legend_width, 4.9), facecolor=SURFACE)
    ax, legend_ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": [7.6, legend_width]})
    legend_ax.axis("off")
    for arm in arms:
        at_arm = stats.loc[arm].reindex(budgets)
        mean, std = at_arm["mean"].to_numpy(), at_arm["std"].to_numpy()
        # Each model seed fits its own scorer and therefore may select different
        # sentences; the band includes both selection and continuation-training noise.
        ax.fill_between(
            x, mean - std, mean + std, color=colors[arm], alpha=0.14, linewidth=0, zorder=2
        )
        ax.plot(
            x,
            mean,
            color=colors[arm],
            linewidth=1.8,
            marker="o",
            markersize=6.5,
            zorder=3,
            label=arm,
        )

    ax.set_xticks(x, [f"{b:.3g}" for b in budgets], color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("Training budget (% of pool)", color=INK_SECONDARY)
    ax.set_ylabel("Entity-level micro F1 (test)", color=INK_SECONDARY)
    ax.set_title(
        "Uncertainty-based selection vs random — PET NER",
        pad=14,
        color=INK_PRIMARY,
    )
    ax.set_ylim(max(0.0, band_low - y_span * 0.08), min(1.0, band_high + y_span * 0.08))
    ax.set_xlim(x[0] - span * 0.04, x[-1] + span * 0.04)
    _style_axes(ax)
    legend_ax.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        labelcolor=INK_PRIMARY,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
        handlelength=2.2,
    )
    fig.tight_layout()
    if show:
        _show_figure(fig)
        return
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)


def plot_results(results: pd.DataFrame, out_path: Path, show: bool = False) -> None:
    """Bars for a single budget, a learning curve for a sweep."""
    if results["budget_pct"].nunique() > 1:
        plot_learning_curve(results, out_path, show)
    else:
        plot_arms(results, out_path, show)


def _wrap_label(label: str) -> str:
    """Break a long arm name over two lines so neighbouring x-ticks don't collide."""
    if ":" in label:
        return label.replace(":", ":\n", 1)
    head, sep, tail = label.rpartition("_")
    return f"{head}\n{tail}" if sep and len(label) > 16 else label
