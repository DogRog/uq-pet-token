import pandas as pd

from uq_pet.plotting import arm_colors, plot_results

ANLP = "avg_neg_logprob_filtered"
PURE = "avg_neg_logprob_pure"


def make_results(uncertainty_f1, random_f1, budget: float = 10.0) -> pd.DataFrame:
    """Only the columns the figures read: one row per (arm, seed) at one budget."""
    rows = []
    for arm, scores in ((ANLP, uncertainty_f1), ("random", random_f1)):
        for seed, f1 in enumerate(scores):
            rows.append(
                {
                    "budget_pct": budget,
                    "arm": arm,
                    "seed": seed,
                    "n_train": int(328 * budget / 100),
                    "entity_f1": f1,
                }
            )
    return pd.DataFrame(rows)


# --- colors -------------------------------------------------------------------


def test_arm_colors_pin_random_and_are_distinct():
    colors = arm_colors(["random", ANLP, PURE])
    assert colors["random"] == "#eb6834"
    assert len(set(colors.values())) == 3


def test_arm_colors_follow_the_arm_not_its_position():
    """Dropping an arm must not repaint the survivors."""
    full = arm_colors(["random", ANLP, PURE])
    fewer = arm_colors(["random", ANLP])
    assert fewer[ANLP] == full[ANLP]


# --- figures ------------------------------------------------------------------


def test_plot_results_writes_bars_for_one_budget_and_a_curve_for_a_sweep(tmp_path):
    single = tmp_path / "bars.png"
    plot_results(make_results([0.6, 0.7], [0.4, 0.5]), single)
    assert single.stat().st_size > 0

    sweep = tmp_path / "curve.png"
    plot_results(
        pd.concat(
            [
                make_results([0.6, 0.65], [0.4, 0.5], budget=5.0),
                make_results([0.5, 0.58], [0.55, 0.6], budget=25.0),
            ]
        ),
        sweep,
    )
    assert sweep.stat().st_size > 0


def test_plot_learning_curve_survives_a_single_seed(tmp_path):
    """One seed means no std; the error bars must degenerate, not raise on NaN."""
    out = tmp_path / "curve.png"
    plot_results(
        pd.concat([make_results([0.6], [0.4], budget=5.0), make_results([0.5], [0.55], 25.0)]),
        out,
    )
    assert out.stat().st_size > 0
