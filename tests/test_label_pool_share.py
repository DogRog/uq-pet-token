import polars as pl
import pytest

from uq_pet.experiment import summarize_label_category_coverage, summarize_label_pool_share


def test_pool_share_uses_full_pool_and_includes_zero_counts():
    selections = pl.DataFrame(
        [
            {
                "run_id": f"run-{seed}-{arm}",
                "seed": seed,
                "arm": arm,
                "label": label,
                "scoreable_pool_tokens": 10,
                "percent_acquired": 10 * (idx + 1),
            }
            for seed in (0, 1)
            for arm in ("uncertainty", "random")
            for idx, label in enumerate(["B-Actor", "O"] if seed == 0 else ["O", "O"])
        ]
    )
    summary = summarize_label_pool_share(selections, 20)
    for arm in ("uncertainty", "random"):
        arm_summary = summary.filter(pl.col("arm") == arm)
        actor = arm_summary.filter(pl.col("label") == "B-Actor").row(0, named=True)
        assert actor["pool_share_mean"] == pytest.approx(5)
        assert actor["n_acquired_mean"] == pytest.approx(0.5)
        assert arm_summary["pool_share_mean"].sum() == pytest.approx(20)
    assert summarize_label_pool_share(selections, 0)["pool_share_mean"].sum() == 0
    assert summarize_label_pool_share(selections, 10).filter(pl.col("arm") == "random")[
        "pool_share_mean"
    ].sum() == pytest.approx(10)


def test_pool_share_accounts_for_smaller_final_round():
    selections = pl.DataFrame(
        [
            {
                "run_id": "run-0",
                "seed": 0,
                "arm": arm,
                "label": "O",
                "scoreable_pool_tokens": 5,
                "percent_acquired": percent,
            }
            for arm in ("uncertainty", "random")
            for percent in (40, 40, 80, 80, 100)
        ]
    )
    for budget, expected in ((80, 80), (99, 80), (100, 100)):
        summary = summarize_label_pool_share(selections, budget)
        assert summary["pool_share_mean"].to_list() == [expected, expected]


def test_label_category_coverage_uses_each_labels_total_and_zero_counts():
    selections = pl.DataFrame(
        [
            {
                "run_id": "run-0",
                "seed": seed,
                "arm": arm,
                "label": label,
                "pool_idx": pool_idx,
                "word_idx": 0,
                "percent_acquired": 10 * (idx + 1),
            }
            for seed in (0, 1)
            for arm in ("uncertainty", "random")
            for idx, (pool_idx, label) in enumerate(
                [(0, "B-Actor"), (1, "O")] if seed == 0 else [(1, "O"), (0, "B-Actor")]
            )
        ]
    )
    summary = summarize_label_category_coverage(selections, 10)
    for arm in ("uncertainty", "random"):
        arm_summary = summary.filter(pl.col("arm") == arm)
        actor = arm_summary.filter(pl.col("label") == "B-Actor").row(0, named=True)
        assert actor["label_coverage_mean"] == pytest.approx(50)
        assert actor["n_acquired_mean"] == pytest.approx(0.5)
        assert actor["n_label_tokens_mean"] == pytest.approx(1)
    assert summarize_label_category_coverage(selections, 0)[
        "label_coverage_mean"
    ].sum() == pytest.approx(0)
    assert summarize_label_category_coverage(selections, 20)["label_coverage_mean"].to_list() == [
        100,
        100,
        100,
        100,
    ]


def test_label_category_coverage_accounts_for_smaller_final_round():
    selections = pl.DataFrame(
        [
            {
                "run_id": "run-0",
                "seed": 0,
                "arm": arm,
                "label": "O",
                "pool_idx": idx,
                "word_idx": 0,
                "percent_acquired": percent,
            }
            for arm in ("uncertainty", "random")
            for idx, percent in enumerate((40, 40, 80, 80, 100))
        ]
    )
    for budget, expected in ((80, 80), (99, 80), (100, 100)):
        summary = summarize_label_category_coverage(selections, budget)
        assert summary["label_coverage_mean"].to_list() == [expected, expected]
