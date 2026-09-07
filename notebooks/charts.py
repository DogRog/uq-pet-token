"""Shared charts and W&B comparison media for the marimo notebooks."""

import altair as alt
import polars as pl

from uq_pet.experiment import summarize_label_pool_share


def make_learning_chart(result_records: list[dict], seed: int, uq_metric: str):
    """Build the paired entity-F1, macro-F1, and accuracy chart for one model seed."""
    records = (
        pl.DataFrame(result_records)
        if result_records
        else pl.DataFrame(
            schema={
                "seed": pl.Int64,
                "arm": pl.String,
                "percent_acquired": pl.Float64,
                "n_acquired": pl.Int64,
                "entity_f1": pl.Float64,
                "entity_macro_f1": pl.Float64,
                "token_accuracy": pl.Float64,
            }
        )
    )
    curve = (
        records.filter(pl.col("seed") == seed)
        .with_columns(pl.col("arm").replace({"uncertainty": uq_metric}))
        .sort(["arm", "percent_acquired"])
    )
    arm_order = ["random", uq_metric]
    arm_color = alt.Color(
        "arm:N",
        title=None,
        sort=arm_order,
        scale=alt.Scale(domain=arm_order, range=["#4C78A8", "#F58518"]),
        legend=alt.Legend(orient="top"),
    )
    shared_x = alt.X(
        "percent_acquired:Q",
        title="Scoreable pool acquired (%)",
        scale=alt.Scale(domain=[0, 100]),
    )

    def metric_chart(field: str, title: str, y_title: str):
        return (
            alt.Chart(curve)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=shared_x,
                y=alt.Y(f"{field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                color=arm_color,
                order=alt.Order("percent_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
                    alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                    alt.Tooltip(f"{field}:Q", title=title, format=".3f"),
                ],
            )
            .properties(title=title, width=500, height=320)
        )

    return alt.hconcat(
        metric_chart("entity_f1", "Entity F1", "F1"),
        metric_chart("entity_macro_f1", "Macro entity F1", "Macro F1"),
        metric_chart("token_accuracy", "Token accuracy", "Accuracy"),
        spacing=35,
    ).resolve_scale(color="shared")


def make_tag_coverage_chart(
    selections: pl.DataFrame,
    acquisition_percent: float,
    uq_metric: str,
    label_order: list[str] | None = None,
):
    """Compare revealed label shares across arms in either notebook."""
    summary = summarize_label_pool_share(selections, acquisition_percent).with_columns(
        pl.col("arm").replace({"uncertainty": uq_metric})
    )
    if label_order is None:
        label_order = (
            summary.group_by("label")
            .agg(pl.col("n_acquired_mean").mean().alias("count"))
            .sort("count", descending=True)
            .get_column("label")
            .to_list()
        )
    arm_order = ["random", uq_metric]
    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "pool_share_mean:Q",
                title="All scoreable pool tokens (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("label:N", title="Gold label", sort=label_order),
            yOffset=alt.YOffset("arm:N", sort=arm_order),
            color=alt.Color(
                "arm:N",
                title=None,
                sort=arm_order,
                scale=alt.Scale(domain=arm_order, range=["#4C78A8", "#F58518"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("label:N", title="Gold label"),
                alt.Tooltip("pool_share_mean:Q", title="Mean pool share", format=".3f"),
                alt.Tooltip("pool_share_std:Q", title="Pool-share std. dev.", format=".3f"),
                alt.Tooltip("n_acquired_mean:Q", title="Mean acquired", format=".1f"),
                alt.Tooltip(
                    "scoreable_pool_tokens_mean:Q", title="Scoreable pool tokens", format=".1f"
                ),
            ],
        )
        .properties(
            title=f"Labels revealed by {acquisition_percent:.1f}% pool acquisition",
            width=1000,
            height=max(380, 28 * len(label_order)),
        )
    )


def make_variance_chart(
    results_frame: pl.DataFrame, uq_metric: str, acquisition_percent: float | None = None
):
    """Plot seed means and clipped ±1 SD bands, including older runs without macro F1."""
    metrics = [
        (field, title, y_title)
        for field, title, y_title in (
            ("entity_f1", "Entity F1", "F1"),
            ("entity_macro_f1", "Macro entity F1", "Macro F1"),
            ("token_accuracy", "Token accuracy", "Accuracy"),
        )
        if field != "entity_macro_f1" or field in results_frame.columns
    ]
    summary = (
        results_frame.with_columns(pl.col("arm").replace({"uncertainty": uq_metric}))
        .group_by(["arm", "n_acquired", "percent_acquired"])
        .agg(
            expression
            for field, _, _ in metrics
            for expression in (
                pl.col(field).mean().alias(f"{field}_mean"),
                pl.col(field).std().fill_null(0.0).alias(f"{field}_std"),
            )
        )
        .with_columns(
            (pl.col(f"{field}_mean") + sign * pl.col(f"{field}_std"))
            .clip(0.0, 1.0)
            .alias(f"{field}_{bound}")
            for field, _, _ in metrics
            for bound, sign in (("lower", -1), ("upper", 1))
        )
        .sort(["arm", "percent_acquired"])
    )
    arm_order = ["random", uq_metric]
    base = alt.Chart(summary).encode(
        x=alt.X(
            "percent_acquired:Q",
            title="Scoreable pool acquired (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "arm:N",
            title=None,
            sort=arm_order,
            scale=alt.Scale(domain=arm_order, range=["#4C78A8", "#F58518"]),
            legend=alt.Legend(orient="top"),
        ),
    )

    def metric_chart(field, title, y_title):
        band = base.mark_area(opacity=0.18).encode(
            y=alt.Y(f"{field}_lower:Q", title=y_title, scale=alt.Scale(zero=False)),
            y2=alt.Y2(f"{field}_upper:Q"),
        )
        mean_line = base.mark_line(strokeWidth=3).encode(
            y=alt.Y(f"{field}_mean:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
                alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                alt.Tooltip(f"{field}_mean:Q", title="Mean", format=".3f"),
                alt.Tooltip(f"{field}_std:Q", title="Std. dev.", format=".3f"),
            ],
        )
        layers = [band, mean_line]
        if acquisition_percent is not None:
            layers.append(
                alt.Chart(pl.DataFrame({"selected_percent": [float(acquisition_percent)]}))
                .mark_rule(color="#E45756", strokeDash=[7, 5], strokeWidth=2)
                .encode(
                    x=alt.X("selected_percent:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("selected_percent:Q", title="Coverage budget (%)", format=".0f")
                    ],
                )
            )
        return alt.layer(*layers).properties(title=title, width=500, height=320)

    return alt.hconcat(*(metric_chart(*metric) for metric in metrics), spacing=35).resolve_scale(
        color="shared"
    )


def make_wandb_comparison_media(wandb_module, result_records: list[dict], uq_metric: str) -> dict:
    """Build one table-free interactive comparison panel per model seed."""
    return {
        f"final_comparison/seed_{seed}_{uq_metric}_vs_random": wandb_module.Html(
            make_learning_chart(result_records, seed, uq_metric).to_html(),
            inject=False,
        )
        for seed in sorted({row["seed"] for row in result_records})
    }
