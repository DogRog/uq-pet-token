import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fixed-all-metrics sweep: uncertainty vs random acquisition
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from charts import make_tag_coverage_chart, make_variance_chart

    alt.data_transformers.disable_max_rows()
    alt.renderers.set_embed_options(scaleFactor=2)
    return Path, make_tag_coverage_chart, make_variance_chart, mo, pl


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parents[1]
    sweep_dir = (
        project_root
        / "results"
        / "sweeps"
        / "bert_token_uq_20260825_090329_945236_fixed-all-metrics"
    )
    combined_path = sweep_dir / "combined_results.csv"
    selection_paths_by_run = {
        path.parents[1].name: path
        for path in sweep_dir.glob("runs/*/bert_token_uq_*/selections.json")
    }
    return combined_path, selection_paths_by_run


@app.cell
def _(combined_path, pl):
    results = pl.read_csv(combined_path)
    return (results,)


@app.cell
def _(mo, results):
    checkpoint_options = sorted(set(results.get_column("config_checkpoint").unique().to_list()))
    checkpoint_selector = mo.ui.dropdown(
        options=checkpoint_options,
        value="distilbert-base-cased",
        label="Checkpoint",
        full_width=True,
    )
    uq_metric_selector = mo.ui.dropdown(
        options=sorted(results.get_column("config_uq_metric").unique().to_list()),
        value="entropy",
        label="UQ metric",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## Learning curves across seeds"),
            mo.hstack(
                [checkpoint_selector, uq_metric_selector],
                justify="start",
                widths=[2, 1],
            ),
        ]
    )
    return checkpoint_selector, uq_metric_selector


@app.cell
def _(
    checkpoint_selector,
    coverage_percent_slider,
    make_variance_chart,
    mo,
    pl,
    results,
    uq_metric_selector,
):
    chart_results = results.filter(
        (pl.col("config_checkpoint") == checkpoint_selector.value)
        & (pl.col("config_uq_metric") == uq_metric_selector.value)
    )
    learning_curve_chart = make_variance_chart(
        chart_results,
        uq_metric_selector.value,
        coverage_percent_slider.value,
    )
    macro_f1_notice = (
        mo.md("")
        if "entity_macro_f1" in chart_results.columns
        else mo.callout(
            mo.md(
                "**Macro entity F1 is unavailable for this completed sweep.** "
                "Its result files predate the metric and do not contain per-class "
                "predictions or counts. Rerun the sweep with the updated evaluator "
                "to populate the third curve."
            ),
            kind="warn",
        )
    )
    mo.vstack([macro_f1_notice, learning_curve_chart])
    return


@app.cell
def _(mo):
    coverage_percent_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=25,
        label="Scoreable pool acquired (%)",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(r"""
            ## Cumulative NER-tag coverage

            Each bar is the acquired count for that gold label as a percentage of
            all scoreable pool tokens. Bars show the mean across the displayed runs,
            including zero counts. Gold labels are used only after acquisition.
            """),
            coverage_percent_slider,
        ]
    )
    return (coverage_percent_slider,)


@app.cell
def _(
    checkpoint_selector,
    pl,
    results,
    selection_paths_by_run,
    uq_metric_selector,
):
    coverage_run_ids = (
        results.filter(
            (pl.col("config_checkpoint") == checkpoint_selector.value)
            & (pl.col("config_uq_metric") == uq_metric_selector.value)
        )
        .get_column("run_id")
        .unique()
        .sort()
        .to_list()
    )
    coverage_selection_frames = [
        pl.read_json(selection_paths_by_run[coverage_run_id]).with_columns(
            pl.lit(coverage_run_id).alias("run_id")
        )
        for coverage_run_id in coverage_run_ids
    ]
    coverage_selections = pl.concat(coverage_selection_frames)
    coverage_round_progress = (
        results.filter(pl.col("run_id").is_in(coverage_run_ids))
        .select("run_id", "seed", "arm", "round", "percent_acquired", "scoreable_pool_tokens")
        .unique()
    )
    coverage_selections_with_progress = coverage_selections.join(
        coverage_round_progress,
        on=["run_id", "seed", "arm", "round"],
        how="left",
        validate="m:1",
    )
    return (coverage_selections_with_progress,)


@app.cell
def _(
    coverage_percent_slider,
    coverage_selections_with_progress,
    make_tag_coverage_chart,
    uq_metric_selector,
):
    make_tag_coverage_chart(
        coverage_selections_with_progress,
        coverage_percent_slider.value,
        uq_metric_selector.value,
    )
    return


if __name__ == "__main__":
    app.run()
