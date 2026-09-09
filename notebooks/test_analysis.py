import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test performance: uncertainty vs random acquisition

    Select one saved experiment run. Curves show held-out test performance after bootstrap
    (round 0) and each acquisition round, with means and SD bands across model seeds.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from charts import (
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
    )

    alt.data_transformers.disable_max_rows()
    alt.renderers.set_embed_options(scaleFactor=2)
    return (
        Path,
        alt,
        json,
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
        mo,
        pl,
    )


@app.cell
def _(Path, json):
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results"
    available_runs = []
    # Only top-level legacy regular runs may omit the split marker.
    for results_path in sorted(results_root.glob("bert_token_uq_*/results.csv")):
        run_config = json.loads(results_path.with_name("config.json").read_text())
        if (
            run_config.get("evaluation_split", "test") == "test"
            and run_config.get("mode") != "optuna_tuning"
        ):
            available_runs.append(
                {
                    "source": "Regular experiments",
                    "run_id": results_path.parent.name,
                    "checkpoint": run_config["checkpoint"],
                    "uq_metric": run_config["uq_metric"],
                    "path": results_path,
                }
            )
    sweep_summaries = {}
    for summary_path in sorted((results_root / "random_search").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("evaluation_split") == "test":
            source = summary_path.parent.name
            sweep_summaries[source] = summary
            for comparison in summary["comparisons"]:
                if comparison["status"] == "complete":
                    run_path = summary_path.parent / comparison["run_dir"]
                    run_config = json.loads((run_path / "config.json").read_text())
                    if run_config.get("evaluation_split") != "test":
                        raise ValueError(f"Expected test evaluation in {run_path}")
                    available_runs.append(
                        {
                            "source": source,
                            "run_id": f"{source}/{comparison['config_id']}/{comparison['uq_metric']}",
                            "checkpoint": run_config["checkpoint"],
                            "uq_metric": comparison["uq_metric"],
                            "path": run_path / "results.csv",
                        }
                    )
    return available_runs, sweep_summaries


@app.cell
def _(available_runs, mo, sweep_summaries):
    source_options = sorted({run["source"] for run in available_runs} | set(sweep_summaries))
    mo.stop(
        not source_options,
        mo.md("No saved test runs. Run a regular experiment or a random sweep first."),
    )
    source_selector = mo.ui.dropdown(
        options=source_options, value=source_options[0], label="Experiment / sweep", full_width=True
    )
    source_selector
    return (source_selector,)


@app.cell
def _(alt, mo, pl, source_selector, sweep_summaries):
    selected_summary = sweep_summaries.get(source_selector.value)
    mo.stop(
        selected_summary is None,
        mo.md("Select a random sweep to see consistency across configurations."),
    )
    summary_table = pl.DataFrame(selected_summary["by_metric"])
    completed_comparisons = [
        row for row in selected_summary["comparisons"] if row["status"] == "complete"
    ]
    pending_comparisons = [
        {
            "config_id": row["config_id"],
            "uq_metric": row["uq_metric"],
            "status": row["status"],
            "error": row.get("error", ""),
        }
        for row in selected_summary["comparisons"]
        if row["status"] != "complete"
    ]
    summary_notice = mo.md(
        "All planned comparisons are complete. Each configuration has equal weight."
        if selected_summary["complete"]
        else "**Incomplete sweep:** these summaries cover completed comparisons only. "
        "Failed and pending comparisons are listed below; do not treat this as the full sweep."
    )
    consistency_plot = mo.md("No completed comparisons yet.")
    if completed_comparisons:
        gap_data = pl.DataFrame(
            [
                {
                    key: row[key]
                    for key in (
                        "config_id",
                        "uq_metric",
                        "mean_test_entity_f1_gap_auc",
                        "mean_final_test_entity_f1_gap",
                    )
                }
                for row in completed_comparisons
            ]
        )
        dots = (
            alt.Chart(gap_data)
            .mark_circle(size=70)
            .encode(
                x=alt.X(
                    "mean_test_entity_f1_gap_auc:Q", title="Test entity F1 gap AUC (UQ − random)"
                ),
                y=alt.Y("config_id:N", sort="ascending", title="Sampled configuration"),
                color=alt.Color("uq_metric:N", title="UQ metric"),
                tooltip=[
                    "config_id:N",
                    "uq_metric:N",
                    "mean_test_entity_f1_gap_auc:Q",
                    "mean_final_test_entity_f1_gap:Q",
                ],
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"zero": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="zero:Q")
        )
        consistency_plot = (dots + zero).properties(
            height=max(200, gap_data["config_id"].n_unique() * 22)
        )
    mo.vstack(
        [
            mo.md("## UQ benefit across all sampled configurations"),
            mo.md(
                "Positive gaps favor UQ. AUC is normalized by acquired-pool percentage and averaged across seeds. Wins, ties, and losses use each configuration's mean AUC gap; seeds are not counted as independent configurations."
            ),
            summary_notice,
            mo.ui.table(summary_table, selection=None),
            consistency_plot,
            mo.ui.table(pending_comparisons, selection=None) if pending_comparisons else mo.md(""),
        ]
    )
    return


@app.cell
def _(available_runs, mo, source_selector):
    test_runs = [run for run in available_runs if run["source"] == source_selector.value]
    mo.stop(not test_runs, mo.md("No completed runs in this sweep yet."))
    return (test_runs,)


@app.cell
def _(mo, test_runs):
    checkpoint_options = sorted({run["checkpoint"] for run in test_runs})
    checkpoint_selector = mo.ui.dropdown(
        options=checkpoint_options,
        value=checkpoint_options[0],
        label="Checkpoint",
        full_width=True,
    )
    checkpoint_selector
    return (checkpoint_selector,)


@app.cell
def _(checkpoint_selector, mo, test_runs):
    checkpoint_runs = [run for run in test_runs if run["checkpoint"] == checkpoint_selector.value]
    metric_options = sorted({run["uq_metric"] for run in checkpoint_runs})
    uq_metric_selector = mo.ui.dropdown(
        options=metric_options,
        value=metric_options[0],
        label="UQ metric",
        full_width=True,
    )
    uq_metric_selector
    return checkpoint_runs, uq_metric_selector


@app.cell
def _(checkpoint_runs, mo, uq_metric_selector):
    metric_runs = sorted(
        [run for run in checkpoint_runs if run["uq_metric"] == uq_metric_selector.value],
        key=lambda run: run["run_id"],
        reverse=True,
    )
    run_options = {run["run_id"]: index for index, run in enumerate(metric_runs)}
    run_selector = mo.ui.dropdown(
        options=run_options, value=next(iter(run_options)), label="Run", full_width=True
    )
    mo.vstack([mo.md("## Test learning curves across seeds"), run_selector])
    return metric_runs, run_selector


@app.cell
def _(json, metric_runs, mo, pl, run_selector):
    selected_run = metric_runs[run_selector.value]
    run_id = selected_run["run_id"]
    results = pl.read_csv(selected_run["path"]).with_columns(
        pl.lit(selected_run["checkpoint"]).alias("config_checkpoint"),
        pl.lit(selected_run["uq_metric"]).alias("config_uq_metric"),
        pl.lit(run_id).alias("run_id"),
    )
    selection_paths_by_run = {run_id: selected_run["path"].with_name("selections.json")}
    selected_config = json.loads(selected_run["path"].with_name("config.json").read_text())
    mo.ui.table(
        [
            {"setting": key, "value": json.dumps(selected_config[key])}
            for key in (
                "checkpoint",
                "model_seeds",
                "k",
                "bootstrap_epochs",
                "update_passes",
                "learning_rate",
                "batch_size",
                "replay_ratio",
                "weight_decay",
                "max_pool_percent",
            )
        ],
        selection=None,
        label="Selected run settings",
    )
    return results, selection_paths_by_run


@app.cell
def _(
    coverage_percent_slider,
    make_variance_chart,
    mo,
    pl,
    results,
    uq_metric_selector,
):
    chart_results = results.filter(pl.col("config_uq_metric") == uq_metric_selector.value)
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
                "**Macro entity F1 is unavailable for this completed run.** "
                "Its result files predate the metric and do not contain per-class "
                "predictions or counts. Rerun the experiment with the updated evaluator "
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

            The first chart shows the percentage acquired within each gold-label
            category, so rare tags remain visible. The original all-pool-share chart
            is shown underneath. Bars include zero counts across displayed runs.
            Gold labels are used only after acquisition.
            """),
            coverage_percent_slider,
        ]
    )
    return (coverage_percent_slider,)


@app.cell
def _(pl, results, selection_paths_by_run, uq_metric_selector):
    coverage_run_ids = (
        results.filter(pl.col("config_uq_metric") == uq_metric_selector.value)
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
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    uq_metric_selector,
):
    mo.vstack(
        [
            make_tag_category_coverage_chart(
                coverage_selections_with_progress,
                coverage_percent_slider.value,
                uq_metric_selector.value,
            ),
            make_tag_coverage_chart(
                coverage_selections_with_progress,
                coverage_percent_slider.value,
                uq_metric_selector.value,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
