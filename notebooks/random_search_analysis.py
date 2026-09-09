import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random-search test analysis

    Inspect one saved random sweep without rerunning training. Sweep-level views give every
    sampled configuration equal weight; the drill-down shows held-out test curves and acquired
    NER-tag coverage for one UQ metric and configuration.
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
    random_search_root = project_root / "results" / "random_search"
    sweep_paths = {}
    for summary_path in sorted(random_search_root.glob("*/summary.json")):
        summary_metadata = json.loads(summary_path.read_text())
        if summary_metadata.get("evaluation_split") == "test":
            sweep_paths[summary_path.parent.name] = summary_path.parent
    return (sweep_paths,)


@app.cell
def _(mo, sweep_paths):
    sweep_options = sorted(sweep_paths, reverse=True)
    mo.stop(
        not sweep_options,
        mo.md("No saved random-search test sweeps were found under `results/random_search/`."),
    )
    sweep_selector = mo.ui.dropdown(
        options=sweep_options,
        value=sweep_options[0],
        label="Random sweep",
        full_width=True,
    )
    sweep_selector
    return (sweep_selector,)


@app.cell
def _(json, sweep_paths, sweep_selector):
    sweep_dir = sweep_paths[sweep_selector.value]
    sweep_summary = json.loads((sweep_dir / "summary.json").read_text())
    sweep_plan = json.loads((sweep_dir / "plan.json").read_text())
    return sweep_dir, sweep_plan, sweep_summary


@app.cell
def _(alt, mo, pl, sweep_plan, sweep_summary):
    completed_comparisons = [
        comparison
        for comparison in sweep_summary["comparisons"]
        if comparison["status"] == "complete"
    ]
    incomplete_comparisons = [
        {
            "config_id": comparison["config_id"],
            "uq_metric": comparison["uq_metric"],
            "status": comparison["status"],
            "error": comparison.get("error", ""),
        }
        for comparison in sweep_summary["comparisons"]
        if comparison["status"] != "complete"
    ]
    completion_count = len(completed_comparisons)
    planned_count = len(sweep_summary["comparisons"])
    completion_notice = (
        mo.callout(
            mo.md(
                f"**Complete sweep:** {completion_count}/{planned_count} metric/configuration "
                "comparisons finished."
            ),
            kind="success",
        )
        if sweep_summary["complete"]
        else mo.callout(
            mo.md(
                f"**Incomplete sweep:** {completion_count}/{planned_count} comparisons finished. "
                "Aggregate values below cover completed comparisons only."
            ),
            kind="warn",
        )
    )
    metric_summary = pl.DataFrame(sweep_summary["by_metric"])
    consistency_plot = mo.md("No completed comparisons yet.")
    if completed_comparisons:
        gap_rows = pl.DataFrame(
            [
                {
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "mean_test_entity_f1_gap_auc": comparison["mean_test_entity_f1_gap_auc"],
                    "mean_final_test_entity_f1_gap": comparison["mean_final_test_entity_f1_gap"],
                }
                for comparison in completed_comparisons
            ]
        )
        gap_dots = (
            alt.Chart(gap_rows)
            .mark_circle(size=70)
            .encode(
                x=alt.X(
                    "mean_test_entity_f1_gap_auc:Q",
                    title="Test entity F1 gap AUC (UQ − random)",
                ),
                y=alt.Y("config_id:N", sort="ascending", title="Sampled configuration"),
                color=alt.Color("uq_metric:N", title="UQ metric"),
                tooltip=[
                    alt.Tooltip("config_id:N", title="Configuration"),
                    alt.Tooltip("uq_metric:N", title="UQ metric"),
                    alt.Tooltip(
                        "mean_test_entity_f1_gap_auc:Q", title="Mean gap AUC", format=".4f"
                    ),
                    alt.Tooltip(
                        "mean_final_test_entity_f1_gap:Q",
                        title="Mean final gap",
                        format=".4f",
                    ),
                ],
            )
        )
        zero_rule = (
            alt.Chart(pl.DataFrame({"zero": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="zero:Q")
        )
        consistency_plot = (gap_dots + zero_rule).properties(
            height=max(240, gap_rows["config_id"].n_unique() * 22)
        )
    plan_details = {
        "Sampling": sweep_plan["sampling"],
        "Sampler seed": sweep_plan["sampler_seed"],
        "Sampled configurations": sweep_plan["num_configs"],
        "UQ metrics": ", ".join(sweep_plan["uq_metrics"]),
        "Effective precision": sweep_plan.get("effective_precision", "not recorded"),
    }
    mo.vstack(
        [
            mo.md("## Sweep overview"),
            completion_notice,
            mo.ui.table([plan_details], selection=None),
            mo.md("### Aggregate results by UQ metric"),
            mo.ui.table(metric_summary, selection=None),
            mo.md("### UQ benefit across sampled configurations"),
            mo.md(
                "Positive gaps favor UQ. AUC is normalized by acquired-pool percentage and "
                "averaged across model seeds."
            ),
            mo.ui.altair_chart(consistency_plot),
            mo.vstack(
                [
                    mo.md("### Failed or pending comparisons"),
                    mo.ui.table(incomplete_comparisons, selection=None),
                ]
            )
            if incomplete_comparisons
            else mo.md(""),
        ]
    )
    return (completed_comparisons,)


@app.cell
def _(completed_comparisons, mo, sweep_plan):
    completed_metrics = {comparison["uq_metric"] for comparison in completed_comparisons}
    metric_options = [metric for metric in sweep_plan["uq_metrics"] if metric in completed_metrics]
    mo.stop(
        not metric_options,
        mo.md("There are no completed comparisons to inspect yet."),
    )
    metric_selector = mo.ui.dropdown(
        options=metric_options,
        value=metric_options[0],
        label="UQ metric",
        full_width=True,
    )
    return (metric_selector,)


@app.cell
def _(completed_comparisons, metric_selector, mo):
    metric_comparisons = sorted(
        [
            comparison
            for comparison in completed_comparisons
            if comparison["uq_metric"] == metric_selector.value
        ],
        key=lambda comparison: comparison["config_id"],
    )
    config_options = {
        comparison["config_id"]: index for index, comparison in enumerate(metric_comparisons)
    }
    config_selector = mo.ui.dropdown(
        options=config_options,
        value=next(iter(config_options)),
        label="Sampled configuration",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## Configuration drill-down"),
            mo.hstack([metric_selector, config_selector], widths=[1, 1]),
        ]
    )
    return config_selector, metric_comparisons


@app.cell
def _(config_selector, json, metric_comparisons, mo, pl, sweep_dir):
    selected_comparison = metric_comparisons[config_selector.value]
    selected_run_dir = sweep_dir / selected_comparison["run_dir"]
    selected_run_config = json.loads((selected_run_dir / "config.json").read_text())
    selected_run_id = f"{selected_comparison['config_id']}/{selected_comparison['uq_metric']}"
    selected_results = pl.read_csv(selected_run_dir / "results.csv").with_columns(
        pl.lit(selected_run_id).alias("run_id")
    )
    selected_settings = {
        "config_id": selected_comparison["config_id"],
        "uq_metric": selected_comparison["uq_metric"],
        **selected_comparison["parameters"],
        "model_seeds": json.dumps(selected_run_config["model_seeds"]),
        "max_pool_percent": selected_run_config["max_pool_percent"],
        "mean_test_entity_f1_gap_auc": selected_comparison["mean_test_entity_f1_gap_auc"],
        "mean_final_test_entity_f1_gap": selected_comparison["mean_final_test_entity_f1_gap"],
    }
    mo.ui.table([selected_settings], selection=None, label="Selected comparison")
    return selected_comparison, selected_results, selected_run_dir


@app.cell
def _(
    coverage_percent_slider,
    make_variance_chart,
    mo,
    selected_comparison,
    selected_results,
):
    learning_curve_chart = make_variance_chart(
        selected_results,
        selected_comparison["uq_metric"],
        coverage_percent_slider.value,
    )
    macro_f1_notice = (
        mo.md("")
        if "entity_macro_f1" in selected_results.columns
        else mo.callout(
            mo.md(
                "**Macro entity F1 is unavailable for this comparison.** Its result file "
                "predates that metric; rerun the sweep to populate the third curve."
            ),
            kind="warn",
        )
    )
    mo.vstack(
        [mo.md("### Test learning curves across seeds"), macro_f1_notice, learning_curve_chart]
    )
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
            ### Cumulative NER-tag coverage

            The first chart shows the percentage acquired within each gold-label
            category, so rare tags remain visible. The original all-pool-share chart
            is shown underneath. Bars include zero counts across seeds. Gold labels
            were used only after acquisition.
            """),
            coverage_percent_slider,
        ]
    )
    return (coverage_percent_slider,)


@app.cell
def _(pl, selected_comparison, selected_results, selected_run_dir):
    coverage_run_id = f"{selected_comparison['config_id']}/{selected_comparison['uq_metric']}"
    selected_selections = pl.read_json(selected_run_dir / "selections.json").with_columns(
        pl.lit(coverage_run_id).alias("run_id")
    )
    round_progress = selected_results.select(
        "run_id", "seed", "arm", "round", "percent_acquired", "scoreable_pool_tokens"
    ).unique()
    selections_with_progress = selected_selections.join(
        round_progress,
        on=["run_id", "seed", "arm", "round"],
        how="left",
        validate="m:1",
    )
    return (selections_with_progress,)


@app.cell
def _(
    coverage_percent_slider,
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    selected_comparison,
    selections_with_progress,
):
    mo.vstack(
        [
            mo.ui.altair_chart(
                make_tag_category_coverage_chart(
                    selections_with_progress,
                    coverage_percent_slider.value,
                    selected_comparison["uq_metric"],
                )
            ),
            mo.ui.altair_chart(
                make_tag_coverage_chart(
                    selections_with_progress,
                    coverage_percent_slider.value,
                    selected_comparison["uq_metric"],
                )
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
