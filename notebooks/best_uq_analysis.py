import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Best-config UQ results

    Compare saved runs in `results/best_uq/` without downloading or training models.
    Positive gaps favor uncertainty sampling over random acquisition. Gap AUC is
    normalized over acquired-pool percentage and averaged across seeds; final gap
    compares the last evaluation. These are descriptive test results, not tuning scores.
    Final random and UQ scores are absolute entity F1 scores averaged across seeds.
    Models may use different training settings, so compare UQ metrics within each model.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from utils.charts import make_variance_chart

    return Path, alt, json, make_variance_chart, mo, pl


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh saved results")
    refresh
    return (refresh,)


@app.cell
def _(Path, json, mo, pl, refresh):
    refresh.value
    results_root = Path(__file__).resolve().parents[1] / "results" / "best_uq"
    comparisons = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("evaluation_split") != "test":
            continue
        for comparison in summary["comparisons"]:
            comparison_run_path = (
                summary_path.parent / comparison["run_dir"] if comparison.get("run_dir") else None
            )
            final_scores = {}
            if (
                comparison["status"] == "complete"
                and comparison_run_path is not None
                and (comparison_run_path / "results.csv").is_file()
            ):
                comparison_results = pl.read_csv(comparison_run_path / "results.csv")
                final_scores = dict(
                    comparison_results.filter(pl.col("round") == pl.col("round").max().over("seed"))
                    .group_by("arm")
                    .agg(pl.col("entity_f1").mean())
                    .iter_rows()
                )
            comparisons.append(
                {
                    "model_run": summary_path.parent.name,
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "status": comparison["status"],
                    "final_random_f1": final_scores.get("random"),
                    "final_uq_f1": final_scores.get("uncertainty"),
                    "gap_auc": comparison.get("mean_test_entity_f1_gap_auc"),
                    "final_gap": comparison.get("mean_final_test_entity_f1_gap"),
                    "error": comparison.get("error", ""),
                    "run_path": str(comparison_run_path) if comparison_run_path is not None else "",
                }
            )
    mo.stop(not comparisons, mo.md("No saved test comparisons found in `results/best_uq/`."))
    overview = pl.DataFrame(comparisons)
    completed = overview.filter(pl.col("status") == "complete")
    mo.vstack(
        [
            mo.md(
                f"**{completed.height}/{overview.height} comparisons complete.** "
                "Charts include completed comparisons only."
            ),
            mo.ui.table(overview.drop("run_path"), selection=None, label="All comparisons"),
        ]
    )
    return (completed,)


@app.cell
def _(alt, completed, mo):
    mo.stop(completed.is_empty(), mo.md("No completed comparisons to plot yet."))
    gap_data = completed.unpivot(
        on=["gap_auc", "final_gap"],
        index=["model_run", "config_id", "uq_metric"],
        variable_name="measure",
        value_name="gap",
    )
    gap_points = (
        alt.Chart(gap_data)
        .mark_circle(size=100)
        .encode(
            x=alt.X("gap:Q", title="Entity F1 gap (UQ − random)"),
            y=alt.Y("model_run:N", title=None),
            color=alt.Color("uq_metric:N", title="UQ metric"),
            yOffset="uq_metric:N",
            tooltip=[
                "model_run",
                "config_id",
                "uq_metric",
                "measure",
                alt.Tooltip("gap:Q", format=".4f"),
            ],
        )
    )
    zero_line = (
        alt.Chart(gap_data).mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2).encode(x=alt.datum(0))
    )
    gap_chart = (
        (gap_points + zero_line)
        .properties(width=420, height=300)
        .facet(column=alt.Column("measure:N", title=None))
    )
    mo.ui.altair_chart(gap_chart)
    return


@app.cell
def _(completed, mo):
    mo.stop(completed.is_empty())
    model_options = sorted(completed["model_run"].unique().to_list())
    model_selector = mo.ui.dropdown(model_options, value=model_options[0], label="Model run")
    model_selector
    return (model_selector,)


@app.cell
def _(completed, mo, model_selector, pl):
    model_runs = completed.filter(pl.col("model_run") == model_selector.value).to_dicts()
    run_options = {
        f"{run['uq_metric']} / {run['config_id']}": index for index, run in enumerate(model_runs)
    }
    run_selector = mo.ui.dropdown(run_options, value=next(iter(run_options)), label="UQ comparison")
    run_selector
    return model_runs, run_selector


@app.cell
def _(Path, json, mo, model_runs, pl, run_selector):
    selected_run = model_runs[run_selector.value]
    run_dir = Path(selected_run["run_path"])
    mo.stop(
        not (run_dir / "results.csv").exists() or not (run_dir / "config.json").exists(),
        mo.md("This summary's run files are missing. Copy the run directory to inspect curves."),
    )
    results = pl.read_csv(run_dir / "results.csv")
    settings = json.loads((run_dir / "config.json").read_text())
    mo.accordion({"Saved experiment settings": mo.json(settings)})
    return results, selected_run


@app.cell
def _(make_variance_chart, mo, results, selected_run):
    _chart = make_variance_chart(results, selected_run["uq_metric"])
    chart = mo.ui.altair_chart(_chart)
    mo.vstack(
        [
            mo.md(
                "## Test learning curves\nLines show seed means; bands show ±1 standard "
                "deviation, not confidence intervals."
            ),
            chart,
        ]
    )
    return


@app.cell
def _(mo, results):
    final_rows = results.sort("round").group_by("seed", "arm", maintain_order=True).last()
    mo.vstack(
        [
            mo.md("## Final evaluation per seed"),
            mo.ui.table(
                final_rows.select(
                    "seed",
                    "arm",
                    "round",
                    "percent_acquired",
                    "n_acquired",
                    *[
                        field
                        for field in ["entity_f1", "entity_macro_f1", "token_accuracy"]
                        if field in results.columns
                    ],
                ).sort("seed", "arm"),
                selection=None,
            ),
            mo.accordion({"All evaluation rows": mo.ui.table(results, selection=None)}),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
