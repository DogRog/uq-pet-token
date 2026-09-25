import marimo

__generated_with = "0.25.0"
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
    from wigglystuff import FloatingPanel
    from scipy import stats

    from utils.charts import (
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
    )

    return (
        FloatingPanel,
        Path,
        alt,
        json,
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
        mo,
        pl,
        stats,
    )


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
        alt.Chart(gap_data)
        .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2)
        .encode(x=alt.datum(0))
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
def _(make_variance_chart, mo, results, selected_run, tag_coverage_percent):
    _chart = make_variance_chart(results, selected_run["uq_metric"], tag_coverage_percent.value)
    chart = mo.ui.altair_chart(_chart)
    mo.vstack(
        [
            mo.md(
                "## Test learning curves\nLines show seed means; bands show ±1 standard "
                "deviation, not confidence intervals. The dashed vertical line marks the "
                "pool percentage selected in the cumulative NER-tag coverage slider below."
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Why does UQ help? Saved-selection diagnostics

    These analyses describe **what UQ prioritizes and when**, not causal effects.
    Each model/metric comparison uses the selector above. Means weight seeds equally;
    seeds share the same data split. Metrics share a random baseline, so they are not
    independent replications. No models are loaded or trained.
    """)
    return


@app.cell
def _():
    from utils.acquisition_analysis import (
        acquisition_diagnostics,
        threshold_efficiency,
    )

    return acquisition_diagnostics, threshold_efficiency


@app.cell
def _(Path, acquisition_diagnostics, json, mo, results, selected_run):
    selection_path = Path(selected_run["run_path"]) / "selections.json"
    mo.stop(not selection_path.is_file(), mo.md("Selections are missing for this run."))
    saved_selections = json.loads(selection_path.read_text())
    diagnostics = acquisition_diagnostics(saved_selections, results)
    mo.accordion(
        {
            "Log validation and final-set coverage": mo.ui.table(
                diagnostics["validation"], selection=None
            )
        }
    )
    return diagnostics, saved_selections


@app.cell
def _(mo):
    tag_coverage_percent = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=25,
        label="Scoreable pool acquired (%)",
        show_value=True,
        full_width=True,
    )
    mo.md("""
        ### Cumulative NER-tag coverage

        The first chart shows the fraction of **each gold tag's available tokens**
        acquired, keeping rare tags visible. The second shows acquired tokens of
        each tag as a fraction of **all scoreable pool tokens**. Bars are seed means
        and include zero counts. Full BIO tags are shown in both charts.

        This slider controls the tag charts, diagnostic tables, and red dashed
        vertical markers on all learning, composition, and coverage curves. Only completed acquisition rounds
        at or below the chosen percentage are included, without interpolation.
        Tag totals come from the completed acquisition logs; this retrospective
        analysis does not use gold labels to select tokens.
        """)
    return (tag_coverage_percent,)


@app.cell
def _(FloatingPanel, mo, selected_token_budget, tag_coverage_percent):
    FloatingPanel(
        mo.vstack(
            [
                tag_coverage_percent,
                mo.md(f"**Acquired tokens per arm: {selected_token_budget:,}**"),
                mo.md("Latest completed round at or below the selected percentage."),
            ]
        )
    )
    return


@app.cell
def _(
    diagnostics,
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    pl,
    results,
    saved_selections,
    selected_run,
    tag_coverage_percent,
):
    mo.stop(
        not (
            diagnostics["validation"]["full_pool"].all()
            and diagnostics["validation"]["same_final_set"].all()
        ),
        mo.md(
            "Full-pool tag coverage requires both arms to have acquired the same complete pool. "
            "The remaining diagnostics still work with partial logs."
        ),
    )
    tag_selections_with_progress = (
        pl.DataFrame(saved_selections)
        .with_columns(pl.lit(selected_run["run_path"]).alias("run_id"))
        .join(
            results.select("seed", "arm", "round", "percent_acquired", "scoreable_pool_tokens"),
            on=["seed", "arm", "round"],
            how="left",
            validate="m:1",
        )
    )
    tag_label_order = (
        tag_selections_with_progress.select("pool_idx", "word_idx", "label")
        .unique()
        .group_by("label")
        .len()
        .sort(["len", "label"], descending=[True, False])["label"]
        .to_list()
    )
    mo.vstack(
        [
            mo.ui.altair_chart(
                make_tag_category_coverage_chart(
                    tag_selections_with_progress,
                    tag_coverage_percent.value,
                    selected_run["uq_metric"],
                    label_order=tag_label_order,
                )
            ),
            mo.ui.altair_chart(
                make_tag_coverage_chart(
                    tag_selections_with_progress,
                    tag_coverage_percent.value,
                    selected_run["uq_metric"],
                    label_order=tag_label_order,
                )
            ),
        ]
    )
    return


@app.cell
def _(mo, pl, results, tag_coverage_percent):
    selected_token_cutoff = (
        results["scoreable_pool_tokens"].max() * tag_coverage_percent.value / 100
    )
    selected_token_budget = (
        results.filter(pl.col("percent_acquired") <= tag_coverage_percent.value)["n_acquired"].max()
        or 0
    )
    grouping_selector = mo.ui.dropdown(
        ["Entity type", "BIO", "Entity / O"], value="Entity type", label="Label grouping"
    )
    grouping_selector
    return grouping_selector, selected_token_budget, selected_token_cutoff


@app.cell
def _(
    alt,
    diagnostics,
    grouping_selector,
    mo,
    pl,
    selected_token_budget,
    selected_token_cutoff,
):
    composition_data = diagnostics["composition"].filter(
        pl.col("grouping") == grouping_selector.value
    )
    composition_means = composition_data.group_by("n_acquired", "arm", "category").agg(
        pl.col("share_percent").mean()
    )
    composition_lines = (
        alt.Chart(composition_means)
        .mark_line()
        .encode(
            x=alt.X("n_acquired:Q", title="Acquired tokens per arm"),
            y=alt.Y("share_percent:Q", title="Cumulative share of selected tokens (%)"),
            color="arm:N",
            tooltip=["arm", "category", "n_acquired", alt.Tooltip("share_percent:Q", format=".2f")],
        )
    )
    composition_marker = (
        alt.Chart(composition_means)
        .transform_aggregate(groupby=["category"])
        .mark_rule(color="#E45756", strokeDash=[7, 5], strokeWidth=2)
        .encode(x=alt.datum(selected_token_cutoff))
    )
    composition_points = composition_lines.transform_filter(
        alt.datum.n_acquired == selected_token_budget
    ).mark_circle(size=65)
    composition_chart = (
        (composition_lines + composition_marker + composition_points)
        .properties(width=240, height=170)
        .facet(facet="category:N", columns=3)
        .resolve_scale(y="independent")
    )
    composition_at_budget = composition_data.filter(pl.col("n_acquired") == selected_token_budget)
    composition_pairs = (
        composition_at_budget.filter(pl.col("arm") == "uncertainty")
        .join(
            composition_at_budget.filter(pl.col("arm") == "random"),
            on=["seed", "category"],
            suffix="_random",
            validate="1:1",
        )
        .with_columns(
            (pl.col("share_percent") - pl.col("share_percent_random")).alias("enrichment_pp")
        )
    )
    enrichment_table = (
        composition_pairs.group_by("n_acquired", "category")
        .agg(
            pl.col("share_percent_random").mean().alias("random_share_percent"),
            pl.col("share_percent").mean().alias("uq_share_percent"),
            pl.col("enrichment_pp").mean(),
            (pl.col("enrichment_pp") > 0).sum().alias("seeds_enriched"),
            pl.len().alias("seeds"),
        )
        .sort("enrichment_pp", descending=True)
    )
    mo.vstack(
        [
            mo.md(
                "### Label composition over budget \n"
                "Positive enrichment means UQ spends more of its budget on that label."
            ),
            mo.ui.altair_chart(composition_chart),
            mo.ui.table(enrichment_table, selection=None),
        ]
    )
    return


@app.cell
def _(alt, diagnostics, mo, pl, selected_token_budget, selected_token_cutoff):
    coverage_fields = [
        "distinct_words",
        "distinct_sentences",
        "distinct_documents",
        "distinct_word_label_pairs",
        "repeated_word_label_percent",
    ]
    coverage_means = (
        diagnostics["coverage"]
        .group_by("n_acquired", "arm")
        .agg(*[pl.col(field).mean() for field in coverage_fields])
    )
    coverage_lines = (
        alt.Chart(
            coverage_means.unpivot(
                index=["n_acquired", "arm"],
                on=coverage_fields,
                variable_name="measure",
                value_name="value",
            )
        )
        .mark_line()
        .encode(
            x=alt.X("n_acquired:Q", title="Acquired tokens per arm"),
            y="value:Q",
            color="arm:N",
            tooltip=["arm", "measure", "n_acquired", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    coverage_marker = (
        alt.Chart(coverage_lines.data)
        .transform_aggregate(groupby=["measure"])
        .mark_rule(color="#E45756", strokeDash=[7, 5], strokeWidth=2)
        .encode(x=alt.datum(selected_token_cutoff))
    )
    coverage_chart = (
        (coverage_lines + coverage_marker)
        .properties(width=240, height=170)
        .facet(facet="measure:N", columns=3)
        .resolve_scale(y="independent")
    )
    mo.vstack(
        [
            mo.md(
                "### Coverage and repetition\nWords are lowercased. Repetition is the percentage "
                "of selections beyond the first occurrence of each word–BIO-label pair. "
            ),
            mo.ui.altair_chart(coverage_chart),
            mo.ui.table(
                coverage_means.filter(pl.col("n_acquired") == selected_token_budget), selection=None
            ),
        ]
    )
    return


@app.cell
def _(mo):
    f1_target = mo.ui.slider(start=0.1, stop=0.95, step=0.01, value=0.7, label="Entity F1 target")
    mo.vstack([mo.md("### Label efficiency"), f1_target])
    return (f1_target,)


@app.cell
def _(f1_target, mo, results, threshold_efficiency):
    efficiency_table = threshold_efficiency(results, f1_target.value)
    mo.vstack(
        [
            mo.md(
                "First **observed** target crossing, with no interpolation; a later score can fall "
                "below the target. Null means not reached within the saved budget. Positive labels "
                "saved favors UQ. Round 0 can reach the target with zero additional labels."
            ),
            mo.ui.table(efficiency_table, selection=None),
        ]
    )
    return


@app.cell
def _(alt, diagnostics, grouping_selector, mo, pl):
    timing_data = diagnostics["timing"]
    mo.stop(
        timing_data.is_empty(),
        mo.md("No tokens were acquired by both arms; timing cannot be paired."),
    )
    timing_group = {"Entity type": "entity_type", "BIO": "BIO", "Entity / O": "entity_status"}[
        grouping_selector.value
    ]
    timing_with_group = timing_data.with_columns(
        pl.when(pl.col("label") == "O")
        .then(pl.lit("O"))
        .otherwise(pl.lit("Entity"))
        .alias("entity_status")
    )
    timing_seed_means = timing_with_group.group_by("seed", timing_group).agg(
        pl.col("advance_pp").mean(), pl.len().alias("matched_tokens")
    )
    timing_summary = (
        timing_seed_means.group_by(timing_group)
        .agg(
            pl.col("advance_pp").mean().alias("mean_advance_pp"),
            pl.col("advance_pp").min().alias("min_seed_advance_pp"),
            pl.col("advance_pp").max().alias("max_seed_advance_pp"),
            (pl.col("advance_pp") > 0).sum().alias("seeds_earlier"),
            pl.len().alias("seeds"),
        )
        .sort("mean_advance_pp", descending=True)
    )
    timing_chart = (
        alt.Chart(timing_seed_means)
        .mark_circle(size=80)
        .encode(
            x=alt.X("advance_pp:Q", title="Acquired earlier by UQ (pool percentage points)"),
            y=alt.Y(f"{timing_group}:N", title=None),
            color="seed:N",
            tooltip=[
                "seed",
                timing_group,
                "matched_tokens",
                alt.Tooltip("advance_pp:Q", format=".2f"),
            ],
        )
        .properties(width=650, height=240)
    )
    full_timing = (
        diagnostics["validation"]["full_pool"].all()
        and diagnostics["validation"]["same_final_set"].all()
    )
    mo.vstack(
        [
            mo.md(
                "### Which labels does UQ acquire earlier?\n"
                "Each round is tied at the midpoint of its acquisition positions, divided by the "
                "scoreable pool size. Positive random-minus-UQ position means UQ acquired the token "
                "earlier. Points summarize tokens within each seed; seeds are then weighted equally. "
                "A +20 value means 20 percentage points of the pool earlier, not 20% fewer labels.\n\n"
                + (
                    "Both arms acquired the same complete pool: timing covers every candidate."
                    if full_timing
                    else "**Partial coverage:** only tokens acquired by both arms are included. "
                    "This subset is selected by the policies; it does not describe the full pool. See log validation."
                )
            ),
            mo.ui.altair_chart(
                timing_chart
                + alt.Chart(timing_seed_means)
                .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2)
                .encode(x=alt.datum(0))
            ),
            mo.ui.table(timing_summary, selection=None),
        ]
    )
    return (timing_data,)


@app.cell
def _(Path, json, mo, pl, timing_data):
    token_timing = (
        timing_data.group_by(
            "pool_idx", "word_idx", "document_name", "sentence_id", "token", "label"
        )
        .agg(
            pl.col("advance_pp").mean().alias("mean_advance_pp"),
            pl.col("advance_pp").min().alias("min_advance_pp"),
            (pl.col("advance_pp") > 0).sum().alias("seeds_earlier"),
            pl.len().alias("paired_seeds"),
        )
        .sort("mean_advance_pp", descending=True)
    )
    context_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "PETv1.1-entities.jsonl"
    context_lookup = {}
    if context_path.is_file():
        for context_line in context_path.read_text().splitlines():
            context_record = json.loads(context_line)
            context_lookup[context_record["document name"], context_record["sentence-ID"]] = (
                context_record["tokens"]
            )
    example_rows = []
    for example in token_timing.to_dicts():
        sentence_tokens = context_lookup.get((example["document_name"], example["sentence_id"]))
        matches_context = (
            sentence_tokens is not None
            and example["word_idx"] < len(sentence_tokens)
            and sentence_tokens[example["word_idx"]] == example["token"]
        )
        example_rows.append(
            {
                **example,
                "sentence_context": " ".join(
                    f"**{word}**" if index == example["word_idx"] else word
                    for index, word in enumerate(sentence_tokens)
                )
                if matches_context
                else "Local sentence context unavailable or token mismatch",
            }
        )
    mo.vstack(
        [
            mo.md(
                "### Inspect tokens brought forward or delayed\n"
                "Sorted by mean timing advantage. Sort ascending to inspect delayed tokens. "
                "`seeds_earlier` and `paired_seeds` distinguish consistency from a large mean. "
                "Context is read only from the local PET file; no data is downloaded."
            ),
            mo.ui.table(
                pl.DataFrame(example_rows),
                selection=None,
                page_size=15,
                wrapped_columns=["sentence_context"],
                column_widths={"sentence_context": 420},
                format_mapping={
                    "sentence_context": lambda value: mo.md(value).style(
                        {"white-space": "normal", "overflow-wrap": "anywhere"}
                    )
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T-test based on 30 configurations for each model (results/random_search)
    """)
    return


@app.cell
def _(Path, json, mo, stats):

    def build_ttest_report(root):
        rows = []

        for path in sorted(root.glob("*/summary.json")):
            summary = json.loads(path.read_text())
            model = path.parent.name.removesuffix("-random-5-seeds")

            for metric in ("entropy", "least_confidence", "margin"):
                gaps = [
                    row["mean_test_entity_f1_gap_auc"]
                    for row in summary["comparisons"]
                    if row["uq_metric"] == metric
                    and row["status"] == "complete"
                ]

                if len(gaps) < 2:
                    continue

                test = stats.ttest_1samp(gaps, 0, alternative="two-sided")
                ci = test.confidence_interval(confidence_level=0.95)

                rows.append({
                    "Model": model,
                    "UQ metric": metric,
                    "Configurations": len(gaps),
                    "Mean improvement (pp)": 100 * sum(gaps) / len(gaps),
                    "95% CI lower (pp)": 100 * ci.low,
                    "95% CI upper (pp)": 100 * ci.high,
                    "p-value": float(test.pvalue),
                })

        return rows


    ttest_report = build_ttest_report(Path("results/random_search"))

    mo.ui.table(
        ttest_report,
        label="UQ versus random — normalized learning-curve AUC",
        selection=None,
        page_size=20,
        format_mapping={
            "Mean improvement (pp)": "{:.2f}",
            "95% CI lower (pp)": "{:.2f}",
            "95% CI upper (pp)": "{:.2f}",
            "p-value": "{:.3g}",
        },
    )
    return


if __name__ == "__main__":
    app.run()
