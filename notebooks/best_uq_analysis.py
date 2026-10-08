import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Best-config UQ results

    Saved test runs from `results/best_uq/`. Positive gaps favour uncertainty sampling
    over random acquisition. Compare UQ metrics within each model.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Gap AUC** is the UQ-minus-random entity-F1 AUC, normalized over acquired-pool
      percentage and averaged across seeds. **Final gap** compares the last evaluation.
    - Final random and UQ scores are absolute entity F1, averaged across seeds.
    - These are descriptive test results, not tuning scores. Models may use different
      training settings, so compare UQ metrics within a model, not across models.
    - Nothing is downloaded or trained.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from wigglystuff import FloatingPanel

    from utils.acquisition_analysis import (
        acquisition_diagnostics,
        enrichment_at_budget,
        pet_sentences,
        tag_selections,
        threshold_efficiency,
        timing_by_group,
        token_timing,
    )
    from utils.charts import (
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
    )
    from utils.comparison_summaries import load_test_comparisons, uq_gap_ttests
    from utils.oracle_analysis import (
        find_oracle_comparison,
        search_winner,
        validation_against_oracle,
    )

    repo_root = Path(__file__).resolve().parents[1]
    return (
        FloatingPanel,
        Path,
        acquisition_diagnostics,
        alt,
        enrichment_at_budget,
        find_oracle_comparison,
        json,
        load_test_comparisons,
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        make_variance_chart,
        mo,
        pet_sentences,
        pl,
        repo_root,
        search_winner,
        tag_selections,
        threshold_efficiency,
        timing_by_group,
        token_timing,
        uq_gap_ttests,
        validation_against_oracle,
    )


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh saved results")
    refresh
    return (refresh,)


@app.cell
def _(load_test_comparisons, mo, pl, refresh, repo_root):
    refresh.value
    overview = load_test_comparisons(repo_root / "results" / "best_uq")
    mo.stop(overview.is_empty(), mo.md("No saved test comparisons found in `results/best_uq/`."))
    completed = overview.filter(pl.col("status") == "complete")
    return completed, overview


@app.cell
def _(alt, completed, mo, overview):
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
    mo.vstack(
        [
            mo.md(
                f"## Overview\n**{completed.height}/{overview.height} comparisons complete.** "
                "The chart includes completed comparisons only."
            ),
            mo.ui.altair_chart(gap_chart),
            mo.accordion(
                {
                    "All comparisons": mo.ui.table(
                        overview.drop("run_path"), selection=None, label="All comparisons"
                    )
                }
            ),
        ]
    )
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
    return results, selected_run, settings


@app.cell
def _(repo_root, settings):
    from uq_pet.supervised import load_test_results

    supervised_saved = load_test_results(
        repo_root / "results" / "supervised",
        settings["checkpoint"],
        settings.get("dataset", "pet"),
        settings.get("dataset_percent", 100.0),
    )
    supervised_sweep, supervised_results = supervised_saved or (None, None)
    return supervised_results, supervised_sweep


@app.cell
def _(
    make_variance_chart,
    mo,
    results,
    selected_run,
    settings,
    supervised_results,
    supervised_sweep,
    tag_coverage_percent,
):
    final_rows = results.sort("round").group_by("seed", "arm", maintain_order=True).last()
    mo.vstack(
        [
            mo.md(
                "## Test learning curves\nLines are seed means with ±1 SD bands. "
                + (
                    f"The green dashed line is the fully supervised upper bound "
                    f"(`results/supervised/{supervised_sweep}/`)."
                    if supervised_sweep is not None
                    else "No supervised baseline is saved for this checkpoint."
                )
            ),
            mo.ui.altair_chart(
                make_variance_chart(
                    results,
                    selected_run["uq_metric"],
                    tag_coverage_percent.value,
                    supervised=supervised_results,
                )
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Bands show ±1 standard deviation across seeds, not confidence intervals.
    - The red dashed vertical line marks the pool percentage set by the floating slider.
    - The supervised baseline trains fresh models for a fixed number of epochs on every
      labelled seed and pool sentence, with hyperparameters from a separate validation
      grid search. It is an upper bound, not a matched-budget arm. Run
      `scripts/run_supervised.py` to add it for a checkpoint that lacks it.
    """),
                    "Final evaluation per seed": mo.ui.table(
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
                    "All evaluation rows": mo.ui.table(results, selection=None),
                    "Saved experiment settings": mo.json(settings),
                }
            ),
        ]
    )
    return


@app.cell
def _(
    find_oracle_comparison,
    json,
    make_variance_chart,
    mo,
    pl,
    repo_root,
    selected_run,
    settings,
    supervised_results,
    tag_coverage_percent,
):
    oracle_comparison = find_oracle_comparison(
        repo_root / "results" / "oracle_random" / "uq",
        settings["checkpoint"],
        settings.get("dataset", "pet"),
        selected_run["uq_metric"],
    )
    mo.stop(
        oracle_comparison is None,
        mo.md(
            "No completed test-tuned random oracle comparison for this checkpoint and UQ "
            "metric in `results/oracle_random/uq/`."
        ),
    )
    oracle_results = pl.read_csv(oracle_comparison["run_dir"] / "results.csv")
    mo.vstack(
        [
            mo.md(
                "## Appendix: test curves at the test-tuned random oracle\n"
                "The same model and UQ metric, rerun at random's best test configuration "
                "(ADR 0007). Tuning on the test split favours random, so this is an appendix "
                f"check. Gap AUC: **{100 * selected_run['gap_auc']:.2f} pp** at the validation "
                f"winner, **{100 * oracle_comparison['mean_test_entity_f1_gap_auc']:.2f} pp** "
                "at the oracle."
            ),
            mo.ui.altair_chart(
                make_variance_chart(
                    oracle_results,
                    selected_run["uq_metric"],
                    tag_coverage_percent.value,
                    supervised=supervised_results,
                )
            ),
            mo.accordion(
                {
                    f"Oracle experiment settings ({oracle_comparison['sweep']})": mo.json(
                        json.loads((oracle_comparison["run_dir"] / "config.json").read_text())
                    )
                }
            ),
        ]
    )
    return oracle_comparison, oracle_results


@app.cell
def _(
    mo,
    oracle_comparison,
    oracle_results,
    repo_root,
    results,
    search_winner,
    settings,
    validation_against_oracle,
):
    _dataset = settings.get("dataset", "pet")
    tuned = validation_against_oracle(
        (
            results,
            settings,
            search_winner(
                repo_root / "results" / "random_baseline_search", settings["checkpoint"], _dataset
            ),
        ),
        (
            oracle_results,
            oracle_comparison["parameters"],
            search_winner(
                repo_root / "results" / "oracle_random", settings["checkpoint"], _dataset
            ),
        ),
    )
    validation_row, oracle_row = tuned.to_dicts()
    same_winner = all(
        validation_row[name] == oracle_row[name] for name in oracle_comparison["parameters"]
    )
    mo.vstack(
        [
            mo.md(
                "### Random at the validation winner against the oracle\n"
                "Tuning random on the test split changes its AUC by "
                f"**{oracle_row['random_auc_pp'] - validation_row['random_auc_pp']:+.2f} pp** "
                f"and the gap by **{oracle_row['gap_auc_pp'] - validation_row['gap_auc_pp']:+.2f} "
                "pp**."
                + (
                    "\n\n**The validation winner is also the test-tuned oracle** for this "
                    "checkpoint, so both rows rerun the same hyperparameters with the same seeds."
                    if same_winner
                    else ""
                )
            ),
            mo.ui.table(
                tuned,
                selection=None,
                label="Validation winner against the test-tuned random oracle",
                format_mapping={
                    column: "{:.2f}" for column in tuned.columns if column.endswith("_pp")
                }
                | {"learning_rate": "{:.3g}"},
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - AUC is normalized test entity-F1 AUC over acquired-pool percentage, averaged across
      model seeds. UQ runs at each row's configuration with the selected UQ metric.
    - The oracle's random gain includes the winner's curse of picking the best of the
      sampled configurations on the test split.
    - `search_config_id` is the configuration's ID in the shared 100-configuration
      random search; both winners come from the same sampled configurations.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    ## Why does UQ help? Saved-selection diagnostics

    What UQ prioritizes and when, read from the saved selections of the comparison
    selected above. Descriptive, not causal.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Means weight seeds equally; seeds share the same data split.
    - UQ metrics share one random baseline, so they are not independent replications.
    - No models are loaded or trained.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(Path, acquisition_diagnostics, json, mo, results, selected_run):
    selection_path = Path(selected_run["run_path"]) / "selections.json"
    mo.stop(not selection_path.is_file(), mo.md("Selections are missing for this run."))
    saved_selections = json.loads(selection_path.read_text())
    diagnostics = acquisition_diagnostics(saved_selections, results)
    full_pool = (
        diagnostics["validation"]["full_pool"].all()
        and diagnostics["validation"]["same_final_set"].all()
    )
    mo.accordion(
        {
            "Log validation and final-set coverage": mo.ui.table(
                diagnostics["validation"], selection=None
            )
        }
    )
    return diagnostics, full_pool, saved_selections


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
    mo.vstack(
        [
            mo.md("""
    ### Cumulative NER-tag coverage

    How much of each gold tag each arm has acquired by the pool percentage set in the
    floating slider. The slider also moves the red markers on every curve above and below.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - The first chart shows the fraction of **each gold tag's available tokens**
      acquired, keeping rare tags visible. The second shows acquired tokens of each tag as
      a fraction of **all scoreable pool tokens**. Both use full BIO tags.
    - Bars are seed means and include zero counts.
    - Only completed rounds at or below the chosen percentage count, without
      interpolation.
    - Tag totals come from the completed acquisition logs; this retrospective analysis
      does not use gold labels to select tokens.
    """)
                }
            ),
        ]
    )
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
    full_pool,
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    results,
    saved_selections,
    selected_run,
    tag_coverage_percent,
    tag_selections,
):
    mo.stop(
        not full_pool,
        mo.md(
            "Full-pool tag coverage requires both arms to have acquired the same complete pool. "
            "The remaining diagnostics still work with partial logs."
        ),
    )
    tag_frame, tag_label_order = tag_selections(saved_selections, results, selected_run["run_path"])
    mo.vstack(
        [
            mo.ui.altair_chart(
                make_tag_category_coverage_chart(
                    tag_frame,
                    tag_coverage_percent.value,
                    selected_run["uq_metric"],
                    label_order=tag_label_order,
                )
            ),
            mo.ui.altair_chart(
                make_tag_coverage_chart(
                    tag_frame,
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
    enrichment_at_budget,
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
    mo.vstack(
        [
            mo.md(
                "### Label composition over budget\n"
                "Positive enrichment means UQ spends more of its budget on that label."
            ),
            mo.ui.altair_chart(composition_chart),
            mo.ui.table(
                enrichment_at_budget(composition_data, selected_token_budget), selection=None
            ),
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
                "of selections beyond the first occurrence of each word–BIO-label pair."
            ),
            mo.ui.altair_chart(coverage_chart),
            mo.accordion(
                {
                    "Means at the selected budget": mo.ui.table(
                        coverage_means.filter(pl.col("n_acquired") == selected_token_budget),
                        selection=None,
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    f1_target = mo.ui.slider(start=0.1, stop=0.95, step=0.01, value=0.7, label="Entity F1 target")
    mo.vstack(
        [
            mo.md(
                "### Label efficiency\nLabels each arm needs to first reach the target, per "
                "seed. Positive `labels_saved` favours UQ."
            ),
            f1_target,
        ]
    )
    return (f1_target,)


@app.cell
def _(f1_target, mo, results, threshold_efficiency):
    mo.vstack(
        [
            mo.ui.table(threshold_efficiency(results, f1_target.value), selection=None),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - The first **observed** crossing, with no interpolation; a later score can fall
      below the target.
    - Null means not reached within the saved budget.
    - Round 0 can reach the target with zero additional labels.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(alt, diagnostics, full_pool, grouping_selector, mo, timing_by_group):
    timing_data = diagnostics["timing"]
    mo.stop(
        timing_data.is_empty(),
        mo.md("No tokens were acquired by both arms; timing cannot be paired."),
    )
    timing_group = {"Entity type": "entity_type", "BIO": "BIO", "Entity / O": "entity_status"}[
        grouping_selector.value
    ]
    timing_seed_means, timing_summary = timing_by_group(timing_data, timing_group)
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
    mo.vstack(
        [
            mo.md(
                "### Which labels does UQ acquire earlier?\n"
                "Positive values mean UQ acquired that label's tokens earlier, in percentage "
                "points of the pool. Each point is one seed."
                + (
                    ""
                    if full_pool
                    else "\n\n**Partial coverage:** only tokens acquired by both arms are "
                    "included. The policies select this subset, so it does not describe the "
                    "full pool. See log validation."
                )
            ),
            mo.ui.altair_chart(
                timing_chart
                + alt.Chart(timing_seed_means)
                .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2)
                .encode(x=alt.datum(0))
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Each round's tokens are tied at the midpoint of its acquisition positions, divided
      by the scoreable pool size. The value is random's position minus UQ's.
    - Points average tokens within each seed; the summary then weights seeds equally.
    - +20 means 20 percentage points of the pool earlier, not 20% fewer labels.
    """),
                    "Summary by label group": mo.ui.table(timing_summary, selection=None),
                }
            ),
        ]
    )
    return (timing_data,)


@app.cell
def _(mo, pet_sentences, repo_root, timing_data, token_timing):
    mo.vstack(
        [
            mo.md(
                "### Inspect tokens brought forward or delayed\n"
                "Sorted by mean timing advantage; sort ascending to see delayed tokens."
            ),
            mo.ui.table(
                token_timing(
                    timing_data,
                    pet_sentences(repo_root / "data" / "raw" / "PETv1.1-entities.jsonl"),
                ),
                selection=None,
                page_size=15,
                wrapped_columns=["sentence_context"],
                column_widths={"sentence_context": 420},
                format_mapping={
                    "mean_uncertainty_score": "{:.3g}",
                    "mean_advance_pp": "{:.3f}",
                    "min_advance_pp": "{:.3f}",
                    "sentence_context": lambda value: mo.md(value).style(
                        {"white-space": "normal", "overflow-wrap": "anywhere"}
                    ),
                },
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - `seeds_earlier` and `paired_seeds` separate consistency from a large mean.
    - `mean_uncertainty_score` averages the uncertainty arm's score at acquisition across
      paired seeds (higher means more uncertain; missing scores are excluded).
    - Context is read only from the local PET file; nothing is downloaded.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, repo_root, uq_gap_ttests):
    mo.vstack(
        [
            mo.md(
                "## UQ against random across 30 random-search configurations\n"
                "One-sample t-tests of each model's gap AUC across the configurations in "
                "`results/random_search/`. Holm and Bonferroni adjust across the three UQ "
                "metrics within each model."
            ),
            mo.ui.table(
                uq_gap_ttests(repo_root / "results" / "random_search"),
                label="UQ versus random — normalized learning-curve AUC",
                selection=None,
                page_size=20,
                format_mapping={
                    "Mean improvement (pp)": "{:.2f}",
                    "95% CI lower (pp)": "{:.2f}",
                    "95% CI upper (pp)": "{:.2f}",
                    "p-value": "{:.3g}",
                    "Adjusted p-value (Holm)": "{:.3g}",
                    "Adjusted p-value (Bonferroni)": "{:.3g}",
                },
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
