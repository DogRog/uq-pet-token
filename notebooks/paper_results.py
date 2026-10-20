import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Paper rerun results

    The full rerun from `scripts/run_all_experiments.sh`: five models on PET, Quishpi,
    and MedicalProcessInstruks, read from `results/`, and the LLM baseline from
    `llm-annotation-eval/res/pred/`. Each chapter answers one question
    and is listed in the outline. Positive gaps favour the uncertainty arm.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Gap AUC** is the UQ-minus-random test entity-F1 area under the learning curve,
      normalized over the acquired pool percentage (0–100%) and averaged across model
      seeds. It is in F1 points: 3.0 pp means UQ was 3 F1 points ahead on average over
      the whole acquisition run.
    - **Final gap** compares the last round, after both arms have acquired the whole pool,
      so it measures only the order in which the same words were learned.
    - **Best UQ** runs each model at its validation winner: the random arm's best
      validation-holdout AUC among 100 sampled configurations (ADR 0007).
    - **Top-K** takes the K most uncertain candidates per round; **Gumbel noise** samples
      them in proportion to their scores (ADR 0009).
    - The floating panel switches every chapter between exact match (headline, tuning
      objective) and partial match (reported only, ADR 0010). It also picks the
      acquisition shown in chapters 3–6, the UQ metric of chapters 3 and 5, and the gap
      of chapters 2, 8, and 9. Chapter 10 marks unmatched entities by the chosen matching.
    - **F1 average** switches every chapter between micro F1 (pooled over all entities,
      the headline) and macro F1 (the mean of per-type F1, so a rare type weighs as much
      as a common one; reported only). Precision and recall exist for micro F1 only.
    - **Pool to reach a target** (chapters 4, 5, and 6) is the first observed round at
      which a model's seed-mean curve reaches a fixed target, not interpolated (ADR 0011).
      **Labeling efficiency** is random's pool divided by UQ's, so above 1 favours UQ;
      across models it is a geometric mean, and a lower bound (≥) where random never
      reaches the target. Each chapter's **Robustness** table reads single runs instead.
    - Nothing is downloaded or trained.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import html
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from scipy import stats
    from wigglystuff import FloatingPanel, ParallelCoordinates

    from analysis import paper_results as pr
    from analysis.acquisition_analysis import tag_selections
    from analysis.charts import make_tag_category_coverage_chart, make_tag_coverage_chart

    alt.data_transformers.disable_max_rows()
    results_root = Path(__file__).resolve().parents[1] / "results"
    llm_root = results_root.parent / "llm-annotation-eval" / "res" / "pred"
    series_colors = {
        "entropy": "#F58518",
        "least_confidence": "#E45756",
        "margin": "#B279A2",
        "random": "#4C78A8",
        "supervised": "#54A24B",
    }

    def series_color(series):
        """Color the given series consistently across chapters, listing only those."""
        names = [name for name in series_colors if name in set(series)]
        return alt.Color(
            "series:N",
            title=None,
            sort=names,
            scale=alt.Scale(domain=names, range=[series_colors[name] for name in names]),
            legend=alt.Legend(orient="top"),
        )

    return (
        FloatingPanel,
        ParallelCoordinates,
        Path,
        alt,
        html,
        json,
        llm_root,
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        mo,
        pl,
        pr,
        results_root,
        series_color,
        series_colors,
        stats,
        tag_selections,
    )


@app.cell
def _(FloatingPanel, mo, pr):
    refresh = mo.ui.run_button(label="Reload results")
    entity_matching = mo.ui.dropdown(
        {"exact": "entity_f1", "partial": "entity_partial_f1"},
        value="exact",
        label="Entity matching",
    )
    f1_average = mo.ui.dropdown({"micro": "", "macro": "_macro"}, value="micro", label="F1 average")
    acquisition = mo.ui.dropdown(pr.ACQUISITIONS, value="Top-K", label="Acquisition")
    uq_metric = mo.ui.dropdown(pr.UQ_METRICS, value="entropy", label="UQ metric")
    gap_names = {"gap_auc": "gap AUC", "final_gap": "final F1 gap"}
    gap = mo.ui.dropdown(
        {"Gap AUC": "gap_auc", "Final F1 gap": "final_gap"},
        value="Gap AUC",
        label="Gap",
    )
    pool_percent = mo.ui.slider(
        start=0, stop=100, step=1, value=25, label="Pool acquired, %", show_value=True
    )
    FloatingPanel(
        mo.vstack([entity_matching, f1_average, acquisition, uq_metric, gap, pool_percent, refresh])
    )
    return (
        acquisition,
        entity_matching,
        f1_average,
        gap,
        gap_names,
        pool_percent,
        refresh,
        uq_metric,
    )


@app.cell
def _(llm_root, pl, pr, refresh, results_root):
    refresh.value
    status = pr.sweep_status(results_root)
    curves = pr.comparison_curves(results_root)
    sweep_curves = pr.comparison_curves(results_root, "random_search")
    supervised = pr.supervised_scores(results_root)
    sizes = pr.dataset_sizes(curves)
    winners = pr.tuning_winners(results_root)
    llm = pr.llm_test_scores(llm_root)
    best_runs = (
        curves.select(*pr.SETTING, pl.col("run_dir").cast(pl.String)).unique().sort(pr.SETTING)
        if not curves.is_empty()
        else pl.DataFrame(schema=[*pr.SETTING, "run_dir"])
    )
    # "dataset · model · acquisition · UQ metric" -> the comparison's run folder.
    run_options = {" · ".join(row[:4]): row[4] for row in best_runs.iter_rows()}
    return (
        best_runs,
        curves,
        llm,
        run_options,
        sizes,
        status,
        supervised,
        sweep_curves,
        winners,
    )


@app.cell
def _(curves, entity_matching, f1_average, pr, supervised, sweep_curves):
    # entity_f1 -> entity_macro_f1, entity_partial_f1 -> entity_partial_macro_f1.
    field = entity_matching.value.replace("_f1", f"{f1_average.value}_f1")
    matching = entity_matching.selected_key
    score_name = f"{matching}-match {'macro ' if f1_average.value else ''}entity F1"
    seeds = pr.seed_scores(curves, field)
    gaps = pr.gap_curves(curves, field)
    sweep = pr.config_scores(sweep_curves, field)
    bounds = pr.floor_and_ceiling(curves, supervised, field)
    dataset_bounds = pr.dataset_bounds(bounds, supervised, field)
    return (
        bounds,
        dataset_bounds,
        field,
        gaps,
        matching,
        score_name,
        seeds,
        sweep,
    )


@app.cell(hide_code=True)
def _(mo, pl, pr, sizes, status, winners):
    _stage_order = list(pr.STAGES.values())
    # Best UQ runs one sweep per model and acquisition.
    _expected = (
        pl.when(pl.col("stage") == pr.STAGES["best_uq"])
        .then(len(pr.ACQUISITIONS) * len(pr.MODELS))
        .otherwise(len(pr.MODELS))
    )
    _progress = (
        status.group_by("stage", "dataset")
        .agg(pl.col("done").sum().alias("finished"), (~pl.col("done")).sum().alias("running"))
        .with_columns(
            progress=pl.format("{}/{} done", "finished", _expected)
            + pl.when(pl.col("running") > 0)
            .then(pl.format(", {} running", "running"))
            .otherwise(pl.lit(""))
        )
        .pivot(on="dataset", index="stage", values="progress")
        .sort(pl.col("stage").replace_strict(_stage_order, list(range(len(_stage_order)))))
    )
    _progress = _progress.select(
        "stage",
        *[
            (pl.col(d) if d in _progress.columns else pl.lit(None, pl.String))
            .fill_null("not started")
            .alias(d)
            for d in pr.DATASET_ORDER
        ],
    )
    _open = status.filter(~pl.col("done")).select(
        "stage", "dataset", "model", "complete", "failed", "planned", "tested", "sweep"
    )
    mo.vstack(
        [
            mo.md(
                "## 1. Where the rerun stands\n"
                "Finished sweeps per stage and dataset, out of five models. Chapters use "
                "whatever is complete; **Reload results** in the floating panel picks up "
                "new runs."
            ),
            mo.ui.table(_progress, selection=None, label="Completed sweeps"),
            mo.md("### Sweeps in progress"),
            mo.ui.table(_open, selection=None)
            if _open.height
            else mo.md("Every started sweep is complete."),
            mo.callout(
                mo.md(
                    "A failed trial in a tuning sweep is a CUDA out-of-memory error from the "
                    "parallel chains, not a modelling failure; rerunning the script resumes it."
                ),
                kind="info",
            )
            if _open["failed"].sum()
            else mo.md(""),
            mo.md("### Datasets"),
            mo.ui.table(sizes, selection=None),
            mo.accordion(
                {
                    "Random-baseline validation winners": mo.ui.table(
                        winners,
                        selection=None,
                        format_mapping={
                            "validation_score": "{:.3f}",
                            "learning_rate": "{:.2e}",
                        },
                    )
                }
            )
            if winners.height
            else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, gap, gap_names, mo, pl, pr, score_name, seeds):
    _gap = gap.value
    _other = "final_gap" if _gap == "gap_auc" else "gap_auc"
    _shown = f"{_gap}_pp"
    _settings = seeds.group_by(pr.SETTING).agg(
        (100 * pl.col("gap_auc").mean()).alias("gap_auc_pp"),
        (100 * pl.col("gap_auc").std()).alias("gap_auc_sd_pp"),
        (100 * pl.col("final_gap").mean()).alias("final_gap_pp"),
        (100 * pl.col("final_gap").std()).alias("final_gap_sd_pp"),
        (pl.col(_gap) > 0).sum().alias("seeds_favouring_uq"),
        pl.len().alias("seeds"),
        (100 * pl.col("random_auc").mean()).alias("random_auc_pp"),
        (100 * pl.col("uq_auc").mean()).alias("uq_auc_pp"),
    )
    _by_dataset = {
        dataset: group for (dataset,), group in _settings.group_by("dataset", maintain_order=True)
    }
    _stats = [
        mo.stat(
            f"{(_settings[_shown] > 0).sum()}/{_settings.height}",
            label="Settings where UQ wins",
            caption="dataset × model × acquisition × UQ metric",
            bordered=True,
        ),
        mo.stat(
            f"{(seeds[_gap] > 0).sum()}/{seeds.height}",
            label="Model seeds where UQ wins",
            caption=f"per-seed {gap_names[_gap]} > 0",
            bordered=True,
        ),
        *[
            mo.stat(
                f"{_by_dataset[dataset][_shown].mean():+.2f} pp",
                label=f"Mean {gap_names[_gap]} · {dataset}",
                caption=f"{gap_names[_other]} {_by_dataset[dataset][f'{_other}_pp'].mean():+.2f} pp",
                bordered=True,
            )
            for dataset in pr.DATASET_ORDER
            if dataset in _by_dataset
        ],
    ]
    _limit = float(_settings[_shown].abs().max())
    _base = alt.Chart(_settings).encode(
        x=alt.X("uq_metric:N", title=None, sort=pr.UQ_METRICS),
        y=alt.Y("model:N", title=None, sort=pr.MODEL_ORDER),
    )
    _heatmap = (
        (
            _base.mark_rect().encode(
                color=alt.Color(
                    f"{_shown}:Q",
                    title=f"{gap_names[_gap].capitalize()} (pp)",
                    scale=alt.Scale(scheme="blueorange", domain=[-_limit, _limit]),
                ),
                tooltip=[
                    *pr.SETTING,
                    alt.Tooltip("gap_auc_pp:Q", format="+.2f"),
                    alt.Tooltip("gap_auc_sd_pp:Q", format=".2f"),
                    alt.Tooltip("final_gap_pp:Q", format="+.2f"),
                    alt.Tooltip("final_gap_sd_pp:Q", format=".2f"),
                    "seeds_favouring_uq",
                ],
            )
            + _base.mark_text(fontSize=12)
            .encode(
                text=alt.Text("label:N"),
            )
            .transform_calculate(
                label=f"format(datum.{_shown}, '+.1f') + ' (' + datum.seeds_favouring_uq + '/' + datum.seeds + ')'"
            )
        )
        .properties(width=330, height=190)
        .facet(
            row=alt.Row("acquisition:N", title=None, sort=pr.ACQUISITIONS),
            column=alt.Column("dataset:N", title=None, sort=pr.DATASET_ORDER),
        )
    )
    mo.vstack(
        [
            mo.md(
                "## 2. Headline: UQ beats random at the validation winners\n"
                f"Test {score_name} {gap_names[_gap].replace(' F1', '')} for every model, UQ metric, "
                "and acquisition, each model at its validation winner with 5 model seeds. "
                "Cells show the gap in pp and how many seeds favour UQ."
            ),
            mo.hstack(_stats, justify="start", gap=1, wrap=True),
            mo.ui.altair_chart(_heatmap),
            mo.accordion(
                {
                    "All settings": mo.ui.table(
                        _settings.sort(pr.SETTING),
                        selection=None,
                        page_size=20,
                        format_mapping={
                            column: "{:+.2f}" if "gap" in column else "{:.2f}"
                            for column in _settings.columns
                            if column.endswith("_pp")
                        },
                    ),
                    "How to read this": mo.md("""
    - The three UQ metrics of one model share its bootstrap and random arm, so they are
      not independent replications; Top-K and Gumbel noise share them too.
    - Seeds share the data split; a seed fixes model initialisation and selection.
    - The panel's **Gap** switches the cards and the heatmap between gap AUC and the
      final F1 gap: the last round's UQ-minus-random test entity F1, after both arms
      acquired the whole pool. The final gap is noisy on Quishpi (41 test sentences) and only reflects
      learning order. Gap AUC is the headline.
    - Chapter 9 checks the same comparison at 30 sampled configurations instead of
      one tuned configuration.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(acquisition, mo, pr, seeds, uq_metric):
    curve_dataset = mo.ui.dropdown(
        [d for d in pr.DATASET_ORDER if d in seeds["dataset"].unique().to_list()],
        value="PET",
        label="Dataset",
    )
    curve_view = mo.ui.radio(
        ["F1 curves", "Precision & recall", "Gap curves"],
        value="F1 curves",
        label="Show",
        inline=True,
    )
    mo.vstack(
        [
            mo.md(
                "## 3. Learning curves\n"
                f"Test entity F1 against the acquired pool, {uq_metric.value} against random "
                f"with {acquisition.value} acquisition. "
                "**F1 curves** show seed means with ±1 SD bands and the supervised upper "
                "bound (green dashes); **precision & recall** splits F1 into false "
                "positives (precision) and missed entities (recall), one row per arm; "
                "**gap curves** show UQ − random per round. The red line marks the pool "
                "percentage set in the floating panel."
            ),
            mo.hstack([curve_dataset, curve_view], justify="start", gap=2),
        ]
    )
    return curve_dataset, curve_view


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    curve_dataset,
    curve_view,
    curves,
    field,
    gaps,
    matching,
    mo,
    pl,
    pool_percent,
    pr,
    score_name,
    series_color,
    supervised,
    uq_metric,
):
    mo.stop(
        curve_view.value == "Precision & recall" and "_macro" in field,
        mo.callout(
            mo.md(
                "Precision and recall are saved for micro F1 only; set **F1 average** to "
                "micro in the floating panel."
            ),
            kind="warn",
        ),
    )
    _x = alt.X("percent:Q", title="Pool acquired (%)", scale=alt.Scale(domain=[0, 100]))

    def _marker(base, panels):
        """One red rule per facet panel at the floating panel's pool percentage.

        It reuses the layer's own data, because a faceted layer chart needs one dataset.
        """
        return (
            base.transform_aggregate(rows="count()", groupby=panels)
            .mark_rule(color="red", strokeWidth=1.5)
            .encode(x=alt.datum(pool_percent.value))
        )

    _selected = (
        (pl.col("dataset") == curve_dataset.value)
        & (pl.col("acquisition") == acquisition.value)
        & (pl.col("uq_metric") == uq_metric.value)
    )
    _color = series_color([uq_metric.value] + ["random"] * (curve_view.value != "Gap curves"))
    _runs = pr.arm_series(curves.filter(_selected))
    _facet = {"facet": alt.Facet("model:N", title=None, sort=pr.MODEL_ORDER), "columns": 5}
    _size = {"width": 215, "height": 230}
    if curve_view.value == "Precision & recall":
        _prefix = field.removesuffix("_f1")
        _measures = {
            field: "F1",
            f"{_prefix}_precision": "precision",
            f"{_prefix}_recall": "recall",
        }
        _data = (
            _runs.unpivot(
                on=list(_measures),
                index=["model", "series", "round", "percent_acquired"],
                variable_name="measure",
            )
            .with_columns(pl.col("measure").replace_strict(_measures))
            .group_by("model", "series", "round", "measure")
            .agg(pl.col("percent_acquired").mean().alias("percent"), pl.col("value").mean())
        )
        _base = alt.Chart(_data)
        _chart = (
            _base.mark_line(strokeWidth=2).encode(
                x=_x,
                y=alt.Y("value:Q", title=f"{matching.capitalize()}-match score"),
                color=alt.Color(
                    "measure:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["F1", "precision", "recall"],
                        range=["#9D9D9D", "#4C9BE8", "#F2A93B"],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                strokeDash=alt.condition(
                    alt.datum.measure == "F1", alt.value([4, 3]), alt.value([1, 0])
                ),
                tooltip=[
                    "model",
                    "series",
                    "measure",
                    alt.Tooltip("percent:Q", format=".1f"),
                    alt.Tooltip("value:Q", format=".3f"),
                ],
            )
        ) + _marker(_base, ["series", "model"])
        _facet = {
            "row": alt.Row("series:N", title=None, sort=["random", *pr.UQ_METRICS]),
            "column": alt.Column("model:N", title=None, sort=pr.MODEL_ORDER),
        }
        _size = {"width": 215, "height": 160}
    elif curve_view.value == "F1 curves":
        _data = (
            _runs.group_by("model", "series", "round")
            .agg(
                pl.col("percent_acquired").mean().alias("percent"),
                pl.col(field).mean().alias("mean"),
                pl.col(field).std().alias("sd"),
            )
            .with_columns(
                low=(pl.col("mean") - pl.col("sd")).clip(0, 1),
                high=(pl.col("mean") + pl.col("sd")).clip(0, 1),
            )
        )
        _sup = (
            supervised.filter(pl.col("dataset") == curve_dataset.value)
            .group_by("model")
            .agg(pl.col(field).mean().alias("supervised"))
        )
        _data = _data.join(_sup, on="model", how="left")
        _base = alt.Chart(_data)
        _chart = (
            _base.mark_area(opacity=0.15).encode(
                x=_x,
                y=alt.Y("low:Q", title=score_name[0].upper() + score_name[1:]),
                y2="high:Q",
                color=_color,
            )
            + _base.mark_line(strokeWidth=2).encode(
                x=_x,
                y=alt.Y("mean:Q", scale=alt.Scale(zero=False)),
                color=_color,
                tooltip=[
                    "model",
                    "series",
                    alt.Tooltip("percent:Q", format=".1f"),
                    alt.Tooltip("mean:Q", format=".3f"),
                    alt.Tooltip("sd:Q", format=".3f"),
                ],
            )
            + _base.mark_rule(strokeDash=[6, 4], color="#54A24B", strokeWidth=2).encode(
                y="mean(supervised):Q"
            )
            + _marker(_base, ["model"])
        )
    else:
        _data = (
            gaps.filter(_selected)
            .group_by("model", "uq_metric", "round")
            .agg(
                pl.col("percent_acquired").mean().alias("percent"),
                (100 * pl.col("gap")).mean().alias("mean"),
                (100 * pl.col("gap")).std().alias("sd"),
            )
            .with_columns(
                low=pl.col("mean") - pl.col("sd"),
                high=pl.col("mean") + pl.col("sd"),
                series=pl.col("uq_metric"),
            )
        )
        _base = alt.Chart(_data)
        _chart = (
            _base.mark_area(opacity=0.12).encode(
                x=_x,
                y=alt.Y("low:Q", title="UQ − random entity F1 (pp)"),
                y2="high:Q",
                color=_color,
            )
            + _base.mark_line(strokeWidth=2).encode(
                x=_x,
                y="mean:Q",
                color=_color,
                tooltip=[
                    "model",
                    "series",
                    alt.Tooltip("percent:Q", format=".1f"),
                    alt.Tooltip("mean:Q", format="+.2f"),
                    alt.Tooltip("sd:Q", format=".2f"),
                ],
            )
            + _base.mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.datum(0))
            + _marker(_base, ["model"])
        )
    mo.vstack(
        [
            mo.ui.altair_chart(_chart.properties(**_size).facet(**_facet)),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Bands are ±1 SD across the 5 model seeds, not confidence intervals.
    - The supervised line trains fresh models for a fixed number of epochs on the whole
      labelled pool (ADR 0006): an upper bound, not a matched-budget arm.
    - Both arms end on the same full pool, so the curves meet near 100%; the advantage
      is how much sooner UQ gets there.
    - The red line is the pool percentage from the floating panel's slider; the coverage
      charts below show what each arm had acquired by then.
    - **Precision & recall** shows seed means only. Precision below recall means more
      false positives than missed entities; recall below precision means the reverse.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, pr):
    coverage_model = mo.ui.dropdown(["all five", *pr.MODEL_ORDER], value="all five", label="Model")
    return (coverage_model,)


@app.cell
def _(
    Path,
    acquisition,
    best_runs,
    coverage_model,
    curve_dataset,
    json,
    pl,
    pr,
    tag_selections,
    uq_metric,
):
    # Parsing the selection logs is slow, so moving the pool slider reuses them.
    _runs = best_runs.filter(
        (pl.col("dataset") == curve_dataset.value)
        & (pl.col("acquisition") == acquisition.value)
        & (pl.col("uq_metric") == uq_metric.value)
    )
    if coverage_model.value != "all five":
        _runs = _runs.filter(pl.col("model") == coverage_model.value)
    _frames, coverage_order, coverage_results, coverage_partial = [], None, [], []
    for _model, _run_dir in _runs.select("model", "run_dir").iter_rows():
        _results = pl.read_csv(Path(_run_dir) / "results.csv")
        _ends = _results.group_by("seed", "arm").agg(pl.col("percent_acquired").max())
        if not (_ends["percent_acquired"] >= 100).all():
            coverage_partial.append(_model)
            continue
        _frame, _order = tag_selections(
            json.loads((Path(_run_dir) / "selections.json").read_text()), _results, _run_dir
        )
        _frames.append(_frame)
        coverage_order = coverage_order or [pr.tag_name(label) for label in _order]
        coverage_results.append(_results)
    # Every run of a dataset shares its pool, so the summaries average over models and seeds.
    coverage_selections = (
        pl.concat(_frames, how="diagonal_relaxed").with_columns(
            pl.col("label").map_elements(pr.tag_name, return_dtype=pl.String)
        )
        if _frames
        else None
    )
    return (
        coverage_order,
        coverage_partial,
        coverage_results,
        coverage_selections,
    )


@app.cell(hide_code=True)
def _(
    acquisition,
    coverage_model,
    coverage_order,
    coverage_partial,
    coverage_results,
    coverage_selections,
    curve_dataset,
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    pl,
    pool_percent,
    uq_metric,
):
    _heading = mo.md(
        "### What each arm acquired by the red line\n"
        f"How much of each gold tag each arm has acquired once {pool_percent.value}% of the "
        f"pool is labelled: {curve_dataset.value}, {uq_metric.value}, "
        f"{acquisition.value} acquisition, "
        + (
            "averaged over the five models and their seeds. "
            if coverage_model.value == "all five"
            else f"{coverage_model.value}, averaged over its seeds. "
        )
        + "The first chart counts each tag against its own tokens in the pool, which keeps "
        "rare tags visible; the second counts it against all scoreable pool tokens."
    )
    mo.stop(
        coverage_selections is None,
        mo.vstack(
            [
                _heading,
                coverage_model,
                mo.md("No completed run covers the whole pool for this choice."),
            ]
        ),
    )
    _acquired = [
        _results.filter(pl.col("percent_acquired") <= pool_percent.value)["n_acquired"].max() or 0
        for _results in coverage_results
    ]
    _note = (
        f"**Acquired words per arm: {min(_acquired):,}**"
        if min(_acquired) == max(_acquired)
        else f"**Acquired words per arm: {min(_acquired):,}–{max(_acquired):,}** across models"
    ) + ", the latest completed round at or below the red line."
    if coverage_partial:
        _note += (
            f" Left out because they did not acquire the whole pool: {', '.join(coverage_partial)}."
        )
    mo.vstack(
        [
            _heading,
            coverage_model,
            mo.md(_note),
            mo.ui.altair_chart(
                make_tag_category_coverage_chart(
                    coverage_selections,
                    pool_percent.value,
                    uq_metric.value,
                    label_order=coverage_order,
                )
            ),
            mo.ui.altair_chart(
                make_tag_coverage_chart(
                    coverage_selections,
                    pool_percent.value,
                    uq_metric.value,
                    label_order=coverage_order,
                )
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Bars are means over the model seeds of the chosen model, or over every model and seed
      with **all five**, and include zero counts. The model choice drives these charts
      only; the curves above always show all five. Both charts use full BIO tags, ordered
      by how many distinct pool words carry each tag.
    - Random acquisition sits near the red line's percentage on every tag in the first
      chart; bars far from it show what the uncertainty arm prioritizes or postpones.
    - Only completed rounds at or below the red line count, without interpolation. Models
      with a larger K reach a given percentage in fewer, bigger rounds.
    - Tag totals come from the completed acquisition logs; this retrospective view does
      not use gold labels to select words.
    - Medical tags show the dataset authors' type names; the saved runs store them by
      begin id (`B-T03` is `B-Activity`).
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(acquisition, mo, uq_metric):
    target_fraction = mo.ui.slider(
        start=0.80,
        stop=1.0,
        step=0.01,
        value=0.95,
        show_value=True,
        label="Target: share of random's full-pool F1",
    )
    mo.vstack(
        [
            mo.md(
                "## 4. Label efficiency\n"
                "How much of the pool each model's seed-mean curve needs before its test "
                "entity F1 first reaches a share of what random scores, averaged over seeds, "
                f"after acquiring the whole pool ({acquisition.value} acquisition). "
                "**Labeling efficiency** is random's pool divided by UQ's: random needs "
                f"that many times the labelled words {uq_metric.value} needs. Above 1 "
                "favours UQ."
            ),
            target_fraction,
        ]
    )
    return (target_fraction,)


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    curves,
    field,
    mo,
    pl,
    pr,
    series_color,
    sizes,
    target_fraction,
    uq_metric,
):
    _arms = curves.filter(pl.col("acquisition") == acquisition.value)
    _curves = pr.mean_curves(_arms, field)
    # Each model's seed-mean final random score; a target is a share of it.
    _random_final = _curves.filter(
        (pl.col("series") == "random") & (pl.col("round") == pl.col("round").max().over(pr.CURVE))
    ).select("dataset", "model", "acquisition", pl.col(field).alias("final"))

    def _targets(fraction):
        return _random_final.select(
            "dataset", "model", "acquisition", (fraction * pl.col("final")).alias("target")
        )

    def _efficiency(fraction):
        pools = pr.first_round_reaching(_curves, field, _targets(fraction), pr.CURVE)
        return pools, pr.labeling_efficiency(pools)

    _pools, _efficiencies = _efficiency(target_fraction.value)
    _summary = pr.efficiency_summary(
        _efficiencies.filter(pl.col("uq_metric") == uq_metric.value), ["dataset"]
    )

    def _times(value, lower_bound=False):
        return "–" if value is None else f"{'≥ ' if lower_bound else ''}{value:.1f}×"

    _cards = [
        mo.stat(
            _times(row["efficiency"], row["lower_bound"]),
            label=f"{row['dataset']}: random's labels per {uq_metric.value} label",
            caption=f"geometric mean over {row['models']} model{'s' * (row['models'] != 1)}, "
            f"{_times(row['low'])}–{_times(row['high'])}"
            + (f"; UQ misses for {row['uq_missed']}" if row["uq_missed"] else ""),
            bordered=True,
        )
        for row in _summary.iter_rows(named=True)
    ]
    _points = _pools.join(
        sizes.select("dataset", "scoreable_pool_words"), on="dataset", how="left"
    ).with_columns(words=(pl.col("pool") * pl.col("scoreable_pool_words") / 100).round(0))
    _missing = _points.filter(pl.col("pool").is_null())
    _points = _points.drop_nulls("pool")
    _span = _points.group_by("dataset", "model").agg(
        pl.col("pool").min().alias("low"), pl.col("pool").max().alias("high")
    )
    _y = alt.Y("model:N", title=None, sort=pr.MODEL_ORDER)
    _charts = [
        alt.layer(
            alt.Chart(_span.filter(pl.col("dataset") == dataset))
            .mark_rule(color="#bbb", strokeWidth=3)
            .encode(
                x=alt.X("low:Q", title="Pool acquired (%)", scale=alt.Scale(domain=[0, 100])),
                x2="high:Q",
                y=_y,
            ),
            alt.Chart(_points.filter(pl.col("dataset") == dataset))
            .mark_circle(size=120, opacity=0.9)
            .encode(
                x="pool:Q",
                y=_y,
                color=series_color(_points["series"].to_list()),
                tooltip=[
                    "dataset",
                    "model",
                    "series",
                    alt.Tooltip("pool:Q", format=".1f"),
                    "words",
                ],
            ),
        ).properties(title=dataset, width=380, height=180)
        for dataset in _points["dataset"].unique().sort().to_list()
    ]
    # The same ratio over a range of targets, so no single share carries the result.
    _sweep = pl.concat(
        [
            pr.efficiency_summary(
                _efficiency(fraction)[1].filter(pl.col("uq_metric") == uq_metric.value),
                ["dataset"],
            ).with_columns(target=pl.lit(100 * fraction))
            for fraction in [0.80 + 0.01 * step for step in range(21)]
        ]
    )
    _sweep_chart = alt.layer(
        alt.Chart(_sweep)
        .mark_line(point=True)
        .encode(
            x=alt.X("target:Q", title="Target: share of random's full-pool F1 (%)"),
            y=alt.Y("efficiency:Q", title=f"Random's labels per {uq_metric.value} label"),
            color=alt.Color("dataset:N", title=None, sort=pr.DATASET_ORDER),
            tooltip=[
                "dataset",
                alt.Tooltip("target:Q", format=".0f"),
                alt.Tooltip("efficiency:Q", format=".2f"),
                alt.Tooltip("low:Q", format=".2f"),
                alt.Tooltip("high:Q", format=".2f"),
                "lower_bound",
                "uq_missed",
            ],
        ),
        alt.Chart(_sweep).mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.datum(1)),
        alt.Chart(_sweep).mark_rule(color="red").encode(x=alt.datum(100 * target_fraction.value)),
    ).properties(width=420, height=220)
    # Robustness: single runs against the same targets, summarized by their median.
    _runs = pr.first_round_reaching(
        pr.arm_series(_arms),
        field,
        _targets(target_fraction.value),
        [*pr.RUN, "series"],
    )
    _per_run = pr.median_pool(_runs, ["dataset", "series"])
    mo.vstack(
        [
            mo.hstack(_cards, justify="start", gap=1),
            mo.ui.altair_chart(alt.hconcat(*_charts).resolve_scale(color="shared")),
            mo.md(
                "Never reached by the seed-mean curve, left out of the chart: "
                + ", ".join(
                    f"{row['dataset']} {row['model']} {row['series']}"
                    for row in _missing.iter_rows(named=True)
                )
                + "."
            )
            if _missing.height
            else mo.md(""),
            mo.md("### Labeling efficiency across targets"),
            mo.ui.altair_chart(_sweep_chart),
            mo.accordion(
                {
                    "Per model and UQ metric": mo.ui.table(
                        _efficiencies.select(
                            "dataset",
                            "model",
                            "uq_metric",
                            "uq_pool",
                            "random_pool",
                            "efficiency",
                            "lower_bound",
                        ),
                        selection=None,
                        page_size=15,
                        format_mapping={
                            "uq_pool": "{:.1f}",
                            "random_pool": "{:.1f}",
                            "efficiency": "{:.2f}",
                        },
                    ),
                    "Robustness: single runs": mo.vstack(
                        [
                            mo.md(
                                "Each model seed's own curve against the same target, "
                                "summarized by the median over runs (models × seeds); "
                                "a run that never reaches the target ranks beyond the "
                                "pool."
                            ),
                            mo.ui.table(
                                _per_run.rename({"curves": "runs"}),
                                selection=None,
                                format_mapping={"pool": "{:.1f}"},
                            ),
                        ]
                    ),
                    "How to read this": mo.md("""
    - The target is fixed per model: the chosen share of its random arm's final-round F1,
      averaged over the 5 model seeds. Random's seed-mean curve always reaches it by 100%.
    - Each point is the first observed round at which the model's seed-mean curve
      reaches the target, not interpolated; a later round can dip below again. Hover for
      the pool share in words.
    - Labeling efficiency is random's pool divided by UQ's, as in Beck et al. (2021); the
      cards and the line chart take its geometric mean over models, so 2× and 0.5×
      cancel out.
    - **Robustness** reads each seed's own curve instead of the seed mean. Single runs
      are noisier, which shows in the spread, not in the cards.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bounds, dataset_bounds, mo, score_name):
    _format = {
        column: "{:.3f}"
        for column in ("floor", "ceiling", "ceiling_low", "ceiling_high", "floor_to_ceiling")
    }
    mo.vstack(
        [
            mo.md(
                "## 5. Against the supervised ceiling\n"
                "A gap only means something next to the room it has to grow in. This "
                "chapter bounds every score between the bootstrap model and the supervised "
                "baseline, checks where both arms end against the ceiling, and measures how "
                "much of the remaining room UQ closes."
            ),
            mo.md(
                "### Floor and ceiling\n"
                f"Test {score_name} of the bootstrap model before any acquisition (floor) "
                "and of the supervised baseline trained on the whole labelled pool "
                "(ceiling), averaged over the five models."
            ),
            mo.ui.table(dataset_bounds, selection=None, format_mapping=_format)
            if dataset_bounds.height
            else mo.md("No completed runs yet."),
            mo.accordion(
                {
                    "Per model": mo.ui.table(
                        bounds,
                        selection=None,
                        format_mapping={
                            "floor": "{:.3f}",
                            "ceiling": "{:.3f}",
                            "ceiling_sd": "{:.3f}",
                        },
                    ),
                    "How to read this": mo.md("""
    - The floor is the random arm's round-0 row, which every arm and sweep shares: the
      bootstrap model trained on the five bootstrap sentences.
    - The ceiling interval is a 95% t-interval over every supervised test run of the
      dataset (five models × their seeds), so it covers model choice as well as seeds.
    - The supervised baseline is an upper bound of this protocol, not of the task: the
      annotators' own agreement is the real limit, and the small test splits (84 PET,
      41 Quishpi sentences) make every score noisy.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    curves,
    field,
    mo,
    pl,
    pr,
    score_name,
    series_color,
    supervised,
):
    _arms = pr.final_scores(curves.filter(pl.col("acquisition") == acquisition.value), field)
    _sup = supervised.group_by("dataset", "model").agg(
        pl.lit("supervised").alias("series"),
        pl.col(field).mean().alias("f1"),
        pl.col(field).std().alias("sd"),
    )
    _points = pl.concat([_sup, _arms.select(_sup.columns)])
    _widest = _points.drop_nulls("sd").sort("sd", descending=True).head(1).to_dicts()
    _spread = (
        f"; seeds vary most for {_widest[0]['model']} on {_widest[0]['dataset']}, "
        f"{_widest[0]['series']} (SD {100 * _widest[0]['sd']:.1f} pp)"
        if _widest
        else ""
    )
    _chart = (
        alt.Chart(_points)
        .mark_point(size=110, filled=True, opacity=0.9)
        .encode(
            x=alt.X("f1:Q", title=f"Final test {score_name}", scale=alt.Scale(zero=False)),
            y=alt.Y("model:N", title=None, sort=pr.MODEL_ORDER),
            color=series_color(_points["series"].to_list()),
            shape=alt.Shape("series:N", title=None, legend=None),
            tooltip=[
                "dataset",
                "model",
                "series",
                alt.Tooltip("f1:Q", format=".3f"),
                alt.Tooltip("sd:Q", format=".3f"),
            ],
        )
        .properties(width=430, height=200)
        .facet(column=alt.Column("dataset:N", title=None, sort=pr.DATASET_ORDER))
        .resolve_scale(x="independent")
    )
    _table = (
        supervised.group_by("dataset", "model")
        .agg(
            pl.col(field).mean().alias("test_f1"),
            pl.col(field).std().alias("test_f1_sd"),
            pl.col("validation_score").first(),
            pl.col("learning_rate").first(),
            pl.col("batch_size").first(),
            pl.col("weight_decay").first(),
            pl.col("epochs").first(),
        )
        .sort("dataset", "model")
    )
    mo.vstack(
        [
            mo.md(
                "### Final scores\n"
                "Final-round test F1 of both arms after acquiring the whole pool, next to "
                f"the supervised baseline, {acquisition.value} acquisition. Both arms end "
                "close to the supervised score, so sequential acquisition with replay costs "
                f"little final accuracy{_spread}."
            ),
            mo.ui.altair_chart(_chart),
            mo.accordion(
                {
                    "Supervised baseline: settings and test scores": mo.ui.table(
                        _table,
                        selection=None,
                        format_mapping={
                            "test_f1": "{:.3f}",
                            "test_f1_sd": "{:.3f}",
                            "validation_score": "{:.3f}",
                            "learning_rate": "{:.2e}",
                        },
                    ),
                    "How to read this": mo.md("""
    - The supervised baseline is tuned on the validation holdout over 30 sampled
      configurations (batch size, weight decay, learning rate) and trained for a fixed
      number of epochs, then scored on the test split with 5 model seeds (ADR 0006).
    - Its validation score uses exact match whatever the panel shows.
    - The x axis differs per dataset.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    ceiling_fraction = mo.ui.slider(
        start=0.8,
        stop=1.0,
        step=0.01,
        value=0.95,
        label="Share of the ceiling to reach",
        show_value=True,
    )
    return (ceiling_fraction,)


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    bounds,
    ceiling_fraction,
    curves,
    dataset_bounds,
    field,
    mo,
    pl,
    pr,
    score_name,
    series_color,
    uq_metric,
):
    _budgets = [10.0, 25.0, 50.0]
    _bounds = bounds.drop_nulls("ceiling")
    _runs = curves.filter(
        (pl.col("acquisition") == acquisition.value) & (pl.col("uq_metric") == uq_metric.value)
    ).join(_bounds.select("dataset", "model"), on=["dataset", "model"])
    mo.stop(
        _runs.is_empty(),
        mo.md(
            "### How much is left to gain\n"
            "Needs completed best-UQ runs with a finished supervised sweep for the same "
            "dataset and model."
        ),
    )
    _datasets = _runs["dataset"].unique().sort().to_list()
    _curve = (
        pr.arm_series(pr.scores_at_budgets(_runs, field, [float(b) for b in range(0, 101, 2)]))
        .group_by("dataset", "series", "budget")
        .agg((100 * pl.col("score")).mean().alias("score"))
    )
    _ceilings = dataset_bounds.select(
        "dataset",
        *[
            (100 * pl.col(column)).alias(name)
            for column, name in (
                ("ceiling", "ceiling"),
                ("ceiling_low", "low"),
                ("ceiling_high", "high"),
                ("floor", "floor"),
            )
        ],
    )
    _color = series_color(["random", uq_metric.value, "supervised"])
    _x = alt.X("budget:Q", title="Pool acquired (%)", scale=alt.Scale(domain=[0, 100]))

    def _panel(dataset):
        bound = _ceilings.filter(pl.col("dataset") == dataset)
        band = bound.with_columns(series=pl.lit("supervised"))
        return alt.layer(
            alt.Chart(band).mark_rect(opacity=0.18).encode(y="low:Q", y2="high:Q", color=_color),
            alt.Chart(band)
            .mark_rule(strokeDash=[6, 4], strokeWidth=2)
            .encode(
                y=alt.Y("ceiling:Q", title=f"Test {score_name} (%)"),
                color=_color,
                tooltip=[
                    alt.Tooltip("ceiling:Q", title="supervised", format=".1f"),
                    alt.Tooltip("low:Q", format=".1f"),
                    alt.Tooltip("high:Q", format=".1f"),
                ],
            ),
            alt.Chart(bound)
            .mark_rule(color="gray", strokeDash=[2, 3])
            .encode(y="floor:Q", tooltip=[alt.Tooltip("floor:Q", title="round 0", format=".1f")]),
            alt.Chart(_curve.filter(pl.col("dataset") == dataset))
            .mark_line(strokeWidth=2.5)
            .encode(
                x=_x,
                y="score:Q",
                color=_color,
                tooltip=["series", "budget", alt.Tooltip("score:Q", format=".1f")],
            ),
        ).properties(title=dataset, width=330, height=240)

    _figure = alt.hconcat(*[_panel(d) for d in _datasets]).resolve_scale(color="shared")
    _at = pr.scores_at_budgets(_runs, field, _budgets)
    _by_dataset = pr.headroom_summary(_at, _bounds, ["dataset"])
    _by_model = pr.headroom_summary(_at, _bounds, ["dataset", "model"])
    _ceiling_targets = _bounds.select(
        "dataset", "model", (ceiling_fraction.value * pl.col("ceiling")).alias("target")
    )
    _ceiling_pools = pr.first_round_reaching(
        pr.mean_curves(_runs, field), field, _ceiling_targets, pr.CURVE
    )
    _reach = pr.median_pool(_ceiling_pools, ["dataset", "series"])
    _ceiling_efficiency = pr.efficiency_summary(pr.labeling_efficiency(_ceiling_pools), ["dataset"])
    _run_reach = pr.median_pool(
        pr.first_round_reaching(pr.arm_series(_runs), field, _ceiling_targets, [*pr.RUN, "series"]),
        ["dataset", "series"],
    ).rename({"curves": "runs"})

    def _pool(dataset, series):
        rows = _reach.filter((pl.col("dataset") == dataset) & (pl.col("series") == series))
        if rows.is_empty():
            return "–", "0/0"
        row = rows.row(0, named=True)
        pool = "not reached" if row["pool"] is None else f"{row['pool']:.0f}%"
        return pool, f"{row['reached']}/{row['curves']}"

    def _ratio(dataset):
        rows = _ceiling_efficiency.filter(pl.col("dataset") == dataset)
        if rows.is_empty() or rows["efficiency"][0] is None:
            return ""
        row = rows.row(0, named=True)
        bound = "≥ " if row["lower_bound"] else ""
        return f"; random needs {bound}{row['efficiency']:.1f}× the labels"

    _cards = []
    for _dataset in _datasets:
        _uq_pool, _uq_reached = _pool(_dataset, uq_metric.value)
        _random_pool, _random_reached = _pool(_dataset, "random")
        _cards.append(
            mo.stat(
                f"{_uq_pool} vs {_random_pool}",
                label=f"{_dataset}: pool to {ceiling_fraction.value:.0%} of the ceiling",
                caption=f"UQ vs random, median model; reached by {_uq_reached} vs "
                f"{_random_reached} models{_ratio(_dataset)}",
                bordered=True,
            )
        )

    _pp_columns = ("uq", "gap", "gap_low", "gap_high", "ceiling", "headroom")

    def _table(summary, keys):
        return summary.select(
            *keys,
            pl.col("budget").alias("pool_pct"),
            (100 * pl.col("random")).alias("random"),
            *[pl.col(c) * 100 for c in _pp_columns],
            (100 * pl.col("share_closed")).alias("share_closed_pct"),
            "pairs",
        )

    _pp = {
        column: "{:+.1f}" if column.startswith("gap") else "{:.1f}"
        for column in ("random", *_pp_columns, "share_closed_pct")
    }
    mo.vstack(
        [
            mo.md(
                "### How much is left to gain\n"
                f"Each panel averages the five models' seeds for {uq_metric.value} and "
                f"random ({acquisition.value} acquisition) between the round-0 floor (grey "
                "dots) and the supervised ceiling (green band, 95% interval). The table "
                "reads the same curves at fixed pool budgets: **headroom** is how far "
                "random still is from the ceiling, **share closed** the part of it UQ "
                "gains."
            ),
            mo.ui.altair_chart(_figure),
            mo.ui.table(_table(_by_dataset, ["dataset"]), selection=None, format_mapping=_pp),
            ceiling_fraction,
            mo.hstack(_cards, justify="start", gap=1, wrap=True),
            mo.accordion(
                {
                    "Per model": mo.ui.table(
                        _table(_by_model, ["dataset", "model"]),
                        selection=None,
                        page_size=15,
                        format_mapping=_pp,
                    ),
                    "Pool to the ceiling, per model": mo.ui.table(
                        _ceiling_pools.pivot(
                            on="series", index=["dataset", "model"], values="pool"
                        ),
                        selection=None,
                        format_mapping={"random": "{:.1f}", uq_metric.value: "{:.1f}"},
                    ),
                    "Robustness: pool to the ceiling, single runs": mo.ui.table(
                        _run_reach, selection=None, format_mapping={"pool": "{:.1f}"}
                    ),
                    "How to read this": mo.md("""
    - Everything is in F1 points (pp). Models acquire a different K per round, so each
      run is interpolated linearly onto the same pool percentages before averaging.
    - **Gap** pairs each model seed's UQ and random scores; its 95% t-interval runs over
      those pairs (five models × five seeds). Pairs of one model share a bootstrap, so the
      interval is optimistic.
    - **Headroom** is the mean supervised score minus the mean random score, each model
      against its own supervised baseline; **share closed** is the mean gap divided by
      it, left empty when random already matches the ceiling. Above 100% UQ passes the
      supervised score; below 0% it trails random.
    - The ceiling is an upper bound of this protocol, trained on the whole labelled pool
      (ADR 0006); arms can pass it, which makes headroom negative.
    - **Pool to the ceiling** is the first observed round at which a model's seed-mean
      curve reaches the chosen share of its supervised score, unlike chapter 4, whose
      target is random's own final score. The cards show the median model, "not
      reached" once half the models never get there, and the geometric-mean labeling
      efficiency where UQ gets there; it is a lower bound (≥) when random does not.
      **Robustness** reads single runs instead.
    """),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, pr):
    llm_band = mo.ui.dropdown(
        {
            "Supervised": "supervised",
            "Random, final round": "random",
            **{f"UQ ({metric}), final round": metric for metric in pr.UQ_METRICS},
        },
        value="Supervised",
        label="Band",
    )
    return (llm_band,)


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    curves,
    field,
    llm,
    llm_band,
    mo,
    pl,
    pr,
    score_name,
    series_colors,
    supervised,
    uq_metric,
):
    mo.stop(
        llm.is_empty(),
        mo.md(
            "## 6. Against LLM annotation\n"
            "No finished LLM runs in `llm-annotation-eval/res/pred/`; run "
            "`configs/<dataset>/md-test-sentences.yaml` there."
        ),
    )
    _configs = (
        llm.group_by("dataset", "config", "reasoning_effort", "llm_model")
        .agg(*pr.score_aggs(field))
        .with_columns(
            low=pl.col("f1") - pl.col("sd").fill_null(0),
            high=pl.col("f1") + pl.col("sd").fill_null(0),
        )
        .sort("dataset", "f1", descending=[False, True])
    )
    _best = _configs.group_by("dataset").agg(pl.all().sort_by("f1").last())
    _arms = curves.filter(pl.col("acquisition") == acquisition.value)
    _columns = ["dataset", "method", "config", "llm_model", "runs", "f1", "sd"]
    _columns += ["precision", "recall"]
    _comparison = pl.concat(
        [
            _configs.with_columns(method=pl.lit("LLM")).select(_columns),
            pr.final_scores(_arms, field)
            .with_columns(
                method=pl.when(pl.col("series") == "random")
                .then(pl.lit("random"))
                .otherwise("UQ: " + pl.col("series")),
                config=pl.col("model").cast(pl.String),
                llm_model=pl.lit(None, pl.String),
            )
            .select(_columns),
        ]
    ).sort("dataset", "f1", descending=[False, True])
    _sup = (
        supervised.group_by("dataset", "model")
        .agg(pl.col(field).mean().alias("f1"))
        .group_by("dataset")
        .agg(pl.col("f1").min().alias("low"), pl.col("f1").max().alias("high"))
        if not supervised.is_empty()
        else pl.DataFrame(schema={"dataset": pl.String, "low": pl.Float64, "high": pl.Float64})
    )
    _band_method = {"supervised": None, "random": "random"}.get(
        llm_band.value, f"UQ: {llm_band.value}"
    )
    _band = (
        _comparison.filter(pl.col("method") == _band_method)
        .group_by("dataset")
        .agg(pl.col("f1").min().alias("low"), pl.col("f1").max().alias("high"))
        if _band_method
        else _sup
    )
    _datasets = _configs["dataset"].unique().sort().to_list()

    def _panel(dataset):
        rows = _configs.filter(pl.col("dataset") == dataset)
        y = alt.Y(
            "config:N",
            title=None,
            sort=rows["config"].to_list(),
            axis=alt.Axis(labelLimit=240),
        )
        band = (
            alt.Chart(_band.filter(pl.col("dataset") == dataset))
            .mark_rect(color=series_colors[llm_band.value], opacity=0.18)
            .encode(
                x="low:Q",
                x2="high:Q",
                tooltip=[alt.Tooltip("low:Q", format=".3f"), alt.Tooltip("high:Q", format=".3f")],
            )
        )
        bars = alt.Chart(rows).mark_rule(color="gray").encode(x="low:Q", x2="high:Q", y=y)
        points = (
            alt.Chart(rows)
            .mark_point(size=90, filled=True)
            .encode(
                x=alt.X("f1:Q", title=f"Test {score_name}", scale=alt.Scale(zero=False)),
                y=y,
                color=alt.Color(
                    "reasoning_effort:N",
                    title="Reasoning effort",
                    sort=["none", "low", "medium", "high", "default"],
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    "config",
                    "llm_model",
                    "runs",
                    alt.Tooltip("f1:Q", format=".3f"),
                    alt.Tooltip("sd:Q", format=".3f"),
                ],
            )
        )
        return (band + bars + points).properties(
            title=dataset, width=380, height=max(80, 22 * rows.height)
        )

    _chart = alt.hconcat(*[_panel(d) for d in _datasets]).resolve_scale(x="independent")

    # Each model's seed-mean curve against the best LLM's score of its dataset.
    _llm_target = _best.select("dataset", target="f1")
    _pools = pr.first_round_reaching(pr.mean_curves(_arms, field), field, _llm_target, pr.CURVE)
    _by_series = pr.median_pool(_pools, ["dataset", "series"])
    _efficiency = pr.efficiency_summary(
        pr.labeling_efficiency(_pools).filter(pl.col("uq_metric") == uq_metric.value),
        ["dataset"],
    )
    _run_reach = pr.median_pool(
        pr.first_round_reaching(pr.arm_series(_arms), field, _llm_target, [*pr.RUN, "series"]),
        ["dataset", "series"],
    ).rename({"curves": "runs"})
    _series = ["random", *pr.UQ_METRICS]
    _reach_table = (
        _pools.pivot(on="series", index=["dataset", "model"], values="pool")
        .select("dataset", "model", *[s for s in _series if s in set(_pools["series"])])
        .sort("dataset", "model")
    )

    def _median_pool(dataset, series):
        row = _by_series.filter((pl.col("dataset") == dataset) & (pl.col("series") == series)).row(
            0, named=True
        )
        pool = "not reached" if row["pool"] is None else f"{row['pool']:.0f}%"
        return f"{pool} ({row['reached']}/{row['curves']} models reach it)"

    def _ratio(dataset):
        rows = _efficiency.filter(pl.col("dataset") == dataset)
        if rows.is_empty() or rows["efficiency"][0] is None:
            return ""
        row = rows.row(0, named=True)
        bound = "≥ " if row["lower_bound"] else ""
        return (
            f" Random needs {bound}{row['efficiency']:.1f}× the labels "
            f"(geometric mean over the {row['models']} "
            f"model{'s' * (row['models'] != 1)} where UQ gets there)."
        )

    def _summary(dataset):
        best = _best.filter(pl.col("dataset") == dataset).row(0, named=True)
        sup = _sup.filter(pl.col("dataset") == dataset)
        sup_text = (
            f"; the supervised models score {sup['low'][0]:.3f}–{sup['high'][0]:.3f}"
            if sup.height
            else ""
        )
        series = set(_by_series.filter(pl.col("dataset") == dataset)["series"])
        if not {"random", uq_metric.value} <= series:
            return f"- **{dataset}**: best LLM {best['config']}, {best['f1']:.3f}{sup_text}."
        return (
            f"- **{dataset}**: best LLM {best['config']}, {best['f1']:.3f}{sup_text}. "
            f"Median model's pool to match it: {uq_metric.value} "
            f"{_median_pool(dataset, uq_metric.value)}, random "
            f"{_median_pool(dataset, 'random')}.{_ratio(dataset)}"
        )

    mo.vstack(
        [
            mo.md(
                "## 6. Against LLM annotation\n"
                "Test score of LLMs that annotate each test sentence from instructions and "
                "the 5 bootstrap sentences, with no training, next to a band of token "
                f"classifier scores ({llm_band.selected_key.lower()}, lowest to highest "
                "model). Then how much of the pool a "
                f"token classifier needs, {acquisition.value} acquisition, to match the "
                "best LLM.\n\n" + "\n".join(_summary(d) for d in _datasets)
            ),
            llm_band,
            mo.ui.altair_chart(_chart),
            mo.md(
                "LLM configurations next to each token classifier's final-round test "
                f"score after acquiring the whole pool, {acquisition.value} acquisition."
            ),
            mo.ui.table(
                _comparison,
                selection=None,
                page_size=25,
                format_mapping={c: "{:.3f}" for c in ("f1", "sd", "precision", "recall")},
            ),
            mo.accordion(
                {
                    "Pool acquired (%) to match the best LLM, per model": mo.ui.table(
                        _reach_table,
                        selection=None,
                        format_mapping={s: "{:.1f}" for s in _series},
                    ),
                    "Robustness: single runs": mo.ui.table(
                        _run_reach, selection=None, format_mapping={"pool": "{:.1f}"}
                    ),
                    "How to read this": mo.md("""
    - The LLM runs are `llm-annotation-eval`'s `md-test-sentences` task: each test sentence
      is sent on its own, without its document, to an open-weight model on the NHR@FAU
      gateway. The prompt holds the dataset's type definitions, a procedure, and the
      5 bootstrap sentences with their gold mentions as the format example, so the LLM
      sees the same labelled sentences the token classifiers start from.
    - Each configuration is one model at one reasoning effort, run 5 times; points are
      the mean, grey bars ±1 SD over runs.
    - Predictions are rescored here with `evaluate_predictions` on the rebuilt test
      split, so the panel's matching and F1 average apply. Types outside the dataset's
      schema count as wrong.
    - The best LLM is picked on the test split, which favours the LLM. The pool share is
      the first observed round at which a model's seed-mean curve reaches its score, not
      interpolated; a blank cell means it never gets there. The summary takes the median
      model for the panel's UQ metric and random, and the geometric-mean labeling
      efficiency, a lower bound (≥) where random never gets there. **Robustness** reads
      single runs instead, with the median over models × seeds.
    - In the table, random and UQ rows are a classifier's final round, mean ± SD over
      model seeds (`runs`); UQ rows are one per metric, `config` is the classifier.
    - Chapter 10 can show an LLM's predictions next to the token classifiers'.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, pl, pr, seeds, stats):
    _paired = (
        seeds.group_by(pr.SETTING)
        .agg((100 * pl.col("gap_auc").mean()).alias("gap"))
        .pivot(on="acquisition", index=["dataset", "model", "uq_metric"], values="gap")
        .drop_nulls()
        .with_columns(difference=pl.col("Gumbel noise") - pl.col("Top-K"))
    )
    _low = float(min(_paired["Top-K"].min(), _paired["Gumbel noise"].min())) - 0.5
    _high = float(max(_paired["Top-K"].max(), _paired["Gumbel noise"].max())) + 0.5
    _diagonal = (
        alt.Chart(pl.DataFrame({"x": [_low, _high]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q", y="x:Q")
    )
    _scatter = (
        alt.Chart(_paired)
        .mark_point(size=110, filled=True, opacity=0.85)
        .encode(
            x=alt.X("Top-K:Q", title="Top-K gap AUC (pp)", scale=alt.Scale(domain=[_low, _high])),
            y=alt.Y(
                "Gumbel noise:Q",
                title="Gumbel-noise gap AUC (pp)",
                scale=alt.Scale(domain=[_low, _high]),
            ),
            color=alt.Color("model:N", sort=pr.MODEL_ORDER),
            shape=alt.Shape("dataset:N"),
            tooltip=[
                "dataset",
                "model",
                "uq_metric",
                alt.Tooltip("Top-K:Q", format="+.2f"),
                alt.Tooltip("Gumbel noise:Q", format="+.2f"),
                alt.Tooltip("difference:Q", format="+.2f"),
            ],
        )
        .properties(width=420, height=420)
    )
    _test = stats.wilcoxon(_paired["difference"].to_list()) if _paired.height > 1 else None
    _by_dataset = (
        _paired.group_by("dataset")
        .agg(
            pl.col("difference").mean().alias("mean_difference_pp"),
            (pl.col("difference") > 0).sum().alias("gumbel_better"),
            pl.len().alias("pairs"),
        )
        .sort("dataset")
    )
    mo.vstack(
        [
            mo.md(
                "## 7. Top-K or Gumbel noise?\n"
                "Each point is one dataset, model, and UQ metric at the validation winner. "
                "Above the diagonal, Gumbel noise beat Top-K."
            ),
            mo.hstack(
                [
                    mo.ui.altair_chart(_scatter + _diagonal),
                    mo.vstack(
                        [
                            mo.stat(
                                f"{_paired['difference'].mean():+.2f} pp",
                                label="Gumbel − Top-K gap AUC",
                                caption=f"Gumbel better in {(_paired['difference'] > 0).sum()}/{_paired.height} pairs",
                                bordered=True,
                            ),
                            mo.md(
                                f"Wilcoxon signed-rank over the {_paired.height} pairs: "
                                f"p = {_test.pvalue:.3g}."
                                if _test
                                else ""
                            ),
                            mo.ui.table(
                                _by_dataset,
                                selection=None,
                                format_mapping={"mean_difference_pp": "{:+.2f}"},
                            ),
                        ]
                    ),
                ],
                justify="start",
                gap=3,
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Top-K and Gumbel noise run at the same frozen configuration and model seeds and
      share the random arm, so the difference is entirely the uncertainty arm's.
    - Pairs within one model share its bootstrap, so the Wilcoxon test is optimistic.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, gap, gap_names, mo, pl, pr, seeds, series_color):
    _name = gap_names[gap.value]
    _means = (
        seeds.group_by(pr.SETTING)
        .agg((100 * pl.col(gap.value).mean()).alias("gap"))
        .with_columns(series=pl.col("uq_metric"))
    )
    _bars = (
        alt.Chart(_means)
        .mark_bar()
        .encode(
            x=alt.X("uq_metric:N", title=None, sort=pr.UQ_METRICS, axis=None),
            y=alt.Y("mean(gap):Q", title=f"Mean {_name} across models (pp)"),
            color=series_color(pr.UQ_METRICS),
            tooltip=["uq_metric", alt.Tooltip("mean(gap):Q", format="+.2f")],
        )
        .properties(width=150, height=200)
    )
    _ticks = (
        alt.Chart(_means)
        .mark_tick(color="black", thickness=1, size=14, opacity=0.5)
        .encode(
            x=alt.X("uq_metric:N", sort=pr.UQ_METRICS),
            y="gap:Q",
            tooltip=["model", alt.Tooltip("gap:Q", format="+.2f")],
        )
    )
    _best = (
        _means.sort("gap", descending=True)
        .group_by("dataset", "model", "acquisition", maintain_order=True)
        .first()
    )
    _winners = {}
    for _acquisition in pr.ACQUISITIONS:
        _table = (
            _best.filter(pl.col("acquisition") == _acquisition)
            .group_by("dataset", "uq_metric")
            .len("times best")
            .pivot(on="dataset", index="uq_metric", values="times best")
            .fill_null(0)
        )
        _winners[_acquisition] = (
            pl.DataFrame({"uq_metric": pr.UQ_METRICS})
            .join(_table, on="uq_metric", how="left")
            .fill_null(0)
            .select("uq_metric", *[d for d in pr.DATASET_ORDER if d in _table.columns])
        )
    _most_best = {
        _acquisition: _table.select(
            "uq_metric", pl.sum_horizontal(pl.exclude("uq_metric")).alias("total")
        )
        .sort("total", descending=True)
        .row(0, named=True)
        for _acquisition, _table in _winners.items()
    }
    _leaders = " ".join(
        f"With {_acquisition}, **{_row['uq_metric']}** gives a model its largest gap most "
        f"often ({_row['total']} of {_best.filter(pl.col('acquisition') == _acquisition).height} times)."
        for _acquisition, _row in _most_best.items()
    )
    _worst = _means.sort("gap").row(0, named=True)
    _spread = (
        _means.group_by("dataset", "model", "acquisition")
        .agg((pl.col("gap").max() - pl.col("gap").min()).alias("spread"))["spread"]
        .mean()
    )
    mo.vstack(
        [
            mo.md(
                "## 8. Which UQ metric?\n"
                f"Bars are the mean {_name} over the five models; ticks are the individual "
                "models. Each table counts how often each metric gave a model its largest "
                "gap with that acquisition (5 models per dataset)."
            ),
            mo.hstack(
                [
                    mo.ui.altair_chart(
                        (_bars + _ticks).facet(
                            column=alt.Column("dataset:N", title=None, sort=pr.DATASET_ORDER),
                            row=alt.Row("acquisition:N", title=None, sort=pr.ACQUISITIONS),
                        )
                    ),
                    mo.vstack(
                        [
                            mo.ui.table(_table, selection=None, label=f"Times best, {_acquisition}")
                            for _acquisition, _table in _winners.items()
                        ],
                        gap=2,
                    ),
                ],
                justify="start",
                gap=3,
            ),
            mo.md(
                f"{_leaders} The weakest setting is "
                f"{_worst['uq_metric']} for {_worst['model']} on {_worst['dataset']} with "
                f"{_worst['acquisition']} ({_worst['gap']:+.2f} pp). Within a setting the "
                f"metrics differ by {_spread:.2f} pp on average, against a mean {_name} of "
                f"{_means['gap'].mean():+.2f} pp."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    alt,
    gap,
    gap_names,
    mo,
    pl,
    pr,
    score_name,
    series_color,
    status,
    sweep,
):
    _name = gap_names[gap.value]
    _ttests = pr.config_gap_ttests(sweep, gap.value) if sweep.height else pl.DataFrame()
    # Each sampled configuration runs once per UQ metric.
    _progress = status.filter(pl.col("stage") == pr.STAGES["random_search"]).select(
        "dataset",
        "model",
        "done",
        (pl.col("complete") // len(pr.UQ_METRICS)).alias("configs"),
        (pl.col("planned") // len(pr.UQ_METRICS)).alias("planned_configs"),
    )
    _so_far = (
        pl.when(pl.col("done"))
        .then(pl.lit(""))
        .otherwise(pl.format(" ({}/{} configs so far)", "configs", "planned_configs"))
    )
    _verdicts = {}
    if _ttests.height:
        for _row in (
            _ttests.group_by("dataset", "model")
            .agg(
                (pl.col("holm_p") < 0.05).sum().alias("significant"),
                pl.len().alias("metrics"),
            )
            .join(_progress, on=["dataset", "model"])
            .with_columns(so_far=_so_far)
            .sort("dataset", "model")
            .iter_rows(named=True)
        ):
            _verdict = (
                "all metrics"
                if _row["significant"] == _row["metrics"]
                else "no metric"
                if _row["significant"] == 0
                else "some metrics"
            )
            _verdicts.setdefault(_verdict, []).append(
                f"{_row['dataset']} {_row['model']}{_row['so_far']}"
            )
    _labelled = (
        sweep.join(_progress, on=["dataset", "model"])
        .sort("dataset", "model", "config_id", "uq_metric")
        .with_columns(
            panel=pl.format("{} · {}{}", "dataset", "model", _so_far),
            gap_pp=100 * pl.col(gap.value),
            series=pl.col("uq_metric"),
            # Spread the points of one panel and metric evenly; stable across renders.
            jitter=(0.618 * pl.int_range(pl.len()).over("dataset", "model", "uq_metric")) % 1,
        )
    )
    # One row per dataset, models in the usual order.
    _panels = _labelled["panel"].unique(maintain_order=True).to_list()
    _strip = (
        alt.Chart(_labelled)
        .mark_circle(size=45, opacity=0.7)
        .encode(
            x=alt.X("uq_metric:N", title=None, sort=pr.UQ_METRICS, axis=alt.Axis(labelAngle=-30)),
            xOffset=alt.XOffset("jitter:Q"),
            y=alt.Y("gap_pp:Q", title=f"{_name[0].upper()}{_name[1:]} (pp)"),
            color=series_color(pr.UQ_METRICS).legend(None),
            tooltip=[
                "config_id",
                "uq_metric",
                alt.Tooltip("gap_pp:Q", format="+.2f"),
                alt.Tooltip("learning_rate:Q", format=".2e"),
                "k",
                "update_passes",
                "replay_ratio",
                "batch_size",
            ],
        )
    )
    _mean = (
        alt.Chart(_labelled)
        .mark_tick(color="black", thickness=3, size=30)
        .encode(x=alt.X("uq_metric:N", sort=pr.UQ_METRICS), y="mean(gap_pp):Q")
    )
    _zero = alt.Chart(_labelled).mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.datum(0))
    _chart = (
        (_strip + _mean + _zero)
        .properties(width=170, height=220)
        .facet(facet=alt.Facet("panel:N", title=None, sort=_panels), columns=len(pr.MODEL_ORDER))
    )
    mo.vstack(
        [
            mo.md(
                "## 9. Robustness: UQ (Top-K) against random at 30 sampled configurations\n"
                "The same 30 configurations, sampled from the tuning search space before any "
                "results, run with both arms on the test split. This asks whether the "
                "headline depends on tuning: a positive mean here means UQ helps across "
                f"typical settings, not just at the winner ({score_name}). The sweep ran "
                "Top-K acquisition only, so the panel's acquisition choice does not apply."
            ),
            mo.ui.altair_chart(_chart)
            if sweep.height
            else mo.md("No completed configurations yet."),
            mo.md(f"### One-sample t-tests of {_name} across configurations"),
            mo.callout(
                mo.md(
                    f"UQ beats random in {_name} across configurations (Holm p < 0.05):\n\n"
                    + "\n".join(
                        f"- **{verdict}**: {', '.join(_names)}"
                        for verdict in ("all metrics", "some metrics", "no metric")
                        if (_names := _verdicts.get(verdict))
                    )
                ),
                kind="neutral",
            )
            if _verdicts
            else mo.md(""),
            mo.ui.table(
                _ttests,
                selection=None,
                page_size=25,
                format_mapping={
                    "mean_gap_pp": "{:+.2f}",
                    "ci_low_pp": "{:+.2f}",
                    "ci_high_pp": "{:+.2f}",
                    "p_value": "{:.2g}",
                    "holm_p": "{:.2g}",
                    "bonferroni_p": "{:.2g}",
                },
            )
            if _ttests.height
            else mo.md(""),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Each point is one configuration's gap, averaged over 5 model seeds; the black tick
      is the mean over configurations.
    - Holm and Bonferroni adjust across the three UQ metrics within each sweep. The
      metrics share each configuration's random arm, so they are correlated.
    - Sweeps still running are labelled with their progress; their tests use the
      configurations finished so far.
    - Configurations are sampled across the whole search space, including poor ones, so
      the spread is far wider than at the validation winners in chapter 2.
    - Points are spread sideways only to keep them apart; the offset carries no value.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ParallelCoordinates, gap, gap_names, mo, pl, pr, sweep):
    mo.stop(sweep.is_empty())
    _random = {"gap_auc": "random_auc", "final_gap": "random_final"}[gap.value]
    config_gaps = (
        sweep.group_by("dataset", "model", "config_id")
        .agg(
            pl.col(pr.CONFIG_PARAMETERS).first(),
            (100 * pl.col(_random).first()).alias("random_pp"),
            (100 * pl.col(gap.value).mean()).alias("gap_pp"),
        )
        .sort("dataset", "model", "config_id")
    )
    drivers_plot = mo.ui.anywidget(
        ParallelCoordinates(
            # HiPlot draws columns right to left; this puts the gap rightmost.
            config_gaps.select(
                "gap_pp",
                "random_pp",
                *reversed(pr.CONFIG_PARAMETERS[1:]),
                pl.col("learning_rate").log10().alias("log10_learning_rate"),
                pl.col("model").cast(pl.String),
                pl.col("dataset").cast(pl.String),
            ),
            color_by="gap_pp",
            height=480,
        )
    )
    _score = "AUC" if gap.value == "gap_auc" else "final F1"
    mo.vstack(
        [
            mo.md(
                "### What drives the gap across configurations?\n"
                "One line per configuration, coloured by `gap_pp`: its "
                f"{gap_names[gap.value]} in pp, averaged over the three UQ metrics. "
                f"`random_pp` is the random arm's {_score}. Brush an axis to select the "
                "lines crossing it, then Keep or Exclude them; drag an axis title to reorder."
            ),
            drivers_plot,
        ]
    )
    return config_gaps, drivers_plot


@app.cell(hide_code=True)
def _(config_gaps, drivers_plot, mo, pl, pr, stats):
    # Only Keep and Exclude reach Python; a brush alone does not.
    _kept = config_gaps[drivers_plot.widget.filtered_indices]
    _columns = [*pr.CONFIG_PARAMETERS, "random_pp"]
    _rho = pl.DataFrame(
        [
            {"sweep": f"{dataset} · {model}", "configurations": group.height}
            | {
                column: stats.spearmanr(group[column], group["gap_pp"]).statistic
                if group[column].n_unique() > 1
                else None
                for column in _columns
            }
            for (dataset, model), group in config_gaps.group_by(
                "dataset", "model", maintain_order=True
            )
            if group.height > 2
        ]
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{_kept['gap_pp'].mean():+.2f} pp" if _kept.height else "–",
                        label="Mean gap of the kept configurations",
                        caption=f"{_kept.height} of {config_gaps.height} configurations",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{(_kept['gap_pp'] > 0).sum()}/{_kept.height}",
                        label="Kept configurations favouring UQ",
                        bordered=True,
                    ),
                ],
                justify="start",
            ),
            mo.accordion(
                {
                    "Kept configurations": mo.ui.table(
                        _kept,
                        selection=None,
                        page_size=10,
                        format_mapping={
                            "learning_rate": "{:.2e}",
                            "random_pp": "{:.2f}",
                            "gap_pp": "{:+.2f}",
                        },
                    ),
                    "Spearman ρ of the gap against each setting, per sweep": mo.vstack(
                        [
                            mo.ui.table(
                                _rho,
                                selection=None,
                                format_mapping={c: "{:+.2f}" for c in _columns},
                            ),
                            mo.md(
                                "Computed over every configuration of the sweep, not just "
                                "the kept ones. Negative ρ against `random_pp` means UQ "
                                "helps most where random learns slowly. These are "
                                "exploratory, unadjusted correlations."
                            ),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, run_options):
    viewer_run = mo.ui.dropdown(
        run_options, value=next(iter(run_options), None), label="Run", searchable=True
    )
    viewer_lines = mo.ui.multiselect(
        ["Gold", "Uncertainty", "Random", "LLM"],
        value=["Gold", "Uncertainty", "Random"],
        label="Lines",
    )
    return viewer_lines, viewer_run


@app.cell
def _(Path, field, json, llm, mo, pl, pr, viewer_run):
    mo.stop(viewer_run.value is None)
    test_sentences, run_rows, run_predictions = pr.load_test_predictions(Path(viewer_run.value))
    viewer_dataset = json.loads((Path(viewer_run.value) / "config.json").read_text())["dataset"]
    # "config · run i" -> its test-predictions.jsonl, best configuration on the test split first.
    _llm_runs = (
        llm.filter(pl.col("dataset") == pr.DATASETS[viewer_dataset])
        .with_columns(pl.col(field).mean().over("config").alias("config_score"))
        .sort("config_score", "run", descending=[True, False])
        if not llm.is_empty()
        else llm
    )
    _llm_options = {
        f"{row['config']} · run {row['run']}": row["predictions_file"]
        for row in _llm_runs.iter_rows(named=True)
    }
    viewer_llm = mo.ui.dropdown(
        _llm_options, value=next(iter(_llm_options), None), label="LLM", searchable=True
    )
    entity_types = sorted(
        {
            tag[2:]
            for rows in (*run_predictions, [s["tags"] for s in test_sentences])
            for tags in rows
            for tag in tags
            if tag != "O"
        }
    )
    _rounds = run_rows["round"].unique().sort()
    _seeds = run_rows["seed"].unique().sort().to_list()
    viewer_seed = mo.ui.dropdown(
        {str(seed): seed for seed in _seeds}, value=str(_seeds[0]), label="Model seed"
    )
    viewer_round = mo.ui.slider(
        _rounds.min(), _rounds.max(), value=_rounds.max() // 4, label="Round", show_value=True
    )
    _documents = sorted({s["document_name"] for s in test_sentences})
    viewer_document = mo.ui.dropdown(
        ["All test sentences", *_documents], value=_documents[0], label="Document"
    )
    return (
        entity_types,
        run_predictions,
        run_rows,
        test_sentences,
        viewer_dataset,
        viewer_document,
        viewer_llm,
        viewer_round,
        viewer_seed,
    )


@app.cell(hide_code=True)
def _(
    Path,
    entity_types,
    field,
    html,
    llm,
    matching,
    mo,
    pl,
    pr,
    run_predictions,
    run_rows,
    score_name,
    test_sentences,
    viewer_dataset,
    viewer_document,
    viewer_lines,
    viewer_llm,
    viewer_round,
    viewer_run,
    viewer_seed,
):
    _colors = {
        kind: f"hsl({360 * i / len(entity_types):.0f} 70% 50% / 0.28)"
        for i, kind in enumerate(entity_types)
    }
    _rows = run_rows.with_row_index().filter(
        (pl.col("seed") == viewer_seed.value) & (pl.col("round") == viewer_round.value)
    )
    _arms = {
        name: _rows.filter(pl.col("arm") == arm).row(0, named=True)
        for name, arm in (("Uncertainty", "uncertainty"), ("Random", "random"))
    }
    _predicted = {line: run_predictions[_arms[line]["index"]] for line in ("Uncertainty", "Random")}
    if viewer_llm.value is not None:
        _predicted["LLM"] = pr.read_llm_predictions(Path(viewer_llm.value), viewer_dataset)
    _shown = [
        line
        for line in ("Gold", "Uncertainty", "Random", "LLM")
        if line in viewer_lines.value and (line == "Gold" or line in _predicted)
    ]

    def _render(tokens, entities):
        pieces = [html.escape(token) for token in tokens]
        for start, end, kind, matched in reversed(entities):
            outline = "" if matched else ";outline:1.5px dashed #E45756"
            # Grey for types outside the dataset's schema, which only an LLM predicts.
            color = _colors.get(kind, "hsl(0 0% 50% / 0.28)")
            pieces[start:end] = [
                f'<span style="background:{color};border-radius:3px;'
                f'padding:1px 3px{outline}" title="{kind}">{" ".join(pieces[start:end])}'
                f"<sub style='opacity:.7'> {html.escape(pr.type_name(kind))}</sub></span>"
            ]
        return " ".join(pieces)

    _reference = next((line for line in _shown if line != "Gold"), None)

    def _sentence(index, sentence):
        tags = sentence["tags"]
        matches = {
            line: pr.match_entities(tags, _predicted[line][index], matching)
            for line in _shown
            if line != "Gold"
        }
        lines = []
        for line in _shown:
            if line != "Gold":
                entities = matches[line][1]
                note = f"{sum(e[3] for e in entities)}/{len(entities)} correct"
            elif _reference:
                entities = matches[_reference][0]
                note = f"{sum(e[3] for e in entities)}/{len(entities)} found"
            else:
                entities = pr.match_entities(tags, tags, matching)[0]
                note = f"{len(entities)} entities"
            lines.append(
                f"<div style='display:flex;gap:8px;line-height:1.9'><span style='flex:0 0 "
                f"190px;white-space:nowrap;opacity:.6;font-size:.85em'>{line} · {note}</span>"
                f"<span>{_render(sentence['tokens'], entities)}</span></div>"
            )
        return (
            f"<tr><td style='opacity:.6;padding:4px 12px 8px 0;vertical-align:top;"
            f"white-space:nowrap'>{html.escape(str(sentence['document_name']))}"
            f"&nbsp;·&nbsp;{sentence['sentence_id']}</td>"
            f"<td style='text-align:left;padding-bottom:8px'>{''.join(lines)}</td></tr>"
        )

    _table = "".join(
        _sentence(index, sentence)
        for index, sentence in sorted(
            enumerate(test_sentences),
            key=lambda item: (item[1]["document_name"], item[1]["sentence_id"]),
        )
        if viewer_document.value in ("All test sentences", sentence["document_name"])
    )
    _legend = " ".join(
        f'<span style="background:{color};border-radius:3px;padding:1px 6px">{kind}</span>'
        for kind, color in _colors.items()
    )
    _uq, _random = _arms["Uncertainty"], _arms["Random"]
    _llm_score = (
        f", LLM ({viewer_llm.selected_key}) **"
        f"{llm.filter(pl.col('predictions_file') == viewer_llm.value)[field][0]:.3f}**"
        if "LLM" in _shown
        else ""
    )
    mo.vstack(
        [
            mo.md(
                "## 10. Gold and predicted entities on the test split\n"
                "Each best-UQ run saves the predicted tags behind every evaluation row, so "
                "any round of either arm can be read sentence by sentence. Red dashed "
                f"outlines mark entities with no {matching} match: on a predicted line, "
                "spurious entities; on the gold line, entities the first predicted line "
                "shown missed. The LLM line shows one run of an LLM configuration from "
                "chapter 6, which annotates the test sentences without training, so it does "
                "not change with the round; grey spans are types outside the schema."
            ),
            mo.hstack(
                [viewer_run, viewer_seed, viewer_round, viewer_document, viewer_lines, viewer_llm],
                justify="start",
                wrap=True,
            ),
            mo.md(
                f"Round {viewer_round.value} of {_uq['total_rounds']}, "
                f"{_uq['percent_acquired']:.1f}% of the pool acquired. Test {score_name}: "
                f"uncertainty **{_uq[field]:.3f}**, random "
                f"**{_random[field]:.3f}**{_llm_score}."
            ),
            mo.Html(f"<div>{_legend}</div><table style='width:100%'>{_table}</table>"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11. Caveats

    - **Test split sizes.** Quishpi tests on 41 sentences and PET on 84, so single-round
      scores move by several F1 points between seeds. Gap AUC averages over every round
      and is the stable summary; final gaps are noisy.
    - **Full-pool budget.** Both arms end with the same labelled pool. The final gap only
      reflects learning order, and the benefit of UQ is in the area before that.
    - **Shared arms.** A model's three UQ metrics, and its Top-K and Gumbel runs, share one
      bootstrap and one random arm. They are not independent replications.
    - **One tuned configuration.** Chapter 2 tunes the random arm on the validation holdout
      and gives UQ the same settings; chapter 9 is the check that the result does not
      hinge on that choice. The test-tuned random oracle is not part of this rerun.
    - **Partial match** is reported only (ADR 0010); tuning and the headline use exact match.
    """)
    return


if __name__ == "__main__":
    app.run()
