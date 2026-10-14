import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Paper rerun results

    The full rerun from `scripts/run_all_experiments.sh`: five models on PET, Quishpi,
    and MedicalProcessInstruks, read from `results/`. Each chapter answers one question
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
      objective) and partial match (reported only, ADR 0010), and picks the acquisition
      shown in chapters 3–5. Chapter 9 marks unmatched entities by the chosen matching.
    - **F1 average** switches every chapter between micro F1 (pooled over all entities,
      the headline) and macro F1 (the mean of per-type F1, so a rare type weighs as much
      as a common one; reported only). Precision and recall exist for micro F1 only.
    - Nothing is downloaded or trained. Per-run diagnostics (tag coverage, timing,
      selected words) are in `notebooks/best_uq_analysis.py`.
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

    from utils import paper_results as pr
    from utils.acquisition_analysis import tag_selections
    from utils.charts import make_tag_category_coverage_chart, make_tag_coverage_chart

    alt.data_transformers.disable_max_rows()
    results_root = Path(__file__).resolve().parents[1] / "results"
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
        make_tag_category_coverage_chart,
        make_tag_coverage_chart,
        mo,
        pl,
        pr,
        results_root,
        series_color,
        stats,
        tag_selections,
    )


@app.cell
def _(FloatingPanel, mo):
    refresh = mo.ui.run_button(label="Reload results")
    entity_matching = mo.ui.dropdown(
        {"exact": "entity_f1", "partial": "entity_partial_f1"},
        value="exact",
        label="Entity matching",
    )
    acquisition = mo.ui.dropdown(
        ["Top-K", "Gumbel noise"], value="Top-K", label="Acquisition (ch. 3–5)"
    )
    f1_average = mo.ui.dropdown({"micro": "", "macro": "_macro"}, value="micro", label="F1 average")
    pool_percent = mo.ui.slider(
        start=0, stop=100, step=1, value=25, label="Pool acquired, % (ch. 3)", show_value=True
    )
    FloatingPanel(mo.vstack([entity_matching, f1_average, acquisition, pool_percent, refresh]))
    return acquisition, entity_matching, f1_average, pool_percent, refresh


@app.cell
def _(pr, refresh, results_root):
    refresh.value
    status = pr.sweep_status(results_root)
    curves = pr.best_uq_curves(results_root)
    supervised = pr.supervised_scores(results_root)
    sizes = pr.dataset_sizes(results_root)
    winners = pr.tuning_winners(results_root)
    return curves, sizes, status, supervised, winners


@app.cell
def _(curves, entity_matching, f1_average, pr, results_root):
    # entity_f1 -> entity_macro_f1, entity_partial_f1 -> entity_partial_macro_f1.
    field = entity_matching.value.replace("_f1", f"{f1_average.value}_f1")
    matching = entity_matching.selected_key
    score_name = f"{matching}-match {'macro ' if f1_average.value else ''}entity F1"
    seeds = pr.seed_scores(curves, field)
    gaps = pr.gap_curves(curves, field)
    sweep = pr.random_sweep_comparisons(results_root, field)
    return field, gaps, matching, score_name, seeds, sweep


@app.cell(hide_code=True)
def _(mo, pl, pr, sizes, status, winners):
    _stage_order = list(pr.STAGES.values())
    # Best UQ runs two sweeps per model, Top-K and Gumbel noise.
    _expected = pl.when(pl.col("stage") == pr.STAGES["best_uq"]).then(10).otherwise(5)
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
                        winners.sort(
                            pl.col("dataset").replace_strict(
                                pr.DATASET_ORDER, list(range(len(pr.DATASET_ORDER)))
                            ),
                            pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))),
                        ),
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


@app.cell
def _(mo):
    headline_gap = mo.ui.dropdown(
        {"Gap AUC": "gap_auc", "Final F1 gap": "final_gap"}, value="Gap AUC", label="Gap"
    )
    return (headline_gap,)


@app.cell(hide_code=True)
def _(alt, headline_gap, mo, pl, pr, score_name, seeds):
    _gap = headline_gap.value
    _other = "final_gap" if _gap == "gap_auc" else "gap_auc"
    _names = {"gap_auc": "gap AUC", "final_gap": "final F1 gap"}
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
            caption=f"per-seed {_names[_gap]} > 0",
            bordered=True,
        ),
        *[
            mo.stat(
                f"{_by_dataset[dataset][_shown].mean():+.2f} pp",
                label=f"Mean {_names[_gap]} · {dataset}",
                caption=f"{_names[_other]} {_by_dataset[dataset][f'{_other}_pp'].mean():+.2f} pp",
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
                    title=f"{_names[_gap].capitalize()} (pp)",
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
            row=alt.Row("acquisition:N", title=None, sort=["Top-K", "Gumbel noise"]),
            column=alt.Column("dataset:N", title=None, sort=pr.DATASET_ORDER),
        )
    )
    mo.vstack(
        [
            mo.md(
                "## 2. Headline: UQ beats random at the validation winners\n"
                f"Test {score_name} {_names[_gap].replace(' F1', '')} for every model, UQ metric, "
                "and acquisition, each model at its validation winner with 5 model seeds. "
                "Cells show the gap in pp and how many seeds favour UQ."
            ),
            headline_gap,
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
    - **Gap** switches the cards and the heatmap between gap AUC and the final F1 gap:
      the last round's UQ-minus-random test entity F1, after both arms acquired the whole
      pool. The final gap is noisy on Quishpi (41 test sentences) and only reflects
      learning order. Gap AUC is the headline.
    - Chapter 8 checks the same comparison at 30 sampled configurations instead of
      one tuned configuration.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(acquisition, mo, pr, seeds):
    curve_dataset = mo.ui.dropdown(
        [d for d in pr.DATASET_ORDER if d in seeds["dataset"].unique().to_list()],
        value="PET",
        label="Dataset",
    )
    curve_metric = mo.ui.dropdown(pr.UQ_METRICS, value="entropy", label="UQ metric")
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
                f"Test entity F1 against the acquired pool, {acquisition.value} acquisition. "
                "**F1 curves** show seed means with ±1 SD bands and the supervised upper "
                "bound (green dashes); **precision & recall** splits F1 into false "
                "positives (precision) and missed entities (recall), one row per arm; "
                "**gap curves** show UQ − random per round. The red line marks the pool "
                "percentage set in the floating panel."
            ),
            mo.hstack([curve_dataset, curve_metric, curve_view], justify="start", gap=2),
        ]
    )
    return curve_dataset, curve_metric, curve_view


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    curve_dataset,
    curve_metric,
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
    _metrics = [curve_metric.value]
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

    _color = series_color(_metrics + ["random"] * (curve_view.value != "Gap curves"))
    _runs = curves.filter(
        (pl.col("dataset") == curve_dataset.value)
        & (pl.col("acquisition") == acquisition.value)
        & pl.col("uq_metric").is_in(_metrics)
    ).with_columns(
        series=pl.when(pl.col("arm") == "random")
        .then(pl.lit("random"))
        .otherwise(pl.col("uq_metric"))
    )
    # The UQ metrics share one random arm; keep a single copy of it.
    _runs = _runs.filter((pl.col("arm") == "uncertainty") | (pl.col("uq_metric") == _metrics[0]))
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
            gaps.filter(
                (pl.col("dataset") == curve_dataset.value)
                & (pl.col("acquisition") == acquisition.value)
                & pl.col("uq_metric").is_in(_metrics)
            )
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


@app.cell(hide_code=True)
def _(
    Path,
    acquisition,
    best_runs,
    coverage_model,
    curve_dataset,
    curve_metric,
    json,
    make_tag_category_coverage_chart,
    make_tag_coverage_chart,
    mo,
    pl,
    pool_percent,
    tag_selections,
):
    _heading = mo.md(
        "### What each arm acquired by the red line\n"
        f"How much of each gold tag each arm has acquired once {pool_percent.value}% of the "
        f"pool is labelled: {curve_dataset.value}, {curve_metric.value}, "
        f"{acquisition.value} acquisition, "
        + (
            "averaged over the five models and their seeds. "
            if coverage_model.value == "all five"
            else f"{coverage_model.value}, averaged over its seeds. "
        )
        + "The first chart counts each tag against its own tokens in the pool, which keeps "
        "rare tags visible; the second counts it against all scoreable pool tokens."
    )
    _runs = best_runs.filter(
        (pl.col("dataset") == curve_dataset.value)
        & (pl.col("acquisition") == acquisition.value)
        & (pl.col("uq_metric") == curve_metric.value)
        & ((pl.col("model") == coverage_model.value) | (coverage_model.value == "all five"))
    )
    _frames, _label_order, _partial, _acquired = [], None, [], []
    for _model, _run_dir in _runs.select("model", "run_dir").iter_rows():
        _results = pl.read_csv(Path(_run_dir) / "results.csv")
        _ends = _results.group_by("seed", "arm").agg(pl.col("percent_acquired").max())
        if not (_ends["percent_acquired"] >= 100).all():
            _partial.append(_model)
            continue
        _frame, _order = tag_selections(
            json.loads((Path(_run_dir) / "selections.json").read_text()), _results, _run_dir
        )
        _frames.append(_frame)
        _label_order = _label_order or _order
        _acquired.append(
            _results.filter(pl.col("percent_acquired") <= pool_percent.value)["n_acquired"].max()
            or 0
        )
    mo.stop(
        not _frames,
        mo.vstack(
            [
                _heading,
                coverage_model,
                mo.md("No completed run covers the whole pool for this choice."),
            ]
        ),
    )
    # Every run of a dataset shares its pool, so the summaries average over models and seeds.
    _selections = pl.concat(_frames, how="diagonal_relaxed")
    _note = (
        f"**Acquired words per arm: {min(_acquired):,}**"
        if min(_acquired) == max(_acquired)
        else f"**Acquired words per arm: {min(_acquired):,}–{max(_acquired):,}** across models"
    ) + ", the latest completed round at or below the red line."
    if _partial:
        _note += f" Left out because they did not acquire the whole pool: {', '.join(_partial)}."
    mo.vstack(
        [
            _heading,
            coverage_model,
            mo.md(_note),
            mo.ui.altair_chart(
                make_tag_category_coverage_chart(
                    _selections, pool_percent.value, curve_metric.value, label_order=_label_order
                )
            ),
            mo.ui.altair_chart(
                make_tag_coverage_chart(
                    _selections, pool_percent.value, curve_metric.value, label_order=_label_order
                )
            ),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - Bars are means over the model seeds of the chosen model, or over every model and seed
      with **all five**, and include zero counts. The model choice drives these charts
      only; the curves above always show all five. Both charts use full BIO tags, ordered by how many distinct pool words carry
      each tag.
    - Random acquisition sits near the red line's percentage on every tag in the first
      chart; bars far from it show what the uncertainty arm prioritizes or postpones.
    - Only completed rounds at or below the red line count, without interpolation. Models
      with a larger K reach a given percentage in fewer, bigger rounds.
    - Tag totals come from the completed acquisition logs; this retrospective view does
      not use gold labels to select words. Per-model, per-seed diagnostics are in
      `notebooks/best_uq_analysis.py`.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(acquisition, mo):
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
                "How much of the pool each arm needs before its test entity F1 first "
                "reaches a share of what random scores after acquiring the whole pool, "
                f"{acquisition.value} acquisition. Fewer labelled words is better."
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
):
    _reach = pr.pool_to_reach(
        curves.filter(pl.col("acquisition") == acquisition.value), field, target_fraction.value
    )
    _unreached = _reach["uq_pool"].null_count()
    # An arm that never reaches the target is counted at the full pool, which favours random.
    _reach = _reach.with_columns(pl.col("uq_pool").fill_null(100.0))
    _random = _reach.group_by("dataset", "model").agg(
        pl.col("random_pool").mean().alias("pool"), pl.lit("random").alias("series")
    )
    _uq = (
        _reach.group_by("dataset", "model", "uq_metric")
        .agg(pl.col("uq_pool").mean().alias("pool"))
        .rename({"uq_metric": "series"})
    )
    _points = (
        pl.concat([_random, _uq.select(_random.columns)])
        .join(sizes.select("dataset", "scoreable_pool_words"), on="dataset", how="left")
        .with_columns(words=(pl.col("pool") * pl.col("scoreable_pool_words") / 100).round(0))
    )
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
        for dataset in pr.DATASET_ORDER
        if dataset in _points["dataset"].unique().to_list()
    ]
    _summary = (
        _reach.group_by("dataset")
        .agg(pl.col("random_pool").mean(), pl.col("uq_pool").mean())
        .sort(
            pl.col("dataset").replace_strict(pr.DATASET_ORDER, list(range(len(pr.DATASET_ORDER))))
        )
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{row['uq_pool']:.0f}% vs {row['random_pool']:.0f}%",
                        label=f"{row['dataset']}: pool needed, UQ vs random",
                        caption=f"UQ labels {100 * (1 - row['uq_pool'] / row['random_pool']):.0f}% fewer words",
                        bordered=True,
                    )
                    for row in _summary.iter_rows(named=True)
                ],
                justify="start",
                gap=1,
            ),
            mo.ui.altair_chart(alt.hconcat(*_charts).resolve_scale(color="shared")),
            mo.md(
                f"UQ missed the target in **{_unreached}** seed run(s); they count as the full pool."
            )
            if _unreached
            else mo.md(""),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - The target is per model seed: the chosen share of that seed's random-arm F1 after
      the whole pool is acquired. Random always reaches it by 100%.
    - Crossings are the first observed round at or above the target, not interpolated;
      a later round can dip below again.
    - Each point averages the 5 model seeds; hover for the pool share in words.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    acquisition,
    alt,
    field,
    mo,
    pl,
    pr,
    score_name,
    seeds,
    series_color,
    supervised,
):
    _arms = seeds.filter(pl.col("acquisition") == acquisition.value)
    _random = (
        _arms.filter(pl.col("uq_metric") == pr.UQ_METRICS[0])
        .group_by("dataset", "model")
        .agg(
            pl.col("random_final").mean().alias("f1"),
            pl.col("random_final").std().alias("sd"),
            pl.lit("random").alias("series"),
        )
    )
    _uq = (
        _arms.group_by("dataset", "model", "uq_metric")
        .agg(pl.col("uq_final").mean().alias("f1"), pl.col("uq_final").std().alias("sd"))
        .rename({"uq_metric": "series"})
        .select(_random.columns)
    )
    _sup = supervised.group_by("dataset", "model").agg(
        pl.col(field).mean().alias("f1"),
        pl.col(field).std().alias("sd"),
        pl.lit("supervised").alias("series"),
    )
    _points = pl.concat([_sup, _random, _uq])
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
        .sort("dataset", pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))))
    )
    mo.vstack(
        [
            mo.md(
                "## 5. Against the supervised upper bound\n"
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
                "## 6. Top-K or Gumbel noise?\n"
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


@app.cell
def _(mo):
    gap_names = {"gap_auc": "gap AUC", "final_gap": "final gap"}
    _options = {"AUC": "gap_auc", "Final F1": "final_gap"}
    metric_gap = mo.ui.dropdown(_options, value="AUC", label="Gap in")
    sweep_gap = mo.ui.dropdown(_options, value="AUC", label="Gap in")
    return gap_names, metric_gap, sweep_gap


@app.cell(hide_code=True)
def _(alt, gap_names, metric_gap, mo, pl, pr, seeds, series_color):
    _name = gap_names[metric_gap.value]
    _means = (
        seeds.group_by(pr.SETTING)
        .agg((100 * pl.col(metric_gap.value).mean()).alias("gap"))
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
    _winners = (
        _means.sort("gap", descending=True)
        .group_by("dataset", "model", "acquisition", maintain_order=True)
        .first()
        .group_by("dataset", "uq_metric")
        .len("times best")
        .pivot(on="dataset", index="uq_metric", values="times best")
        .fill_null(0)
        .sort("uq_metric")
    )
    _winners = _winners.select("uq_metric", *[d for d in pr.DATASET_ORDER if d in _winners.columns])
    _most_best = (
        _winners.select("uq_metric", pl.sum_horizontal(pl.exclude("uq_metric")).alias("total"))
        .sort("total", descending=True)
        .row(0, named=True)
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
                "## 7. Which UQ metric?\n"
                f"Bars are the mean {_name} over the five models; ticks are the individual "
                "models. The table counts how often each metric gave a model its largest "
                "gap (5 models × 2 acquisitions per dataset)."
            ),
            metric_gap,
            mo.hstack(
                [
                    mo.ui.altair_chart(
                        (_bars + _ticks).facet(
                            column=alt.Column("dataset:N", title=None, sort=pr.DATASET_ORDER),
                            row=alt.Row(
                                "acquisition:N", title=None, sort=["Top-K", "Gumbel noise"]
                            ),
                        )
                    ),
                    mo.ui.table(_winners, selection=None, label="Times best per dataset"),
                ],
                justify="start",
                gap=3,
            ),
            mo.md(
                f"**{_most_best['uq_metric']}** gives a model its largest gap most often "
                f"({_most_best['total']} of {_means.height // 3} times). The weakest setting is "
                f"{_worst['uq_metric']} for {_worst['model']} on {_worst['dataset']} with "
                f"{_worst['acquisition']} ({_worst['gap']:+.2f} pp). Within a setting the "
                f"metrics differ by {_spread:.2f} pp on average, against a mean {_name} of "
                f"{_means['gap'].mean():+.2f} pp."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, gap_names, mo, pl, pr, score_name, status, sweep, sweep_gap):
    _name = gap_names[sweep_gap.value]
    _ttests = pr.config_gap_ttests(sweep, sweep_gap.value) if sweep.height else pl.DataFrame()
    _verdicts = {}
    if _ttests.height:
        for _row in (
            _ttests.group_by("dataset", "model")
            .agg(
                (pl.col("holm_p") < 0.05).sum().alias("significant"),
                pl.len().alias("metrics"),
                pl.col("configurations").min(),
            )
            .sort("dataset", pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))))
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
                f"{_row['dataset']} {_row['model']}"
                + (
                    f" ({_row['configurations']} configs so far)"
                    if _row["configurations"] < 30
                    else ""
                )
            )
    _progress = status.filter(pl.col("stage") == pr.STAGES["random_search"]).select(
        "dataset",
        "model",
        "done",
        (pl.col("complete") // 3).alias("configs"),
        (pl.col("planned") // 3).alias("planned_configs"),
    )
    _labelled = sweep.join(_progress, on=["dataset", "model"]).with_columns(
        panel=pl.format(
            "{} · {}{}",
            "dataset",
            "model",
            pl.when(pl.col("done"))
            .then(pl.lit(""))
            .otherwise(pl.format(" ({}/{} so far)", "configs", "planned_configs")),
        ),
        gap_pp=100 * pl.col(sweep_gap.value),
    )
    # One row per dataset, models in the usual order.
    _panels = (
        _labelled.sort(
            pl.col("dataset").replace_strict(pr.DATASET_ORDER, [0, 1, 2]),
            pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))),
        )["panel"]
        .unique(maintain_order=True)
        .to_list()
    )
    _strip = (
        alt.Chart(_labelled)
        .mark_circle(size=45, opacity=0.7)
        .encode(
            x=alt.X("uq_metric:N", title=None, sort=pr.UQ_METRICS, axis=alt.Axis(labelAngle=-30)),
            xOffset=alt.XOffset("jitter:Q"),
            y=alt.Y("gap_pp:Q", title=f"{_name[0].upper()}{_name[1:]} (pp)"),
            color=alt.Color(
                "uq_metric:N",
                legend=None,
                scale=alt.Scale(domain=pr.UQ_METRICS, range=["#F58518", "#E45756", "#B279A2"]),
            ),
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
        .transform_calculate(jitter="random()")
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
                "## 8. Robustness: UQ (Top-K) against random at 30 sampled configurations\n"
                "The same 30 configurations, sampled from the tuning search space before any "
                "results, run with both arms on the test split. This asks whether the "
                "headline depends on tuning: a positive mean here means UQ helps across "
                f"typical settings, not just at the winner ({score_name}). The sweep ran "
                "Top-K acquisition only, so the panel's acquisition choice does not apply."
            ),
            sweep_gap,
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
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ParallelCoordinates, gap_names, mo, pl, pr, sweep, sweep_gap):
    mo.stop(sweep.is_empty())
    config_drivers = [
        "learning_rate",
        "k",
        "update_passes",
        "replay_ratio",
        "batch_size",
        "weight_decay",
    ]
    _random = {"gap_auc": "random_auc", "final_gap": "random_final"}[sweep_gap.value]
    config_gaps = (
        sweep.group_by("dataset", "model", "config_id")
        .agg(
            pl.col(config_drivers).first(),
            (100 * pl.col(_random).first()).alias("random_pp"),
            (100 * pl.col(sweep_gap.value).mean()).alias("gap_pp"),
        )
        .sort(
            pl.col("dataset").replace_strict(pr.DATASET_ORDER, [0, 1, 2]),
            pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))),
            "config_id",
        )
    )
    drivers_plot = mo.ui.anywidget(
        ParallelCoordinates(
            # HiPlot draws columns right to left; this puts the gap rightmost.
            config_gaps.select(
                "gap_pp",
                "random_pp",
                *reversed(config_drivers[1:]),
                pl.col("learning_rate").log10().alias("log10_learning_rate"),
                "model",
                "dataset",
            ),
            color_by="gap_pp",
            height=480,
        )
    )
    _score = "AUC" if sweep_gap.value == "gap_auc" else "final F1"
    mo.vstack(
        [
            mo.md(
                "### What drives the gap across configurations?\n"
                "One line per configuration, coloured by `gap_pp`: its "
                f"{gap_names[sweep_gap.value]} in pp, averaged over the three UQ metrics. "
                f"`random_pp` is the random arm's {_score}. Brush an axis to select the "
                "lines crossing it, then Keep or Exclude them; drag an axis title to reorder."
            ),
            drivers_plot,
        ]
    )
    return config_drivers, config_gaps, drivers_plot


@app.cell(hide_code=True)
def _(config_drivers, config_gaps, drivers_plot, mo, pl, stats):
    # Only Keep and Exclude reach Python; a brush alone does not.
    _kept = config_gaps[drivers_plot.widget.filtered_indices]
    _columns = [*config_drivers, "random_pp"]
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
def _(pl, pr, refresh, results_root):
    refresh.value
    best_runs = pr.best_uq_runs(results_root).sort(
        pl.col("dataset").replace_strict(pr.DATASET_ORDER, [0, 1, 2]),
        pl.col("model").replace_strict(pr.MODEL_ORDER, list(range(5))),
        pl.col("acquisition").replace_strict(["Top-K", "Gumbel noise"], [0, 1]),
        "uq_metric",
    )
    # "dataset · model · acquisition · UQ metric" -> the comparison's run folder.
    run_options = {" · ".join(row[:4]): row[4] for row in best_runs.iter_rows()}
    return best_runs, run_options


@app.cell
def _(mo, run_options):
    viewer_run = mo.ui.dropdown(
        run_options, value=next(iter(run_options), None), label="Run", searchable=True
    )
    viewer_lines = mo.ui.multiselect(
        ["Gold", "Uncertainty", "Random"],
        value=["Gold", "Uncertainty", "Random"],
        label="Lines",
    )
    return viewer_lines, viewer_run


@app.cell
def _(Path, mo, pr, viewer_run):
    mo.stop(viewer_run.value is None)
    test_sentences, run_rows, run_predictions = pr.load_test_predictions(Path(viewer_run.value))
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
        viewer_document,
        viewer_round,
        viewer_seed,
    )


@app.cell(hide_code=True)
def _(
    entity_types,
    field,
    html,
    matching,
    mo,
    pl,
    pr,
    run_predictions,
    run_rows,
    score_name,
    test_sentences,
    viewer_document,
    viewer_lines,
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
    _shown = [line for line in ("Gold", "Uncertainty", "Random") if line in viewer_lines.value]

    def _render(tokens, entities):
        pieces = [html.escape(token) for token in tokens]
        for start, end, kind, matched in reversed(entities):
            outline = "" if matched else ";outline:1.5px dashed #E45756"
            pieces[start:end] = [
                f'<span style="background:{_colors[kind]};border-radius:3px;'
                f'padding:1px 3px{outline}" title="{kind}">{" ".join(pieces[start:end])}'
                f"<sub style='opacity:.7'> {kind}</sub></span>"
            ]
        return " ".join(pieces)

    _reference = next((line for line in _shown if line != "Gold"), None)

    def _sentence(index, sentence):
        tags = sentence["tags"]
        matches = {
            line: pr.match_entities(tags, run_predictions[_arms[line]["index"]][index], matching)
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
    mo.vstack(
        [
            mo.md(
                "## 9. Gold and predicted entities on the test split\n"
                "Each best-UQ run saves the predicted tags behind every evaluation row, so "
                "any round of either arm can be read sentence by sentence. Red dashed "
                f"outlines mark entities with no {matching} match: on a predicted line, "
                "spurious entities; on the gold line, entities the first predicted line "
                "shown missed."
            ),
            mo.hstack(
                [viewer_run, viewer_seed, viewer_round, viewer_document, viewer_lines],
                justify="start",
                wrap=True,
            ),
            mo.md(
                f"Round {viewer_round.value} of {_uq['total_rounds']}, "
                f"{_uq['percent_acquired']:.1f}% of the pool acquired. Test {score_name}: "
                f"uncertainty **{_uq[field]:.3f}**, random "
                f"**{_random[field]:.3f}**."
            ),
            mo.Html(f"<div>{_legend}</div><table style='width:100%'>{_table}</table>"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Caveats

    - **Test split sizes.** Quishpi tests on 41 sentences and PET on 84, so single-round
      scores move by several F1 points between seeds. Gap AUC averages over every round
      and is the stable summary; final gaps are noisy.
    - **Full-pool budget.** Both arms end with the same labelled pool. The final gap only
      reflects learning order, and the benefit of UQ is in the area before that.
    - **Shared arms.** A model's three UQ metrics, and its Top-K and Gumbel runs, share one
      bootstrap and one random arm. They are not independent replications.
    - **One tuned configuration.** Chapter 2 tunes the random arm on the validation holdout
      and gives UQ the same settings; chapter 8 is the check that the result does not
      hinge on that choice. The test-tuned random oracle is not part of this rerun.
    - **Partial match** is reported only (ADR 0010); tuning and the headline use exact match.
    - **Still running.** Chapter 1 lists unfinished sweeps; reload to include them.
    """)
    return


if __name__ == "__main__":
    app.run()
