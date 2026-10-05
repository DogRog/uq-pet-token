import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Test-tuned random oracle (appendix)

    An appendix sensitivity check, never a headline result. The headline protocol tunes
    the random arm on the validation holdout and freezes each checkpoint's validation
    winner (`configs/best_uq/`). Here, the random arm is tuned **on the test split**
    instead (ADR 0007):

    1. **Stage 1** tunes random only, on the full pool, over the same 100 sampled
       configurations, scores each one by seed-mean test entity-F1 AUC, and freezes the
       best as the test-tuned random oracle (`oracle_config.json`).
    2. **Stage 2** runs every UQ metric against random at that oracle configuration.

    This check is biased against UQ twice. The oracle random arm's test score carries the
    winner's curse of picking the best of 100 configurations on PET's 84 test sentences,
    and the uncertainty arm runs at the random arm's best settings rather than its own.
    If UQ still beats the oracle random arm, the headline gap is not just an artefact of
    under-tuning random.

    Reads `results/oracle_random/` without downloading or training models.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from utils.oracle_analysis import comparison_aucs, curve_auc, oracle_table, tuning_aucs

    return (
        Path,
        alt,
        comparison_aucs,
        curve_auc,
        json,
        mo,
        oracle_table,
        pl,
        tuning_aucs,
    )


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh saved results")
    refresh
    return (refresh,)


@app.cell
def _(Path, json, pl, refresh):
    refresh.value
    results_root = Path(__file__).resolve().parents[1] / "results"
    oracle_root = results_root / "oracle_random"

    def by_checkpoint(directory, required):
        """Map each checkpoint to its sweep directory under one results folder."""
        sweeps = {}
        for config_path in sorted(directory.glob("*/search_config.json")):
            if (config_path.parent / required).is_file():
                checkpoint = json.loads(config_path.read_text())["checkpoint"]
                sweeps[checkpoint] = config_path.parent
        return sweeps

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None

    tuning_sweeps = by_checkpoint(oracle_root, "summary.json")
    comparison_sweeps = by_checkpoint(oracle_root / "uq", "summary.json")
    validation_sweeps = by_checkpoint(results_root / "random_baseline_search", "selection.json")
    best_uq_sweeps = by_checkpoint(results_root / "best_uq", "summary.json")
    oracle_ids = {
        checkpoint: selection["config_id"]
        for checkpoint, sweep in tuning_sweeps.items()
        if (selection := read(sweep / "selection.json")) is not None
    }

    coverage_rows = []
    for checkpoint, sweep in tuning_sweeps.items():
        trials = read(sweep / "summary.json")["trials"]
        comparison = (
            read(comparison_sweeps[checkpoint] / "summary.json")
            if checkpoint in comparison_sweeps
            else None
        )
        coverage_rows.append(
            {
                "checkpoint": checkpoint,
                "stage_1_trials": f"{sum(t['status'] == 'complete' for t in trials)}/{len(trials)}",
                "stage_1_failed": sum(t["status"] == "failed" for t in trials),
                "oracle_config_id": oracle_ids.get(checkpoint),
                "stage_2_metrics": "not started"
                if comparison is None
                else f"{sum(c['status'] == 'complete' for c in comparison['comparisons'])}"
                f"/{len(comparison['comparisons'])}",
            }
        )
    coverage = pl.DataFrame(
        coverage_rows,
        schema={
            "checkpoint": pl.String,
            "stage_1_trials": pl.String,
            "stage_1_failed": pl.Int64,
            "oracle_config_id": pl.String,
            "stage_2_metrics": pl.String,
        },
    )
    return (
        best_uq_sweeps,
        comparison_sweeps,
        coverage,
        oracle_ids,
        read,
        tuning_sweeps,
        validation_sweeps,
    )


@app.cell
def _(coverage, mo):
    mo.stop(
        coverage.is_empty(),
        mo.md(
            "No oracle sweeps in `results/oracle_random/` yet. Run "
            "`bash scripts/run_oracle_random_all_models.sh` first."
        ),
    )
    mo.vstack(
        [
            mo.md(
                "Stage 1 must finish before an oracle is frozen; stage 2 starts after that. "
                "The appendix table includes checkpoints with at least one completed UQ "
                "metric."
            ),
            mo.ui.table(coverage, selection=None, label="Oracle sweep coverage"),
        ]
    )
    return


@app.cell
def _(
    comparison_aucs,
    comparison_sweeps,
    oracle_ids,
    oracle_table,
    pl,
    read,
    tuning_aucs,
    tuning_sweeps,
    validation_sweeps,
):
    tuning_by_checkpoint = {
        checkpoint: tuning_aucs(sweep) for checkpoint, sweep in tuning_sweeps.items()
    }
    validation_winners = {
        checkpoint: read(sweep / "selection.json")["config_id"]
        for checkpoint, sweep in validation_sweeps.items()
    }
    started = {
        checkpoint: comparison_aucs(comparison_sweeps[checkpoint])
        for checkpoint in oracle_ids
        if checkpoint in comparison_sweeps
    }
    compared = {checkpoint: aucs for checkpoint, aucs in started.items() if aucs.height}
    oracle = pl.concat(
        [
            oracle_table(
                tuning_by_checkpoint[checkpoint],
                aucs,
                oracle_ids[checkpoint],
                validation_winners.get(checkpoint),
            ).insert_column(0, pl.lit(checkpoint).alias("checkpoint"))
            for checkpoint, aucs in compared.items()
        ]
        or [pl.DataFrame()]
    )
    return oracle, tuning_by_checkpoint, validation_winners


@app.cell(hide_code=True)
def _(mo, oracle, pl):
    mo.stop(
        oracle.is_empty(),
        mo.md("No stage 2 comparisons yet; the appendix table appears once one completes."),
    )
    appendix = oracle.select(
        "checkpoint",
        "uq_metric",
        "configs_ranked",
        "oracle_config_id",
        "validation_config_id",
        (pl.col("validation_random_auc") * 100).alias("validation_random_auc_pp"),
        (pl.col("oracle_random_auc") * 100).alias("oracle_random_auc_pp"),
        "oracle_gain_pp",
        (pl.col("oracle_uq_auc") * 100).alias("oracle_uq_auc_pp"),
        "oracle_gap_pp",
        "oracle_seed_band_sd_pp",
        "oracle_z",
        "oracle_p_holm",
        "random_rerun_diff_pp",
    )
    mo.vstack(
        [
            mo.md("""
    ## Test-tuned random oracle: optimistically biased toward random; not a headline result

    AUC is normalized test entity-F1 AUC over acquired-pool percentage, averaged across
    model seeds, in percentage points.

    - **oracle_gain_pp**: how much choosing random's settings on the test split raises its
      own test AUC above the validation winner, both from stage 1. This is the inflation
      from tuning on the test split (true gain plus winner's curse).
    - **oracle_gap_pp**: UQ minus random at the oracle configuration, paired within each
      model seed in stage 2.
    - **oracle_z**: oracle_gap_pp divided by the seed band, the standard deviation of the
      random arm's AUC across the model seeds in stage 2.
    - **oracle_p_holm**: paired one-sample t-test of the per-seed gaps, Holm-adjusted
      across the three UQ metrics within each checkpoint.
    - **random_rerun_diff_pp**: stage 2's random arm minus stage 1's at the oracle
      configuration; it should be near zero.
            """),
            mo.ui.table(
                appendix,
                selection=None,
                page_size=15,
                label="Appendix: UQ against the test-tuned random oracle",
                format_mapping={
                    column: "{:.2f}"
                    for column in appendix.columns
                    if column.endswith("_pp") or column == "oracle_z"
                }
                | {"oracle_p_holm": "{:.3g}"},
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, oracle_ids, pl, tuning_by_checkpoint, validation_winners):
    random_by_config = pl.concat(
        [
            aucs.group_by("config_id")
            .agg((pl.col("random_auc").mean() * 100).alias("random_auc_pp"))
            .with_columns(pl.lit(checkpoint).alias("checkpoint"))
            for checkpoint, aucs in tuning_by_checkpoint.items()
            if aucs.height
        ]
        or [pl.DataFrame(schema={"config_id": pl.String, "random_auc_pp": pl.Float64})]
    )
    mo.stop(random_by_config.is_empty(), mo.md("No completed stage 1 trials yet."))
    roles = random_by_config.with_columns(
        pl.when(
            pl.col("config_id") == pl.col("checkpoint").replace_strict(oracle_ids, default=None)
        )
        .then(pl.lit("Test-tuned oracle"))
        .when(
            pl.col("config_id")
            == pl.col("checkpoint").replace_strict(validation_winners, default=None)
        )
        .then(pl.lit("Validation winner (headline)"))
        .otherwise(pl.lit("Other sampled configuration"))
        .alias("role")
    )
    random_chart = (
        alt.Chart(roles)
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X(
                "random_auc_pp:Q", title="Random arm test AUC (pp)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y("checkpoint:N", title=None),
            yOffset=alt.YOffset("role:N"),
            color=alt.Color(
                "role:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Other sampled configuration",
                        "Validation winner (headline)",
                        "Test-tuned oracle",
                    ],
                    range=["#b8bcc6", "#2f6fdb", "#d9822b"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            size=alt.condition(
                alt.datum.role == "Other sampled configuration", alt.value(30), alt.value(140)
            ),
            tooltip=[
                "checkpoint",
                "config_id",
                "role",
                alt.Tooltip("random_auc_pp:Q", format=".2f"),
            ],
        )
        .properties(width="container", height=260)
    )
    mo.vstack(
        [
            mo.md("""
    ## Stage 1: random arm across the sampled configurations

    Each point is one configuration's random-only test AUC. Once stage 1 finishes, the
    oracle is the right-most point; its distance from the validation winner is the oracle
    gain.
            """),
            random_chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    best_uq_sweeps,
    curve_auc,
    mo,
    pl,
    read,
    tuning_by_checkpoint,
    validation_winners,
):
    def best_uq_random_auc(sweep):
        """Seed-mean random AUC from the headline comparison; random is shared by metrics."""
        complete = [
            row
            for row in read(sweep / "summary.json")["comparisons"]
            if row["status"] == "complete"
        ]
        if not complete:
            return None
        aucs = curve_auc(pl.read_csv(sweep / complete[0]["run_dir"] / "results.csv"))
        random = [auc for (_, arm), auc in aucs.items() if arm == "random"]
        return sum(random) / len(random)

    reproduction_rows = []
    for model, model_aucs in tuning_by_checkpoint.items():
        if model not in best_uq_sweeps:
            continue
        winner = model_aucs.filter(pl.col("config_id") == (validation_winners.get(model) or ""))
        headline = best_uq_random_auc(best_uq_sweeps[model])
        reproduction_rows.append(
            {
                "checkpoint": model,
                "validation_config_id": validation_winners.get(model),
                "stage_1_random_auc_pp": 100 * winner["random_auc"].mean()
                if winner.height
                else None,
                "best_uq_random_auc_pp": None if headline is None else 100 * headline,
            }
        )
    reproduction = pl.DataFrame(
        reproduction_rows,
        schema={
            "checkpoint": pl.String,
            "validation_config_id": pl.String,
            "stage_1_random_auc_pp": pl.Float64,
            "best_uq_random_auc_pp": pl.Float64,
        },
    ).with_columns(
        (pl.col("stage_1_random_auc_pp") - pl.col("best_uq_random_auc_pp")).alias("difference_pp")
    )
    mo.vstack(
        [
            mo.md("""
    ## Reproduction check

    Stage 1 includes each checkpoint's validation winner. Its random arm should reproduce
    the random arm saved in `results/best_uq/`, up to GPU nondeterminism and a last-bit
    learning-rate roundoff in one non-winning configuration. A large difference points to
    a change in hardware, precision, or code between the runs.
            """),
            mo.ui.table(
                reproduction,
                selection=None,
                label="Validation winner: stage 1 against best_uq",
                format_mapping={
                    "stage_1_random_auc_pp": "{:.3f}",
                    "best_uq_random_auc_pp": "{:.3f}",
                    "difference_pp": "{:.3f}",
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Limitation: the validation holdout is large relative to the bootstrap

    Random-baseline tuning scores hyperparameters on 66 labelled pool sentences, about 13
    times the 5 bootstrap sentences that active learning starts from. A practitioner with
    that many labelled sentences would usually train on them, so a real low-resource
    annotation project could not reproduce this kind of tuning. This limits how realistic
    the absolute scores are, not the fairness of the comparison: only the random arm is
    tuned, and the uncertainty arm inherits its settings, so the holdout favours random.
    The supervised baseline is tuned on the same holdout and carries the same caveat.
    """)
    return


if __name__ == "__main__":
    app.run()
