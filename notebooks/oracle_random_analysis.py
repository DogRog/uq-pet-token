import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Test-tuned random oracle (appendix)

    An appendix sensitivity check, never a headline result. The headline protocol tunes
    the random arm on the validation holdout and freezes each checkpoint's winner
    (`configs/best_uq/`). Here, the random arm's hyperparameters are instead **chosen by
    their test score** among the same 100 sampled configurations, and every UQ metric is
    compared against random at that configuration (ADR 0007).

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

    from utils.oracle_analysis import config_gap_table, curve_auc, oracle_table, sweep_aucs

    return (
        Path,
        alt,
        config_gap_table,
        curve_auc,
        json,
        mo,
        oracle_table,
        pl,
        sweep_aucs,
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

    oracle_sweeps = by_checkpoint(oracle_root, "summary.json")
    validation_sweeps = by_checkpoint(results_root / "random_baseline_search", "selection.json")
    best_uq_sweeps = by_checkpoint(results_root / "best_uq", "summary.json")

    coverage = pl.DataFrame(
        [
            {
                "checkpoint": checkpoint,
                "sweep": sweep.name,
                "complete": summary["complete"],
                "completed_comparisons": sum(
                    row["status"] == "complete" for row in summary["comparisons"]
                ),
                "planned_comparisons": len(summary["comparisons"]),
                "failed_comparisons": sum(
                    row["status"] == "failed" for row in summary["comparisons"]
                ),
            }
            for checkpoint, sweep in oracle_sweeps.items()
            for summary in [json.loads((sweep / "summary.json").read_text())]
        ],
        schema={
            "checkpoint": pl.String,
            "sweep": pl.String,
            "complete": pl.Boolean,
            "completed_comparisons": pl.Int64,
            "planned_comparisons": pl.Int64,
            "failed_comparisons": pl.Int64,
        },
    )
    return best_uq_sweeps, coverage, oracle_root, oracle_sweeps, validation_sweeps


@app.cell
def _(coverage, mo):
    mo.stop(
        coverage.is_empty(),
        mo.md(
            "No oracle sweeps in `results/oracle_random/` yet. Run "
            "`bash scripts/run_oracle_random_all_models.sh` first."
        ),
    )
    all_complete = bool(coverage["complete"].all())
    mo.vstack(
        [
            mo.md(
                "**All sweeps complete.**"
                if all_complete
                else "**Provisional:** some sweeps are incomplete, so the oracle below is the "
                "best of the configurations finished so far and may still change."
            ),
            mo.ui.table(coverage, selection=None, label="Oracle sweep coverage"),
        ]
    )
    return (all_complete,)


@app.cell
def _(
    config_gap_table,
    json,
    oracle_sweeps,
    oracle_table,
    pl,
    sweep_aucs,
    validation_sweeps,
):
    aucs_by_checkpoint = {
        checkpoint: sweep_aucs(sweep) for checkpoint, sweep in oracle_sweeps.items()
    }
    validation_winners = {
        checkpoint: json.loads((sweep / "selection.json").read_text())["config_id"]
        for checkpoint, sweep in validation_sweeps.items()
    }
    ready = {checkpoint: aucs for checkpoint, aucs in aucs_by_checkpoint.items() if aucs.height}
    oracle = pl.concat(
        [
            oracle_table(aucs, validation_winners.get(checkpoint)).insert_column(
                0, pl.lit(checkpoint).alias("checkpoint")
            )
            for checkpoint, aucs in ready.items()
        ]
    )
    across_configs = pl.concat(
        [
            config_gap_table(aucs).insert_column(0, pl.lit(checkpoint).alias("checkpoint"))
            for checkpoint, aucs in ready.items()
        ]
    )
    return across_configs, oracle, ready, validation_winners


@app.cell(hide_code=True)
def _(mo, oracle, pl):
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
        "uq_best_config_id",
        "symmetric_gap_pp",
    )
    mo.vstack(
        [
            mo.md("""
    ## Test-tuned random oracle: optimistically biased toward random; not a headline result

    AUC is normalized test entity-F1 AUC over acquired-pool percentage, averaged across
    model seeds, in percentage points.

    - **oracle_gain_pp**: how much choosing random's settings on the test split raises its
      own test AUC above the validation-tuned winner. This is the inflation from tuning on
      the test split (true gain plus winner's curse).
    - **oracle_gap_pp**: UQ minus the oracle random arm, at the oracle configuration.
    - **oracle_z**: oracle_gap_pp divided by the seed band, the standard deviation of the
      random arm's AUC across the model seeds at the oracle configuration.
    - **oracle_p_holm**: paired one-sample t-test of the per-seed gaps, Holm-adjusted
      across the three UQ metrics within each checkpoint.
    - **symmetric_gap_pp**: each metric's own best configuration by test AUC against the
      same oracle random arm, so that both arms are tuned on the test split.
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
def _(alt, mo, pl, ready, validation_winners):
    random_by_config = pl.concat(
        [
            aucs.unique(["config_id", "seed"])
            .group_by("config_id")
            .agg((pl.col("random_auc").mean() * 100).alias("random_auc_pp"))
            .with_columns(pl.lit(checkpoint).alias("checkpoint"))
            for checkpoint, aucs in ready.items()
        ]
    )
    oracle_ids = (
        random_by_config.sort(["random_auc_pp", "config_id"], descending=[True, False])
        .group_by("checkpoint", maintain_order=True)
        .first()
        .select("checkpoint", pl.col("config_id").alias("oracle_id"))
    )
    roles = random_by_config.join(oracle_ids, on="checkpoint").with_columns(
        pl.when(pl.col("config_id") == pl.col("oracle_id"))
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
    ## Random arm across the 100 sampled configurations

    Each point is one configuration's random-arm test AUC. The oracle is the right-most
    point by construction; its distance from the validation winner is the oracle gain.
            """),
            random_chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(across_configs, mo):
    mo.vstack(
        [
            mo.md("""
    ## UQ against random with no tuning

    The UQ-minus-random gap at every completed configuration, with no hyperparameter
    selection on any split. This table does not depend on the validation holdout or on
    the test-tuned oracle.
            """),
            mo.ui.table(
                across_configs,
                selection=None,
                page_size=15,
                label="UQ minus random across all sampled configurations",
                format_mapping={
                    "median_gap_pp": "{:.2f}",
                    "mean_gap_pp": "{:.2f}",
                    "min_gap_pp": "{:.2f}",
                    "win_fraction": "{:.0%}",
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(best_uq_sweeps, curve_auc, json, mo, pl, ready, validation_winners):
    def best_uq_random_auc(sweep):
        """Seed-mean random AUC from the headline comparison; random is shared by metrics."""
        summary = json.loads((sweep / "summary.json").read_text())
        complete = [row for row in summary["comparisons"] if row["status"] == "complete"]
        if not complete:
            return None
        aucs = curve_auc(pl.read_csv(sweep / complete[0]["run_dir"] / "results.csv"))
        random = [auc for (_, arm), auc in aucs.items() if arm == "random"]
        return sum(random) / len(random)

    reproduction = pl.DataFrame(
        [
            {
                "checkpoint": checkpoint,
                "validation_config_id": validation_winners.get(checkpoint),
                "oracle_sweep_random_auc_pp": 100 * winner["random_auc"].mean()
                if winner.height
                else None,
                "best_uq_random_auc_pp": 100 * headline
                if (headline := best_uq_random_auc(best_uq_sweeps[checkpoint])) is not None
                else None,
            }
            for checkpoint, aucs in ready.items()
            if checkpoint in best_uq_sweeps
            for winner in [
                aucs.filter(
                    pl.col("config_id") == (validation_winners.get(checkpoint) or "")
                ).unique(["seed"])
            ]
        ],
        schema={
            "checkpoint": pl.String,
            "validation_config_id": pl.String,
            "oracle_sweep_random_auc_pp": pl.Float64,
            "best_uq_random_auc_pp": pl.Float64,
        },
    ).with_columns(
        (pl.col("oracle_sweep_random_auc_pp") - pl.col("best_uq_random_auc_pp")).alias(
            "difference_pp"
        )
    )
    mo.vstack(
        [
            mo.md("""
    ## Reproduction check

    The oracle sweep includes each checkpoint's validation winner. Its random arm should
    reproduce the random arm saved in `results/best_uq/`, up to GPU nondeterminism and a
    last-bit learning-rate roundoff in one non-winning configuration.
            """),
            mo.ui.table(
                reproduction,
                selection=None,
                label="Validation winner: oracle sweep against best_uq",
                format_mapping={
                    "oracle_sweep_random_auc_pp": "{:.3f}",
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
    The supervised baseline is tuned on the same holdout and carries the same caveat. The
    table of UQ against random with no tuning shows how the comparison behaves without
    any holdout at all.
    """)
    return


@app.cell
def _(all_complete, mo):
    save = mo.ui.run_button(
        label="Record oracle selection",
        disabled=not all_complete,
        tooltip="Available once every oracle sweep is complete.",
    )
    save
    return (save,)


@app.cell
def _(json, mo, oracle, oracle_root, oracle_sweeps, save):
    mo.stop(not save.value)
    records = []
    for checkpoint, oracle_id in (
        oracle.select("checkpoint", "oracle_config_id").unique().sort("checkpoint").iter_rows()
    ):
        plan = json.loads((oracle_sweeps[checkpoint] / "plan.json").read_text())
        parameters = next(
            entry["parameters"]
            for entry in plan["configurations"]
            if entry["config_id"] == oracle_id
        )
        records.append(
            {
                "checkpoint": checkpoint,
                "sweep": oracle_sweeps[checkpoint].name,
                "config_id": oracle_id,
                "parameters": parameters,
            }
        )
    selection_path = oracle_root / "oracle_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_split": "test",
                "use": "appendix sensitivity only; never a headline or frozen configuration",
                "objective": "seed-mean random test entity-F1 AUC; ties to lowest config ID",
                "oracles": records,
            },
            indent=2,
        )
    )
    mo.md(f"Recorded `{selection_path.relative_to(oracle_root.parent.parent)}`.")
    return


if __name__ == "__main__":
    app.run()
