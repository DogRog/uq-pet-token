import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, np, pl


@app.cell
def _(Path, json, np, pl):
    sweep_dir = (
        Path(__file__).resolve().parents[1]
        / "results/random_baseline_search/distilbert-random-baseline-100"
    )
    summary = json.loads((sweep_dir / "summary.json").read_text())
    selection = json.loads((sweep_dir / "selection.json").read_text())
    plan = json.loads((sweep_dir / "plan.json").read_text())
    trials = pl.DataFrame(
        [
            {"config_id": trial["config_id"], "score": trial["score"], **trial["parameters"]}
            for trial in summary["trials"]
            if trial["status"] == "complete"
        ]
    ).sort("score", "config_id", descending=[True, False])
    validation_frames = []
    seed_records = []
    for trial in summary["trials"]:
        if trial["status"] != "complete":
            continue
        run_dir = sweep_dir / trial["run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        assert config["random_only"] and config["evaluation_split"] == "validation"
        rows = pl.read_csv(run_dir / "results.csv").sort("seed", "round")
        assert rows["arm"].unique().to_list() == ["random"]
        assert rows.select("seed", "round").is_duplicated().sum() == 0
        assert rows["entity_f1"].is_finite().all()
        assert rows["entity_f1"].is_between(0, 1).all()
        scores = []
        for seed in config["model_seeds"]:
            trajectory = rows.filter(pl.col("seed") == seed)
            assert trajectory["round"].to_list() == list(range(trajectory["total_rounds"][0] + 1))
            x = trajectory["percent_acquired"].to_numpy()
            assert np.all(np.diff(x) > 0)
            auc = np.trapezoid(trajectory["entity_f1"], x) / (x[-1] - x[0])
            scores.append(auc)
            seed_records.append(
                {
                    "config_id": trial["config_id"],
                    "seed": seed,
                    "validation_auc": auc,
                    "final_validation_f1": trajectory["entity_f1"][-1],
                }
            )
        assert np.isclose(np.mean(scores), trial["score"])
        validation_frames.append(rows.with_columns(pl.lit(trial["config_id"]).alias("config_id")))
    validation_results = pl.concat(validation_frames)
    seed_scores = pl.DataFrame(seed_records)
    assert trials["config_id"][0] == selection["config_id"]
    assert np.isclose(trials["score"][0], selection["validation_score"])
    return plan, seed_scores, selection, summary, trials, validation_results


@app.cell(hide_code=True)
def _(mo, selection, summary, trials):
    mo.vstack(
        [
            mo.md(f"""
        # DistilBERT: random-selection hyperparameter search

        **{trials.height}/{len(summary["trials"])} configurations completed.**
        Winner: **{selection["config_id"]}**, with mean validation entity-F1 AUC
        **{selection["validation_score"]:.4f}** across five model seeds.
        The runner-up scored **{trials["score"][1]:.4f}**;
        the difference is **{100 * (trials["score"][0] - trials["score"][1]):.2f} percentage points**.

        All search trials use **random token acquisition only**. The objective is
        validation entity-F1 AUC, not an uncertainty metric. The search holds out 66
        validation sentences from the 328-sentence pool, leaving 262 acquisition
        sentences and the original five bootstrap sentences. The 84 test sentences
        are excluded from this analysis and from hyperparameter selection.

        This notebook only reads saved validation results. It performs no training or
        uncertainty scoring. The run directory also contains a separate, previously
        completed test comparison automatically launched by the search script; that
        comparison is outside this notebook's scope.

        AUC is trapezoidal area under F1 versus acquired-pool percentage, normalized
        by the acquisition span and including round 0. Each configuration has equal
        weight in the search plots; its score is the mean across its five seeds.
        """),
            mo.ui.table(
                trials.head(1), selection=None, label="Winning random-selection hyperparameters"
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, trials):
    tuning_chart = (
        alt.Chart(trials)
        .mark_circle(size=65)
        .encode(
            x=alt.X(
                "learning_rate:Q", scale=alt.Scale(type="log"), title="Learning rate (log scale)"
            ),
            y=alt.Y("score:Q", title="Random validation entity-F1 AUC"),
            color=alt.Color("replay_ratio:N", title="Replay ratio"),
            shape="update_passes:N",
            tooltip=[
                "config_id",
                "score",
                "learning_rate",
                "k",
                "batch_size",
                "update_passes",
                "replay_ratio",
            ],
        )
        .properties(height=330, width="container")
        .interactive()
    )
    mo.vstack(
        [
            mo.md(
                "## Search results\nEach dot is one configuration. Parameter associations "
                "are descriptive because several parameters change together."
            ),
            mo.ui.altair_chart(tuning_chart, chart_selection=False, legend_selection=False),
            mo.ui.table(
                trials, selection=None, label="All configurations, ranked by validation AUC"
            ),
        ]
    )
    return


@app.cell
def _(mo, pl, trials):
    parameter_summary = pl.concat(
        [
            trials.group_by(parameter)
            .agg(
                pl.len().alias("configurations"),
                pl.col("score").mean().alias("mean_auc"),
                pl.col("score").median().alias("median_auc"),
                pl.col("score").max().alias("best_auc"),
            )
            .rename({parameter: "value"})
            .with_columns(pl.col("value").cast(pl.String), pl.lit(parameter).alias("parameter"))
            .select("parameter", "value", "configurations", "mean_auc", "median_auc", "best_auc")
            for parameter in ["k", "update_passes", "batch_size", "replay_ratio", "weight_decay"]
        ]
    ).sort("parameter", "value")
    mo.vstack(
        [
            mo.md(
                "## Hyperparameter associations\nGroup means summarize sampled combinations, "
                "not controlled effects of individual parameters. Learning rates span 1e-6–1e-4."
            ),
            mo.ui.table(parameter_summary, selection=None),
        ]
    )
    return


@app.cell
def _(mo, selection, trials):
    config_selector = mo.ui.dropdown(
        trials["config_id"].to_list(), value=selection["config_id"], label="Configuration"
    )
    metric_selector = mo.ui.dropdown(
        ["entity_f1", "entity_macro_f1", "entity_precision", "entity_recall", "token_accuracy"],
        value="entity_f1",
        label="Validation performance metric",
    )
    mo.hstack([config_selector, metric_selector])
    return config_selector, metric_selector


@app.cell
def _(alt, config_selector, metric_selector, mo, pl, seed_scores, trials, validation_results):
    selected_rows = validation_results.filter(pl.col("config_id") == config_selector.value)
    curves = (
        selected_rows.group_by("round", "percent_acquired")
        .agg(
            pl.col(metric_selector.value).mean().alias("mean"),
            pl.col(metric_selector.value).std().alias("sd"),
        )
        .with_columns(
            (pl.col("mean") - pl.col("sd")).alias("lower"),
            (pl.col("mean") + pl.col("sd")).alias("upper"),
        )
    )
    base = alt.Chart(curves).encode(
        x=alt.X("percent_acquired:Q", title="Acquisition pool labeled (%)")
    )
    curve = (
        (
            base.mark_area(opacity=0.18).encode(y="lower:Q", y2="upper:Q")
            + base.mark_line().encode(
                y=alt.Y("mean:Q", title=metric_selector.value),
                tooltip=["percent_acquired", "mean", "sd"],
            )
        )
        .properties(height=340, width="container")
        .interactive()
    )
    individual = (
        alt.Chart(selected_rows)
        .mark_line()
        .encode(
            x=alt.X("percent_acquired:Q", title="Acquisition pool labeled (%)"),
            y=alt.Y(f"{metric_selector.value}:Q"),
            color="seed:N",
            tooltip=["seed", "percent_acquired", metric_selector.value],
        )
        .properties(height=280, width="container")
        .interactive()
    )
    mo.vstack(
        [
            mo.md(
                "## Random-only validation learning curves\nMean ± one sample standard "
                "deviation across five seeds. Shading is not a confidence interval."
            ),
            mo.ui.table(
                trials.filter(pl.col("config_id") == config_selector.value), selection=None
            ),
            mo.ui.altair_chart(curve, chart_selection=False, legend_selection=False),
            mo.md("### Individual seeds"),
            mo.ui.altair_chart(individual, chart_selection=False, legend_selection=False),
            mo.ui.table(
                seed_scores.filter(pl.col("config_id") == config_selector.value), selection=None
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plan):
    mo.vstack(
        [
            mo.md("""
        ## Interpretation and checks

        The winner is the best of the sampled configurations on this validation split.
        Its score is affected by selection among 100 candidates; a small lead over the
        runner-up does not establish a reliably better configuration. The five seeds
        describe training and acquisition variability on one fixed split.

        Verified every completed export is marked random-only and validation, contains
        only random-arm rows, has no duplicate seed/round records, and contains every
        round for each configured seed. Recomputed every validation AUC and verified
        the saved winning configuration and score.
        """),
            mo.accordion({"Recorded search space": mo.json(plan["search_space"])}),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
