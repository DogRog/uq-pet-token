import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Gumbel noise and word form caps

    Did Gumbel noise or a word form cap (ADR 0009) improve the uncertainty arm over plain
    top-K selection? Each variant run is paired with a top-K run of identical settings
    found anywhere under `results/`. Both share one bootstrap and one random arm per seed,
    so the variant-minus-top-K gap isolates the selection change.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Gap AUC** is a difference in entity F1, normalized over the acquired-pool
      percentage and computed per seed; **final gap** compares the last round. Both are
      in percentage points (pp).
    - The headline is **variant − top-K**. Top-K − random and variant − random show
      whether either beats random at these settings.
    - The interval and p-value come from a one-sample t-test over seeds. With five
      seeds, treat them as descriptive.
    - Runs pair only when every scientific setting, the UQ metric, and the precision
      used agree. The random-arm check confirms the pair shares its random trajectory.
    - Nothing is downloaded or trained.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from utils.variant_analysis import (
        gap_tests,
        load_runs,
        matched_variants,
        paired_curves,
        round_jitter,
        seed_gaps,
        selection_redundancy,
    )

    repo_root = Path(__file__).resolve().parents[1]
    return (
        alt,
        gap_tests,
        load_runs,
        matched_variants,
        mo,
        paired_curves,
        pl,
        repo_root,
        round_jitter,
        seed_gaps,
        selection_redundancy,
    )


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh saved results")
    refresh
    return (refresh,)


@app.cell
def _(load_runs, matched_variants, mo, refresh, repo_root):
    refresh.value
    runs = load_runs(repo_root / "results")
    pairs = matched_variants(runs)
    mo.stop(
        pairs.is_empty(),
        mo.md(
            "No variant run has a matching top-K run yet. Run the same settings with "
            "`gumbel_noise` off (or `scripts/run_stochastic_uq_all_models.sh` beside "
            "`results/best_uq/`), then refresh."
        ),
    )
    pair_options = {
        f"{row['checkpoint']} · {row['uq_metric']} · {row['variant']} · {row['run']}": row
        for row in pairs.to_dicts()
    }
    pair_selector = mo.ui.dropdown(
        pair_options, value=next(iter(pair_options)), label="Variant run", full_width=True
    )
    mo.vstack(
        [
            mo.md(f"## Matched runs\n**{pairs.height}** variant run(s) have a top-K partner."),
            pair_selector,
            mo.accordion(
                {
                    "All matched pairs": mo.ui.table(pairs, selection=None),
                    "All runs found": mo.ui.table(runs.drop("settings"), selection=None),
                }
            ),
        ]
    )
    return (pair_selector,)


@app.cell
def _(mo, paired_curves, pair_selector, pl, repo_root):
    pair = pair_selector.value
    baseline_run = repo_root / "results" / pair["baseline_run"]
    variant_run = repo_root / "results" / pair["run"]
    variant = pair["variant"]
    curves = paired_curves(baseline_run, variant_run)
    random_mismatch = (curves["random"] - curves["variant_random"]).abs().max()
    random_check = (
        mo.md(f"Random arms agree in every round and seed (largest difference {random_mismatch}).")
        if random_mismatch == 0
        else mo.callout(
            mo.md(
                f"Random arms differ by up to **{random_mismatch:.4f}** F1 although they "
                "draw identical words: training is not bitwise reproducible on this device. "
                "The **random_rerun** rows measure that noise; a variant effect smaller than "
                "them is not distinguishable from rerunning top-K."
            ),
            kind="warn",
        )
    )
    seeds = curves["seed"].n_unique()
    rounds = curves.filter(pl.col("seed") == curves["seed"][0]).height - 1
    return baseline_run, curves, random_check, rounds, seeds, variant, variant_run


@app.cell
def _(curves, gap_tests, mo, pl, random_check, rounds, seed_gaps, seeds, variant):
    gaps = seed_gaps(curves)
    tests = gap_tests(gaps.drop("seed"))
    headline = tests.filter(pl.col("gap") == "variant_minus_top_k_auc").row(0, named=True)
    rerun_spread = 100 * gaps["random_rerun_auc"].abs().max()
    verdict = (
        "higher"
        if headline["95% CI lower (pp)"] > 0
        else "lower"
        if headline["95% CI upper (pp)"] < 0
        else "not clearly different"
    )
    mo.vstack(
        [
            mo.md(f"""
    ## Did {variant} help?

    Over **{seeds} seeds** and **{rounds} rounds**, {variant} minus top-K has a gap AUC of
    **{headline["mean (pp)"]:+.2f} pp** (95% CI {headline["95% CI lower (pp)"]:+.2f} to
    {headline["95% CI upper (pp)"]:+.2f}; {headline["seeds above 0"]} seeds above zero):
    entity F1 across the curve is **{verdict}** than with top-K selection. Rerunning the
    identical random arm moves gap AUC by up to **{rerun_spread:.2f} pp** per seed.
    """),
            mo.ui.table(
                tests.with_columns(pl.col(pl.Float64).round(4)), selection=None, label="Gaps"
            ),
            random_check,
            mo.accordion(
                {
                    "Per-seed gaps": mo.ui.table(
                        gaps.with_columns(pl.col(pl.Float64).round(4)), selection=None
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(alt, curves, mo, pl, variant):
    arm_names = {"random": "random", "top_k": "top-K", "variant": variant}
    long_curves = curves.unpivot(
        on=list(arm_names),
        index=["seed", "round", "percent_acquired"],
        variable_name="arm",
        value_name="entity_f1",
    ).with_columns(pl.col("arm").replace(arm_names))
    mean_curves = long_curves.group_by("arm", "percent_acquired").agg(
        pl.col("entity_f1").mean().alias("mean"),
        pl.col("entity_f1").std().alias("sd"),
    )
    band = (
        alt.Chart(
            mean_curves.with_columns(
                lower=pl.col("mean") - pl.col("sd"), upper=pl.col("mean") + pl.col("sd")
            )
        )
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("percent_acquired:Q", title="Acquired pool (%)"),
            y=alt.Y("lower:Q", title="Entity F1"),
            y2="upper:Q",
            color=alt.Color("arm:N", title=None),
        )
    )
    lines = (
        alt.Chart(mean_curves)
        .mark_line()
        .encode(
            x="percent_acquired:Q",
            y=alt.Y("mean:Q", title="Entity F1", scale=alt.Scale(zero=False)),
            color="arm:N",
            tooltip=[
                "arm",
                alt.Tooltip("percent_acquired:Q", format=".1f"),
                alt.Tooltip("mean:Q", format=".4f"),
            ],
        )
    )

    differences = curves.with_columns(gap=pl.col("variant") - pl.col("top_k"))
    seed_lines = (
        alt.Chart(differences)
        .mark_line(opacity=0.3, strokeWidth=1)
        .encode(
            x=alt.X("percent_acquired:Q", title="Acquired pool (%)"),
            y=alt.Y("gap:Q", title=f"{variant} − top-K entity F1"),
            detail="seed:N",
            color=alt.value("gray"),
        )
    )
    mean_gap = (
        alt.Chart(differences.group_by("percent_acquired").agg(pl.col("gap").mean()))
        .mark_line(strokeWidth=2.5)
        .encode(x="percent_acquired:Q", y="gap:Q")
    )
    zero = alt.Chart(differences).mark_rule(strokeDash=[4, 4], color="black").encode(y=alt.datum(0))
    mo.vstack(
        [
            mo.md("## Learning curves\nSeed means with ±1 SD bands."),
            mo.ui.altair_chart((band + lines).properties(width=760, height=320)),
            mo.md(f"## {variant} minus top-K\nThin lines are seeds; the thick line is their mean."),
            mo.ui.altair_chart((seed_lines + mean_gap + zero).properties(width=760, height=260)),
        ]
    )
    return


@app.cell
def _(alt, baseline_run, mo, pl, selection_redundancy, variant, variant_run):
    redundancy = pl.concat(
        [
            selection_redundancy(baseline_run, "top-K"),
            selection_redundancy(variant_run, variant).filter(pl.col("selection") != "random"),
        ]
    )
    measures = {
        "repeated_share": "Repeated word forms in the round",
        "o_share": "O-labelled share",
        "words_per_sentence": "Words per sentence",
    }
    redundancy_means = (
        redundancy.unpivot(
            on=list(measures), index=["selection", "seed", "round"], variable_name="measure"
        )
        .group_by("selection", "measure", "round")
        .agg(pl.col("value").mean())
        .with_columns(pl.col("measure").replace(measures))
    )
    redundancy_chart = (
        alt.Chart(redundancy_means)
        .mark_line()
        .encode(
            x=alt.X("round:Q", title="Round"),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("selection:N", title=None),
        )
        .properties(width=360, height=220)
        .facet(column=alt.Column("measure:N", title=None))
        .resolve_scale(y="independent")
    )
    overall = (
        redundancy.group_by("selection")
        .agg(pl.col(list(measures)).mean().round(3))
        .sort("selection")
    )
    mo.vstack(
        [
            mo.md(
                "## Did selection get less redundant?\nSeed means per round. O labels are "
                "read from the saved selections, after acquisition."
            ),
            mo.ui.altair_chart(redundancy_chart),
            mo.ui.table(overall, selection=None, label="Mean over all rounds"),
        ]
    )
    return


@app.cell
def _(curves, mo, pl, round_jitter, variant):
    jitter = round_jitter(curves).with_columns(
        pl.col("arm").replace({"top_k": "top-K", "variant": variant}),
        pl.col(pl.Float64).round(3),
    )
    mo.vstack(
        [
            mo.md("## Round-to-round stability\nChange in entity F1 between consecutive rounds."),
            mo.ui.table(jitter, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
