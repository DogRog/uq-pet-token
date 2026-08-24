import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Does early entropy sampling lack diversity?

    This playground checks whether the early performance deficit of entropy-based
    acquisition is accompanied by a less diverse training set than random acquisition.

    The comparison is paired by model seed and acquisition round. Diversity is measured
    without using pool labels: sentence coverage, document coverage, lexical variety,
    and concentration within the most-selected sentence. Entity-label composition is
    shown only as a **post-selection diagnostic**, because labels were revealed after a
    token was selected.
    """)
    return (mo,)


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import polars as pl

    alt.renderers.set_embed_options(scaleFactor=2)
    return Path, alt, json, math, pl


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "results" / "bert_token_uq_20260824_135841_226964"
    config_path = run_dir / "config.json"
    selections_path = run_dir / "selections.json"
    results_path = run_dir / "results.csv"
    return config_path, results_path, run_dir, selections_path


@app.cell
def _(config_path, json, pl, results_path, selections_path):
    config = json.loads(config_path.read_text())
    selections = pl.read_json(selections_path)
    results = pl.read_csv(results_path)
    return config, results, selections


@app.cell
def _(pl):
    def build_selection_metrics(selection_frame):
        keys = ["seed", "arm", "round"]
        normalized = selection_frame.with_columns(
            pl.col("token").str.to_lowercase().alias("token_normalized")
        )

        batch_metrics = normalized.group_by(keys).agg(
            pl.len().alias("n_selected"),
            pl.col("pool_idx").n_unique().alias("unique_sentences"),
            pl.col("document_name").n_unique().alias("unique_documents"),
            pl.col("token_normalized").n_unique().alias("unique_word_forms"),
            (pl.col("label") != "O").cast(pl.Float64).mean().alias("entity_token_share"),
            pl.col("label").n_unique().alias("observed_label_types"),
        )

        sentence_counts = normalized.group_by([*keys, "pool_idx"]).agg(
            pl.len().alias("tokens_in_sentence")
        )
        concentration = sentence_counts.group_by(keys).agg(
            pl.col("tokens_in_sentence").max().alias("max_sentence_tokens")
        )

        first_seen = (
            normalized.group_by(["seed", "arm", "pool_idx"])
            .agg(pl.col("round").min())
            .group_by(keys)
            .agg(pl.len().alias("new_sentences"))
        )
        cumulative = (
            batch_metrics.select(keys)
            .join(first_seen, on=keys, how="left")
            .with_columns(pl.col("new_sentences").fill_null(0))
            .sort(keys)
            .with_columns(
                pl.col("new_sentences")
                .cum_sum()
                .over(["seed", "arm"])
                .alias("cumulative_sentences")
            )
        )

        return (
            batch_metrics.join(concentration, on=keys)
            .join(cumulative, on=keys)
            .with_columns(
                (pl.col("unique_sentences") / pl.col("n_selected")).alias("sentence_coverage"),
                (pl.col("unique_documents") / pl.col("n_selected")).alias("document_coverage"),
                (pl.col("unique_word_forms") / pl.col("n_selected")).alias("lexical_ttr"),
                (pl.col("max_sentence_tokens") / pl.col("n_selected")).alias(
                    "dominant_sentence_share"
                ),
            )
            .sort(keys)
        )

    return (build_selection_metrics,)


@app.cell
def _(build_selection_metrics, pl, results, selections):
    selection_metrics = build_selection_metrics(selections)
    result_metrics = results.filter(pl.col("round") > 0).select(
        "seed",
        "arm",
        "round",
        "percent_acquired",
        "entity_f1",
        "token_accuracy",
    )
    aligned_metrics = selection_metrics.join(
        result_metrics, on=["seed", "arm", "round"], how="inner"
    )

    paired_columns = [
        "sentence_coverage",
        "document_coverage",
        "lexical_ttr",
        "dominant_sentence_share",
        "entity_token_share",
        "cumulative_sentences",
        "entity_f1",
        "token_accuracy",
    ]
    uncertainty_metrics = aligned_metrics.filter(pl.col("arm") == "uncertainty").drop("arm")
    random_metrics = aligned_metrics.filter(pl.col("arm") == "random").drop("arm")
    paired_rounds = uncertainty_metrics.join(
        random_metrics,
        on=["seed", "round"],
        how="inner",
        suffix="_random",
    ).with_columns(
        *[
            (pl.col(column).cast(pl.Float64) - pl.col(f"{column}_random").cast(pl.Float64)).alias(
                f"{column}_delta"
            )
            for column in paired_columns
        ]
    )
    return aligned_metrics, paired_rounds


@app.cell
def _(mo):
    early_percent = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=10,
        label="Early acquisition window (%)",
        show_value=True,
    )
    outcome = mo.ui.dropdown(
        options={"Entity F1": "entity_f1", "Token accuracy": "token_accuracy"},
        value="Entity F1",
        label="Performance outcome",
    )
    diversity_proxy = mo.ui.dropdown(
        options={
            "Sentence coverage per batch": "sentence_coverage",
            "Document coverage per batch": "document_coverage",
            "Lexical type-token ratio": "lexical_ttr",
            "Dominant-sentence share": "dominant_sentence_share",
        },
        value="Sentence coverage per batch",
        label="Diversity proxy",
    )
    mo.hstack([early_percent, outcome, diversity_proxy], widths="equal", gap=2)
    return diversity_proxy, early_percent, outcome


@app.cell
def _(config, early_percent, math, paired_rounds, pl):
    cutoff_round = max(
        1,
        math.floor(early_percent.value / 100 * config["scoreable_pool_tokens"] / config["k"]),
    )
    early_pairs = paired_rounds.filter(pl.col("round") <= cutoff_round)
    per_seed_summary = (
        early_pairs.group_by("seed")
        .agg(
            pl.col("sentence_coverage_delta").mean().alias("sentence coverage Δ"),
            pl.col("document_coverage_delta").mean().alias("document coverage Δ"),
            pl.col("lexical_ttr_delta").mean().alias("lexical TTR Δ"),
            pl.col("dominant_sentence_share_delta").mean().alias("dominant sentence Δ"),
            pl.col("cumulative_sentences_delta")
            .sort_by("round")
            .last()
            .alias("sentences reached Δ"),
            pl.col("entity_token_share_delta").mean().alias("entity-token share Δ"),
            pl.col("entity_f1_delta").mean().alias("entity F1 Δ"),
            pl.col("token_accuracy_delta").mean().alias("token accuracy Δ"),
        )
        .sort("seed")
    )
    overall_summary = (
        per_seed_summary.select(
            pl.exclude("seed").mean(),
        )
        .with_columns(pl.lit("Mean").alias("seed"))
        .select("seed", *[column for column in per_seed_summary.columns if column != "seed"])
    )
    summary_table = pl.concat(
        [per_seed_summary.with_columns(pl.col("seed").cast(pl.String)), overall_summary]
    ).with_columns(
        pl.col("sentence coverage Δ").round(3),
        pl.col("document coverage Δ").round(3),
        pl.col("lexical TTR Δ").round(3),
        pl.col("dominant sentence Δ").round(3),
        pl.col("sentences reached Δ").round(1),
        pl.col("entity-token share Δ").round(3),
        pl.col("entity F1 Δ").round(3),
        pl.col("token accuracy Δ").round(3),
    )
    return (
        cutoff_round,
        early_pairs,
        overall_summary,
        per_seed_summary,
        summary_table,
    )


@app.cell(hide_code=True)
def _(cutoff_round, early_percent, mo, overall_summary, per_seed_summary):
    mean_row = overall_summary.row(0, named=True)
    all_seeds_less_sentence_diverse = (per_seed_summary["sentence coverage Δ"] < 0).all()
    all_seeds_less_lexically_diverse = (per_seed_summary["lexical TTR Δ"] < 0).all()
    all_seeds_lower_f1 = (per_seed_summary["entity F1 Δ"] < 0).all()
    consistent = (
        all_seeds_less_sentence_diverse and all_seeds_less_lexically_diverse and all_seeds_lower_f1
    )
    verdict = (
        "**The run is consistent with the hypothesis.**"
        if consistent
        else "**The evidence is mixed for this acquisition window.**"
    )
    mo.callout(
        mo.md(
            f"""
            {verdict} Through round **{cutoff_round}** (approximately
            **{early_percent.value}%** of the pool), entropy minus random averages:

            - **{mean_row["sentence coverage Δ"] * 100:+.1f} percentage points** in
              unique-sentence coverage per 32-token batch;
            - **{mean_row["lexical TTR Δ"] * 100:+.1f} points** in lexical variety;
            - **{mean_row["dominant sentence Δ"] * 100:+.1f} points** in concentration
              within the batch's most-selected sentence;
            - **{mean_row["sentences reached Δ"]:+.1f} sentences** reached cumulatively;
            - **{mean_row["entity F1 Δ"]:+.3f} entity F1** and
              **{mean_row["token accuracy Δ"]:+.3f} token accuracy**.

            These are descriptive paired differences across five seeds. They show that
            reduced structural and lexical diversity co-occurs with weaker early
            performance, but do not establish that reduced diversity *causes* the deficit.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo, summary_table):
    mo.vstack(
        [
            mo.md("## Paired early-window summary\n\nAll deltas are **entropy − random**."),
            mo.ui.table(summary_table, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(paired_rounds, pl):
    phase_table = (
        paired_rounds.filter(pl.col("percent_acquired") <= 30)
        .with_columns(
            pl.when(pl.col("percent_acquired") <= 5)
            .then(pl.lit("0–5%"))
            .when(pl.col("percent_acquired") <= 10)
            .then(pl.lit("5–10%"))
            .when(pl.col("percent_acquired") <= 20)
            .then(pl.lit("10–20%"))
            .otherwise(pl.lit("20–30%"))
            .alias("phase"),
            pl.when(pl.col("percent_acquired") <= 5)
            .then(pl.lit(1))
            .when(pl.col("percent_acquired") <= 10)
            .then(pl.lit(2))
            .when(pl.col("percent_acquired") <= 20)
            .then(pl.lit(3))
            .otherwise(pl.lit(4))
            .alias("phase_order"),
        )
        .group_by(["phase", "phase_order"])
        .agg(
            (100 * pl.col("sentence_coverage_delta").mean())
            .round(1)
            .alias("sentence coverage Δ (pp)"),
            (100 * pl.col("lexical_ttr_delta").mean()).round(1).alias("lexical TTR Δ (pp)"),
            (100 * pl.col("dominant_sentence_share_delta").mean())
            .round(1)
            .alias("dominant sentence Δ (pp)"),
            (100 * pl.col("entity_token_share_delta").mean())
            .round(1)
            .alias("entity-token share Δ (pp)"),
            pl.col("entity_f1_delta").mean().round(3).alias("entity F1 Δ"),
            pl.col("token_accuracy_delta").mean().round(3).alias("token accuracy Δ"),
        )
        .sort("phase_order")
        .drop("phase_order")
    )
    return (phase_table,)


@app.cell(hide_code=True)
def _(mo, phase_table):
    mo.vstack(
        [
            mo.md(
                "## Does the relationship change as the model matures?\n\n"
                "Phase means use non-overlapping acquisition windows. Structural "
                "sentence diversity remains lower for entropy sampling, while the "
                "performance deficit fades and then reverses. This temporal pattern is "
                "consistent with diversity mattering more at the beginning, though it "
                "still does not identify a causal mechanism."
            ),
            mo.ui.table(phase_table, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(aligned_metrics, alt, diversity_proxy, early_percent, outcome, pl):
    proxy_labels = {
        "sentence_coverage": "Unique-sentence share",
        "document_coverage": "Unique-document share",
        "lexical_ttr": "Unique-word-form share",
        "dominant_sentence_share": "Dominant-sentence share",
    }
    outcome_labels = {"entity_f1": "Entity F1", "token_accuracy": "Token accuracy"}
    selected_proxy = diversity_proxy.value
    selected_outcome = outcome.value

    chart_summary = (
        aligned_metrics.group_by(["arm", "round", "percent_acquired"])
        .agg(
            pl.col(selected_proxy).mean().alias("diversity_mean"),
            pl.col(selected_outcome).mean().alias("outcome_mean"),
        )
        .sort(["arm", "round"])
    )
    colors = alt.Color(
        "arm:N",
        title=None,
        scale=alt.Scale(
            domain=["random", "uncertainty"],
            range=["#4C78A8", "#F58518"],
        ),
    )
    shared_x = alt.X(
        "percent_acquired:Q",
        title="Scoreable pool acquired (%)",
        scale=alt.Scale(domain=[0, 30]),
    )
    cutoff_rule = (
        alt.Chart(pl.DataFrame({"cutoff": [early_percent.value]}))
        .mark_rule(color="#888", strokeDash=[5, 4])
        .encode(x="cutoff:Q")
    )
    diversity_chart = (
        alt.Chart(chart_summary.filter(pl.col("percent_acquired") <= 30))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=shared_x,
            y=alt.Y("diversity_mean:Q", title=proxy_labels[selected_proxy]),
            color=colors,
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("percent_acquired:Q", title="Acquired (%)", format=".2f"),
                alt.Tooltip("diversity_mean:Q", title="Mean", format=".3f"),
            ],
        )
        .properties(title="Diversity proxy across seeds", width=550, height=320)
    )
    outcome_chart = (
        alt.Chart(chart_summary.filter(pl.col("percent_acquired") <= 30))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=shared_x,
            y=alt.Y("outcome_mean:Q", title=outcome_labels[selected_outcome]),
            color=colors,
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("percent_acquired:Q", title="Acquired (%)", format=".2f"),
                alt.Tooltip("outcome_mean:Q", title="Mean", format=".3f"),
            ],
        )
        .properties(title="Performance across seeds", width=550, height=320)
    )
    alt.hconcat(
        alt.layer(diversity_chart, cutoff_rule),
        alt.layer(outcome_chart, cutoff_rule),
        spacing=30,
    ).resolve_scale(color="shared")
    return


@app.cell
def _(alt, diversity_proxy, early_pairs, mo, outcome, pl):
    proxy_delta = f"{diversity_proxy.value}_delta"
    outcome_delta = f"{outcome.value}_delta"
    correlation_by_seed = (
        early_pairs.group_by("seed")
        .agg(pl.corr(proxy_delta, outcome_delta).alias("Pearson correlation"))
        .sort("seed")
        .with_columns(pl.col("Pearson correlation").round(3))
    )
    association_chart = (
        alt.Chart(early_pairs)
        .mark_circle(size=65, opacity=0.7)
        .encode(
            x=alt.X(f"{proxy_delta}:Q", title="Diversity proxy Δ (entropy − random)"),
            y=alt.Y(f"{outcome_delta}:Q", title="Performance Δ (entropy − random)"),
            color=alt.Color("seed:N", title="Seed"),
            tooltip=[
                "seed:N",
                "round:Q",
                alt.Tooltip(f"{proxy_delta}:Q", title="Diversity Δ", format=".3f"),
                alt.Tooltip(f"{outcome_delta}:Q", title="Performance Δ", format=".3f"),
            ],
        )
        .properties(title="Round-level association in the early window", width=700, height=380)
    )
    mo.vstack(
        [
            mo.md(
                "## Association check\n\n"
                "A positive relationship is compatible with the hypothesis, but rounds "
                "within a seed are autocorrelated, so this is exploratory rather than an "
                "independent-sample significance test."
            ),
            association_chart,
            mo.ui.table(correlation_by_seed, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(aligned_metrics, alt, early_percent, pl):
    label_diagnostic = (
        aligned_metrics.filter(pl.col("percent_acquired") <= early_percent.value)
        .group_by("arm")
        .agg(
            pl.col("entity_token_share").mean().alias("entity_token_share"),
            pl.col("observed_label_types").mean().alias("mean_label_types_per_batch"),
        )
        .sort("arm")
    )
    label_chart = (
        alt.Chart(label_diagnostic)
        .mark_bar()
        .encode(
            x=alt.X("arm:N", title=None),
            y=alt.Y("entity_token_share:Q", title="Selected entity-token share"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=["random", "uncertainty"], range=["#4C78A8", "#F58518"]),
            ),
            tooltip=[
                "arm:N",
                alt.Tooltip("entity_token_share:Q", format=".3f"),
                alt.Tooltip("mean_label_types_per_batch:Q", format=".2f"),
            ],
        )
        .properties(width=500, height=280)
    )
    label_chart
    return


@app.cell(hide_code=True)
def _(mo, run_dir):
    mo.md(f"""
    ## Interpretation and next test

    The strongest defensible statement from this log is: **early entropy acquisition
    is less diverse by several observable proxies, and this coincides with lower early
    test performance in every seed.** The log cannot measure semantic diversity of the
    full sentence contexts, and an observational comparison cannot isolate diversity
    from other properties of high-entropy tokens.

    A causal follow-up would keep uncertainty high while explicitly enforcing
    diversity—for example, select at most one token per sentence in the first 5–10% of
    acquisition, or select a diverse subset from the top entropy candidates. Compare
    that arm against plain entropy and random selection from the same bootstrap state.

    Data source: `{run_dir}`
    """)
    return


if __name__ == "__main__":
    app.run()
