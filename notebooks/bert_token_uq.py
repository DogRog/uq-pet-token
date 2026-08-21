import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # BERT token-level active learning

    Compare online updates from the **K most uncertain words** with the same number
    of **random words**. Both arms start from one seed-trained model, retain their
    weights and optimizer state, and use masked token loss with limited replay.
    """)
    return (mo,)


@app.cell
def _():
    import altair as alt
    import polars as pl

    from uq_pet.active_learning import run_active_learning, write_run
    from uq_pet.pet_data import download_pet_ner, load_pet_splits
    from uq_pet.token_model import get_device

    alt.renderers.set_embed_options(scaleFactor=3)
    return (
        alt,
        download_pet_ner,
        get_device,
        load_pet_splits,
        pl,
        run_active_learning,
        write_run,
    )


@app.cell
def _(alt, pl):
    def make_learning_chart(result_records, seed):
        curve = (
            pl.DataFrame(result_records).filter(pl.col("seed") == seed).sort(["arm", "n_acquired"])
        )
        arm_color = alt.Color(
            "arm:N",
            title=None,
            sort=["random", "uncertainty"],
            scale=alt.Scale(
                domain=["random", "uncertainty"],
                range=["#4C78A8", "#F58518"],
            ),
            legend=alt.Legend(orient="top"),
        )
        shared_x = alt.X(
            "n_acquired:Q",
            title="Acquired pool tokens",
            axis=alt.Axis(format="d"),
        )
        entity_chart = (
            alt.Chart(curve)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=shared_x,
                y=alt.Y("entity_f1:Q", title="F1", scale=alt.Scale(zero=False)),
                color=arm_color,
                order=alt.Order("n_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                    alt.Tooltip("entity_f1:Q", title="Entity F1", format=".3f"),
                ],
            )
            .properties(title="Entity F1", width=500, height=320)
        )
        accuracy_chart = (
            alt.Chart(curve)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=shared_x,
                y=alt.Y("token_accuracy:Q", title="Accuracy", scale=alt.Scale(zero=False)),
                color=arm_color,
                order=alt.Order("n_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                    alt.Tooltip("token_accuracy:Q", title="Token accuracy", format=".3f"),
                ],
            )
            .properties(title="Token accuracy", width=500, height=320)
        )
        return alt.hconcat(entity_chart, accuracy_chart, spacing=35).resolve_scale(color="shared")

    def make_variance_chart(results_frame):
        summary = (
            results_frame.group_by(["arm", "n_acquired"])
            .agg(
                pl.col("entity_f1").mean().alias("entity_f1_mean"),
                pl.col("entity_f1").std().fill_null(0.0).alias("entity_f1_std"),
                pl.col("token_accuracy").mean().alias("token_accuracy_mean"),
                pl.col("token_accuracy").std().fill_null(0.0).alias("token_accuracy_std"),
            )
            .with_columns(
                (pl.col("entity_f1_mean") - pl.col("entity_f1_std"))
                .clip(0.0, 1.0)
                .alias("entity_f1_lower"),
                (pl.col("entity_f1_mean") + pl.col("entity_f1_std"))
                .clip(0.0, 1.0)
                .alias("entity_f1_upper"),
                (pl.col("token_accuracy_mean") - pl.col("token_accuracy_std"))
                .clip(0.0, 1.0)
                .alias("token_accuracy_lower"),
                (pl.col("token_accuracy_mean") + pl.col("token_accuracy_std"))
                .clip(0.0, 1.0)
                .alias("token_accuracy_upper"),
            )
            .sort(["arm", "n_acquired"])
        )
        arm_color = alt.Color(
            "arm:N",
            title=None,
            sort=["random", "uncertainty"],
            scale=alt.Scale(
                domain=["random", "uncertainty"],
                range=["#4C78A8", "#F58518"],
            ),
            legend=alt.Legend(orient="top"),
        )
        shared_x = alt.X(
            "n_acquired:Q",
            title="Acquired pool tokens",
            axis=alt.Axis(format="d"),
        )

        def metric_chart(mean_field, std_field, lower_field, upper_field, title, y_title):
            band = (
                alt.Chart(summary)
                .mark_area(opacity=0.18)
                .encode(
                    x=shared_x,
                    y=alt.Y(f"{lower_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                    y2=alt.Y2(f"{upper_field}:Q"),
                    color=arm_color,
                )
            )
            mean_line = (
                alt.Chart(summary)
                .mark_line(strokeWidth=3)
                .encode(
                    x=shared_x,
                    y=alt.Y(f"{mean_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                    color=arm_color,
                    tooltip=[
                        alt.Tooltip("arm:N", title="Arm"),
                        alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                        alt.Tooltip(f"{mean_field}:Q", title="Mean", format=".3f"),
                        alt.Tooltip(f"{std_field}:Q", title="Std. dev.", format=".3f"),
                    ],
                )
            )
            return alt.layer(band, mean_line).properties(title=title, width=500, height=320)

        entity_chart = metric_chart(
            "entity_f1_mean",
            "entity_f1_std",
            "entity_f1_lower",
            "entity_f1_upper",
            "Entity F1",
            "F1",
        )
        accuracy_chart = metric_chart(
            "token_accuracy_mean",
            "token_accuracy_std",
            "token_accuracy_lower",
            "token_accuracy_upper",
            "Token accuracy",
            "Accuracy",
        )
        return alt.hconcat(entity_chart, accuracy_chart, spacing=35).resolve_scale(color="shared")

    return make_learning_chart, make_variance_chart


@app.cell
def _(mo):
    checkpoint_input = mo.ui.text(
        value="distilbert-base-cased",
        label="Hugging Face checkpoint",
        full_width=True,
    )
    seeds_input = mo.ui.text(value="0", label="Model seeds (comma-separated)")
    uq_metric_input = mo.ui.dropdown(
        options={
            "Predictive entropy": "entropy",
            "Least confidence": "least_confidence",
            "Margin": "margin",
        },
        value="Predictive entropy",
        label="UQ metric",
    )
    k_input = mo.ui.number(start=1, stop=1000, step=1, value=32, label="New tokens per round")
    rounds_input = mo.ui.number(start=1, stop=100, step=1, value=5, label="Rounds")
    bootstrap_input = mo.ui.number(
        start=0,
        stop=100,
        step=1,
        value=20,
        label="Bootstrap epochs",
    )
    passes_input = mo.ui.number(
        start=1,
        stop=20,
        step=1,
        value=1,
        label="Update passes per round",
    )
    replay_input = mo.ui.number(
        start=0,
        stop=4,
        step=0.25,
        value=1.0,
        label="Replay ratio",
    )
    learning_rate_input = mo.ui.number(
        start=0.000001,
        stop=0.001,
        step=0.000001,
        value=0.00005,
        label="Learning rate",
    )
    batch_size_input = mo.ui.number(
        start=1,
        stop=64,
        step=1,
        value=8,
        label="Training batch size",
    )
    run_button = mo.ui.run_button(label="Run experiment")

    controls = mo.vstack(
        [
            checkpoint_input,
            mo.hstack([seeds_input, uq_metric_input, k_input, rounds_input]),
            mo.hstack([bootstrap_input, passes_input, replay_input]),
            mo.hstack([learning_rate_input, batch_size_input]),
            run_button,
        ]
    )
    controls
    return (
        batch_size_input,
        bootstrap_input,
        checkpoint_input,
        k_input,
        learning_rate_input,
        passes_input,
        replay_input,
        rounds_input,
        run_button,
        seeds_input,
        uq_metric_input,
    )


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(
    batch_size_input,
    bootstrap_input,
    checkpoint_input,
    download_pet_ner,
    get_device,
    is_script_mode,
    k_input,
    learning_rate_input,
    load_pet_splits,
    make_learning_chart,
    mo,
    passes_input,
    pl,
    replay_input,
    rounds_input,
    run_active_learning,
    run_button,
    seeds_input,
    uq_metric_input,
    write_run,
):
    model_seeds = [int(value.strip()) for value in seeds_input.value.split(",") if value.strip()]
    config = {
        "checkpoint": checkpoint_input.value,
        "model_seeds": model_seeds,
        "uq_metric": uq_metric_input.value,
        "k": int(k_input.value),
        "rounds": int(rounds_input.value),
        "bootstrap_epochs": int(bootstrap_input.value),
        "update_passes": int(passes_input.value),
        "replay_ratio": float(replay_input.value),
        "learning_rate": float(learning_rate_input.value),
        "weight_decay": 0.01,
        "batch_size": int(batch_size_input.value),
        "score_batch_size": 32,
        "max_length": 256,
    }

    if is_script_mode:
        # Script mode validates notebook execution without downloading model weights.
        result_records = [
            {
                "seed": 0,
                "arm": arm,
                "round": round_idx,
                "n_acquired": round_idx * config["k"],
                "entity_f1": 0.10 + round_idx * (0.03 if arm == "uncertainty" else 0.02),
                "entity_precision": 0.12,
                "entity_recall": 0.10,
                "token_accuracy": 0.70 + round_idx * 0.01,
                "train_loss": 1.0 / (round_idx + 1),
                "n_new": 0 if round_idx == 0 else config["k"],
                "n_replay": 0 if round_idx == 0 else config["k"],
            }
            for arm in ("uncertainty", "random")
            for round_idx in range(3)
        ]
        selection_records = [
            {
                "seed": 0,
                "arm": arm,
                "round": 1,
                "pool_idx": idx,
                "word_idx": 0,
                "document_name": "script-demo",
                "sentence_id": idx,
                "token": token,
                "label": label,
                "uq_metric": config["uq_metric"] if arm == "uncertainty" else None,
                "uq_score": 0.8 if arm == "uncertainty" else None,
            }
            for arm in ("uncertainty", "random")
            for idx, (token, label) in enumerate((("check", "B-Activity"), ("form", "O")))
        ]
        dataset_summary = {
            "mode": "script demo",
            "seed_sentences": 5,
            "pool_sentences": 328,
            "test_sentences": 84,
            "device": "not loaded",
        }
        run_dir = None
    else:
        mo.stop(
            not run_button.value, mo.md("Set the experiment controls and press **Run experiment**.")
        )
        data_path = download_pet_ner()
        seed_examples, pool_inputs, pool_gold, test_examples = load_pet_splits(data_path)
        device = get_device()
        dataset_summary = {
            "mode": "experiment",
            "seed_sentences": len(seed_examples),
            "pool_sentences": len(pool_inputs),
            "pool_tokens": len(pool_gold),
            "test_sentences": len(test_examples),
            "device": str(device),
        }

        def update_live_chart(progress_records):
            latest = progress_records[-1]
            seed_position = config["model_seeds"].index(latest["seed"]) + 1
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md(
                            f"### Live results — seed {latest['seed']} "
                            f"({seed_position} of {len(config['model_seeds'])}), "
                            f"round {latest['round']} of {config['rounds']}"
                        ),
                        make_learning_chart(progress_records, latest["seed"]),
                    ]
                )
            )

        mo.output.replace(mo.md("Training the seed model; the chart will appear after round 0."))
        result_records, selection_records = run_active_learning(
            seed_examples,
            pool_inputs,
            pool_gold,
            test_examples,
            **config,
            device=device,
            progress_callback=update_live_chart,
        )
        run_dir = write_run(config, result_records, selection_records)

    results_df = pl.DataFrame(result_records)
    selections_df = pl.DataFrame(selection_records)
    seed_graphs = []
    for model_seed in config["model_seeds"]:
        seed_graphs.extend(
            [
                mo.md(f"## Seed {model_seed}"),
                make_learning_chart(result_records, model_seed),
            ]
        )
    completed_seed_graphs = mo.vstack(seed_graphs)
    completed_seed_graphs
    return config, dataset_summary, results_df, run_dir, selections_df


@app.cell(hide_code=True)
def _(config, dataset_summary, mo, run_dir):
    output_location = "Script-mode demo: no files written" if run_dir is None else str(run_dir)
    seed_description = ", ".join(str(seed) for seed in config["model_seeds"])
    mo.vstack(
        [
            mo.md(f"""
            ## Run description

            **{config["checkpoint"]}** was bootstrapped for
            **{config["bootstrap_epochs"]} epochs** with model seed(s)
            **{seed_description}**. Each of the **{config["rounds"]} rounds** acquired
            **{config["k"]} new tokens per arm**, trained for
            **{config["update_passes"]} pass(es)**, and replayed
            **{config["replay_ratio"]:g}× K** older labels. The uncertainty arm used
            **{config["uq_metric"].replace("_", " ")}** scoring.
            """),
            mo.ui.table([dataset_summary], selection=None),
            mo.md(f"**Output:** `{output_location}`"),
            mo.accordion({"All settings": mo.ui.table([config], selection=None)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(make_variance_chart, mo, results_df):
    mo.vstack(
        [
            mo.md("""
            ## Across-seed variability

            Lines show the mean across seeds; shaded ranges show ±1 standard deviation.
            """),
            make_variance_chart(results_df),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, results_df):
    mo.vstack([mo.md("## Round results"), mo.ui.table(results_df, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo, pl, results_df):
    paired = results_df.pivot(
        on="arm",
        index=["seed", "round", "n_acquired"],
        values="entity_f1",
    ).with_columns((pl.col("uncertainty") - pl.col("random")).alias("uncertainty_minus_random"))
    gap_table = (
        paired.group_by(["round", "n_acquired"])
        .agg(
            pl.col("uncertainty_minus_random").mean().alias("mean"),
            pl.col("uncertainty_minus_random").std().alias("std"),
        )
        .sort(["round", "n_acquired"])
    )
    mo.vstack([mo.md("## Entity F1 gap"), mo.ui.table(gap_table, selection=None)])
    return


@app.cell
def _(alt, pl, selections_df):
    label_counts = selections_df.group_by(["arm", "label"]).agg(pl.len().alias("count"))
    label_chart = (
        alt.Chart(label_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Acquired tokens"),
            y=alt.Y("label:N", title="Gold label", sort="-x"),
            yOffset=alt.YOffset("arm:N", sort=["random", "uncertainty"]),
            color=alt.Color(
                "arm:N",
                title=None,
                sort=["random", "uncertainty"],
                scale=alt.Scale(
                    domain=["random", "uncertainty"],
                    range=["#4C78A8", "#F58518"],
                ),
            ),
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("label:N", title="Gold label"),
                alt.Tooltip("count:Q", title="Count", format=".0f"),
            ],
        )
        .properties(
            title="Labels revealed after acquisition",
            width=850,
            height=380,
        )
    )
    label_chart
    return


@app.cell(hide_code=True)
def _(mo, selections_df):
    mo.vstack(
        [
            mo.md("## Selection log"),
            mo.ui.table(selections_df, selection=None, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
