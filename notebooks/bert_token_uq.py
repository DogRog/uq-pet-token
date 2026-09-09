import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv(".env")

    import os
    from pathlib import Path

    import altair as alt
    import polars as pl
    import wandb
    from wigglystuff import EnvConfig

    from charts import (
        make_learning_chart,
        make_tag_coverage_chart,
        make_variance_chart,
        make_wandb_comparison_media,
    )
    from uq_pet.active_learning import acquisition_schedule
    from uq_pet.experiment import (
        DEFAULT_CHECKPOINTS,
        ExperimentConfig,
        configure_wandb_metrics,
        execute_experiment,
        make_wandb_evaluation_log,
        require_wandb_credentials,
    )
    from uq_pet.pet_data import RESULTS_DIR
    from uq_pet.token_model import UQ_METRICS

    alt.renderers.set_embed_options(scaleFactor=3)
    return (
        DEFAULT_CHECKPOINTS,
        EnvConfig,
        ExperimentConfig,
        Path,
        RESULTS_DIR,
        UQ_METRICS,
        acquisition_schedule,
        configure_wandb_metrics,
        execute_experiment,
        make_learning_chart,
        make_tag_coverage_chart,
        make_variance_chart,
        make_wandb_comparison_media,
        make_wandb_evaluation_log,
        mo,
        os,
        pl,
        require_wandb_credentials,
        wandb,
    )


@app.cell
def _(ExperimentConfig, acquisition_schedule):
    default_config = ExperimentConfig().model_dump()

    def make_demo_run(config):
        scoreable_tokens = 99
        rounds, token_budget = acquisition_schedule(
            scoreable_tokens, config["k"], config["max_pool_percent"]
        )
        run_config = {
            **config,
            "scoreable_pool_tokens": scoreable_tokens,
            "token_budget": token_budget,
            "rounds": rounds,
            "effective_pool_percent": 100 * token_budget / scoreable_tokens,
        }
        results = [
            {
                "seed": model_seed,
                "arm": arm,
                "round": round_idx,
                "total_rounds": rounds,
                "n_acquired": min(round_idx * config["k"], token_budget),
                "percent_acquired": 100
                * min(round_idx * config["k"], token_budget)
                / scoreable_tokens,
                "scoreable_pool_tokens": scoreable_tokens,
                "token_budget": token_budget,
                "entity_f1": 0.10
                + model_seed * 0.001
                + round_idx * (0.03 if arm == "uncertainty" else 0.02),
                "entity_macro_f1": 0.08
                + model_seed * 0.001
                + round_idx * (0.025 if arm == "uncertainty" else 0.015),
                "entity_precision": 0.12,
                "entity_recall": 0.10,
                "token_accuracy": 0.70 + round_idx * 0.01,
                "train_loss": 1.0 / (round_idx + 1),
                "n_new": 0
                if round_idx == 0
                else min(config["k"], token_budget - (round_idx - 1) * config["k"]),
                "n_replay": 0
                if round_idx == 0
                else round(
                    min(config["k"], token_budget - (round_idx - 1) * config["k"])
                    * config["replay_ratio"]
                ),
            }
            for model_seed in config["model_seeds"]
            for arm in ("uncertainty", "random")
            for round_idx in range(rounds + 1)
        ]
        selections = [
            {
                "seed": model_seed,
                "arm": arm,
                "round": idx // config["k"] + 1,
                "pool_idx": idx,
                "word_idx": 0,
                "document_name": "script-demo",
                "sentence_id": idx,
                "token": "check" if idx % 2 == 0 else "form",
                "label": "B-Activity" if idx % 2 == 0 else "O",
                "uq_metric": config["uq_metric"] if arm == "uncertainty" else None,
                "uq_score": 0.8 if arm == "uncertainty" else None,
            }
            for model_seed in config["model_seeds"]
            for arm in ("uncertainty", "random")
            for idx in range(token_budget)
        ]
        summary = {
            "mode": "script demo",
            "seed_sentences": 5,
            "pool_sentences": 328,
            "scoreable_pool_tokens": scoreable_tokens,
            "acquisition_budget_tokens": token_budget,
            "acquisition_rounds": rounds,
            "effective_pool_percent": run_config["effective_pool_percent"],
            "test_sentences": 84,
            "device": "not loaded",
        }
        return run_config, summary, results, selections

    return default_config, make_demo_run


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    cli_arguments = dict(mo.cli_args()) if is_script_mode else {}
    is_batch_mode = "config-json" in cli_arguments
    return cli_arguments, is_batch_mode, is_script_mode


@app.cell
def _(EnvConfig, is_script_mode, mo, wandb):
    if is_script_mode:
        env_config_widget = None
        env_config_display = None
    else:
        env_config_widget = EnvConfig(
            {"WANDB_API_KEY": lambda key: wandb.login(key=key, verify=True)}
        )
        env_config = mo.ui.anywidget(env_config_widget)
        env_config_display = mo.accordion({"Environment keys (only needed for W&B)": env_config})
    env_config_display
    return (env_config_widget,)


@app.cell
def _(DEFAULT_CHECKPOINTS, UQ_METRICS, default_config, mo):
    def numeric_control(name, label, *, start, stop=None, step=1):
        return mo.ui.number(
            start=start, stop=stop, step=step, value=default_config[name], label=label
        )

    params_form = (
        mo.md("""
        ## Experiment parameters

        {checkpoint}

        {model_seeds} {seed_workers}

        {uq_metric} {k} {max_pool_percent}

        {bootstrap_epochs} {update_passes} {replay_ratio}

        {learning_rate} {weight_decay} {batch_size}

        {score_batch_size} {max_length} {precision}

        {wandb_enabled} {wandb_project}

        {wandb_run_name}
        """)
        .batch(
            checkpoint=mo.ui.dropdown(
                options=list(DEFAULT_CHECKPOINTS),
                value=default_config["checkpoint"],
                label="Hugging Face checkpoint",
                full_width=True,
            ),
            model_seeds=mo.ui.text(
                value=",".join(str(seed) for seed in default_config["model_seeds"]),
                label="Model seeds (comma-separated)",
                full_width=True,
            ),
            seed_workers=numeric_control(
                "seed_workers", "Concurrent seeds (shared device)", start=1
            ),
            uq_metric=mo.ui.dropdown(
                options=list(UQ_METRICS),
                value=default_config["uq_metric"],
                label="UQ metric",
            ),
            k=numeric_control("k", "Maximum new tokens per round", start=1, stop=1000),
            max_pool_percent=numeric_control(
                "max_pool_percent", "Maximum scoreable pool (%)", start=0.1, stop=100, step=0.1
            ),
            bootstrap_epochs=numeric_control(
                "bootstrap_epochs", "Bootstrap epochs", start=0, stop=100
            ),
            update_passes=numeric_control(
                "update_passes", "Update passes per round", start=1, stop=20
            ),
            replay_ratio=numeric_control(
                "replay_ratio", "Replay ratio", start=0, stop=4, step=0.25
            ),
            learning_rate=numeric_control(
                "learning_rate", "Learning rate", start=1e-06, stop=0.001, step=1e-06
            ),
            weight_decay=numeric_control(
                "weight_decay", "Weight decay", start=0, stop=0.2, step=0.001
            ),
            batch_size=numeric_control("batch_size", "Training batch size", start=1, stop=64),
            score_batch_size=numeric_control(
                "score_batch_size", "Scoring batch size", start=1, stop=256
            ),
            max_length=numeric_control(
                "max_length", "Maximum sequence length", start=4, stop=2048, step=4
            ),
            precision=mo.ui.dropdown(
                options=["auto", "fp32", "bf16"],
                value=default_config["precision"],
                label="Precision (auto: BF16 on supported CUDA GPUs)",
            ),
            wandb_enabled=mo.ui.checkbox(
                value=default_config["wandb_enabled"],
                label="Enable W&B logging",
            ),
            wandb_project=mo.ui.text(
                value=default_config["wandb_project"],
                label="W&B project",
                full_width=True,
            ),
            wandb_run_name=mo.ui.text(
                value=default_config["wandb_run_name"],
                label="W&B run name (generated when empty)",
                full_width=True,
            ),
        )
        .form(submit_button_label="Run experiment")
    )
    params_form
    return (params_form,)


@app.cell
def _(
    ExperimentConfig,
    cli_arguments,
    is_batch_mode,
    is_script_mode,
    mo,
    params_form,
):
    mo.stop(
        not is_script_mode and params_form.value is None,
        mo.md("*Submit the form to start training.*"),
    )
    if is_batch_mode:
        experiment_config = ExperimentConfig.model_validate_json(cli_arguments["config-json"])
    elif is_script_mode:
        experiment_config = ExperimentConfig()
    else:
        experiment_config = ExperimentConfig(**params_form.value)
    return (experiment_config,)


@app.cell
def _(
    Path,
    RESULTS_DIR,
    configure_wandb_metrics,
    env_config_widget,
    execute_experiment,
    experiment_config,
    is_batch_mode,
    is_script_mode,
    make_demo_run,
    make_learning_chart,
    make_wandb_comparison_media,
    make_wandb_evaluation_log,
    mo,
    os,
    pl,
    require_wandb_credentials,
    wandb,
):
    config = experiment_config.resolved_dict()
    if is_script_mode and not is_batch_mode:
        config, dataset_summary, result_records, selection_records = make_demo_run(config)
        run_dir = None
    else:
        wandb_run = None
        if config["wandb_enabled"]:
            offline = os.environ.get("WANDB_MODE", "").lower() == "offline"
            if not offline:
                if is_batch_mode:
                    require_wandb_credentials(os.environ)
                    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
                else:
                    env_config_widget.require_valid()
            wandb_run = wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                config=config,
            )
            configure_wandb_metrics(wandb_run, config["uq_metric"])

        seed_output_indices = {seed: index for index, seed in enumerate(config["model_seeds"])}

        def make_live_seed_panel(progress_records, model_seed):
            latest = next(
                (row for row in reversed(progress_records) if row["seed"] == model_seed),
                None,
            )
            status = "Waiting for round 0."
            chart = mo.md("Chart will appear after round 0.")
            if latest is not None:
                status = (
                    f"Round {latest['round']} of {latest['total_rounds']} · "
                    f"{latest['percent_acquired']:.2f}% acquired"
                )
                if latest["round"] == latest["total_rounds"]:
                    status += " · Complete"
                chart = make_learning_chart(progress_records, model_seed, config["uq_metric"])
            return mo.vstack(
                [
                    mo.md(f"### Live results — seed {model_seed}\n\n{status}"),
                    mo.vstack([chart]).style({"height": "420px", "overflow": "auto"}),
                ]
            )

        def update_live_chart(progress_records):
            latest = progress_records[-1]
            if not is_script_mode:
                mo.output.replace_at_index(
                    make_live_seed_panel(progress_records, latest["seed"]),
                    idx=seed_output_indices[latest["seed"]],
                )
            if wandb_run is not None:
                wandb_run.log(make_wandb_evaluation_log(progress_records[-2:], config["uq_metric"]))

        if not is_script_mode:
            mo.output.clear()
            for model_seed in config["model_seeds"]:
                mo.output.append(make_live_seed_panel([], model_seed))
        try:
            results_root = Path(os.environ.get("UQ_PET_RESULTS_DIR", RESULTS_DIR))
            config, dataset_summary, result_records, selection_records, run_dir = (
                execute_experiment(
                    experiment_config,
                    results_dir=results_root,
                    progress_callback=update_live_chart,
                )
            )
            if wandb_run is not None:
                wandb_run.config.update(
                    {
                        key: config[key]
                        for key in (
                            "scoreable_pool_tokens",
                            "token_budget",
                            "rounds",
                            "effective_pool_percent",
                        )
                    }
                )
                wandb_run.log(
                    make_wandb_comparison_media(wandb, result_records, config["uq_metric"])
                )
            if is_batch_mode:
                print(f"Batch output: {run_dir}")
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    results_df = pl.DataFrame(result_records)
    selections_df = pl.DataFrame(selection_records)
    seed_graphs = []
    for model_seed in config["model_seeds"]:
        seed_graphs.extend(
            [
                mo.md(f"## Seed {model_seed}"),
                make_learning_chart(result_records, model_seed, config["uq_metric"]),
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
            **{seed_description}**. Acquisition was capped at
            **{config["max_pool_percent"]:g}%** of the scoreable pool. This produced
            **{config["rounds"]} rounds** of **up to {config["k"]} new tokens per arm**
            and an effective endpoint of **{config["effective_pool_percent"]:.2f}%**.
            Each round trained for
            **{config["update_passes"]} pass(es)**, and replayed
            up to **{config["replay_ratio"]:g} older labels per new token**, including
            in the smaller final round when needed. The uncertainty arm used
            **{config["uq_metric"].replace("_", " ")}** scoring.
            """),
            mo.ui.table([dataset_summary], selection=None),
            mo.md(f"**Output:** `{output_location}`"),
            mo.accordion({"All settings": mo.ui.table([config], selection=None)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, make_variance_chart, mo, results_df):
    mo.vstack(
        [
            mo.md("""
            ## Across-seed variability

            Lines show the mean across seeds; shaded ranges show ±1 standard deviation.
            """),
            make_variance_chart(results_df, config["uq_metric"]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, mo, pl, results_df):
    display_results = results_df.with_columns(
        pl.col("arm").replace({"uncertainty": config["uq_metric"]})
    )
    mo.vstack([mo.md("## Round results"), mo.ui.table(display_results, selection=None)])
    return


@app.cell(hide_code=True)
def _(config, mo, pl, results_df):
    gap_name = f"{config['uq_metric']}_minus_random"
    paired = results_df.pivot(
        on="arm",
        index=["seed", "round", "n_acquired", "percent_acquired"],
        values="entity_f1",
    ).with_columns((pl.col("uncertainty") - pl.col("random")).alias(gap_name))
    gap_table = (
        paired.group_by(["round", "n_acquired", "percent_acquired"])
        .agg(
            pl.col(gap_name).mean().alias("mean"),
            pl.col(gap_name).std().alias("std"),
        )
        .sort(["percent_acquired", "round"])
    )
    mo.vstack(
        [
            mo.md(f"## Entity F1 gap — `{config['uq_metric']}` minus random"),
            mo.ui.table(gap_table, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, mo):
    selection_coverage_slider = mo.ui.slider(
        start=0.0,
        stop=float(config["effective_pool_percent"]),
        step=0.1,
        value=float(config["effective_pool_percent"]),
        label="Scoreable pool acquired (%)",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(r"""
            ## Cumulative NER-tag coverage

            Move the slider to compare the cumulative label mix at the same
            acquisition budget. Each bar is the acquired count for that gold label
            as a percentage of all scoreable pool tokens.
            """),
            selection_coverage_slider,
        ]
    )
    return (selection_coverage_slider,)


@app.cell
def _(
    make_tag_coverage_chart,
    config,
    pl,
    results_df,
    selection_coverage_slider,
    selections_df,
):
    selection_round_progress = results_df.select(
        "seed",
        "arm",
        "round",
        "percent_acquired",
        "scoreable_pool_tokens",
    ).unique()
    selections_with_progress = selections_df.join(
        selection_round_progress,
        on=["seed", "arm", "round"],
        how="left",
        validate="m:1",
    ).with_columns(pl.lit("interactive").alias("run_id"))
    selection_label_order = (
        selections_df.group_by("label")
        .agg(pl.len().alias("endpoint_count"))
        .sort("endpoint_count", descending=True)
        .get_column("label")
        .to_list()
    )
    make_tag_coverage_chart(
        selections_with_progress,
        selection_coverage_slider.value,
        config["uq_metric"],
        label_order=selection_label_order,
    )
    return


@app.cell(hide_code=True)
def _(config, mo, pl, selections_df):
    display_selections = selections_df.with_columns(
        pl.col("arm").replace({"uncertainty": config["uq_metric"]})
    )
    mo.vstack(
        [
            mo.md("## Selection log"),
            mo.ui.table(display_selections, selection=None, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
