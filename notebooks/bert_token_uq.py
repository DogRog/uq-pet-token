import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv(".env")

    mo.md(r"""
    # BERT token-level active learning

    Compare online updates from the **K most uncertain words** with the same number
    of **random words**. Both arms start from one seed-trained model, retain their
    weights and optimizer state, and use masked token loss with limited replay.
    """)
    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import polars as pl
    import wandb
    from wigglystuff import EnvConfig

    from uq_pet.active_learning import run_active_learning, write_run
    from uq_pet.pet_data import RESULTS_DIR, download_pet_ner, load_pet_splits
    from uq_pet.token_model import get_device

    alt.renderers.set_embed_options(scaleFactor=3)
    return (
        EnvConfig,
        Path,
        RESULTS_DIR,
        alt,
        download_pet_ner,
        get_device,
        load_pet_splits,
        os,
        pl,
        run_active_learning,
        sys,
        wandb,
        write_run,
    )


@app.cell
def _(alt, pl):
    def make_learning_chart(result_records, seed):
        curve = (
            pl.DataFrame(result_records)
            .filter(pl.col("seed") == seed)
            .sort(["arm", "percent_acquired"])
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
            "percent_acquired:Q",
            title="Scoreable pool acquired (%)",
            scale=alt.Scale(domain=[0, 100]),
        )
        entity_chart = (
            alt.Chart(curve)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=shared_x,
                y=alt.Y("entity_f1:Q", title="F1", scale=alt.Scale(zero=False)),
                color=arm_color,
                order=alt.Order("percent_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
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
                order=alt.Order("percent_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
                    alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                    alt.Tooltip("token_accuracy:Q", title="Token accuracy", format=".3f"),
                ],
            )
            .properties(title="Token accuracy", width=500, height=320)
        )
        return alt.hconcat(entity_chart, accuracy_chart, spacing=35).resolve_scale(color="shared")

    def make_variance_chart(results_frame):
        summary = (
            results_frame.group_by(["arm", "n_acquired", "percent_acquired"])
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
            .sort(["arm", "percent_acquired"])
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
            "percent_acquired:Q",
            title="Scoreable pool acquired (%)",
            scale=alt.Scale(domain=[0, 100]),
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
                        alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
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
def _():
    import hashlib
    import json
    from typing import Literal, Self

    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
    from rich import box
    from rich.console import Console
    from rich.table import Table

    default_checkpoints = (
        "distilbert-base-cased",
        "bert-base-cased",
        "roberta-base",
        "microsoft/deberta-v3-base",
        "answerdotai/ModernBERT-base",
    )

    class ModelParams(BaseModel):
        model_config = ConfigDict(extra="forbid")

        checkpoint: str = Field(
            default=default_checkpoints[0],
            min_length=1,
            description="Hugging Face token-classification base checkpoint.",
        )
        model_seeds: list[int] = Field(
            default_factory=lambda: [0, 1, 2, 3, 4],
            min_length=1,
            description="Comma-separated model initialization seeds.",
        )
        uq_metric: Literal["entropy", "least_confidence", "margin"] = Field(
            default="entropy",
            description="Larger-is-more-uncertain acquisition metric.",
        )
        k: int = Field(default=32, ge=1, description="New pool tokens selected per round and arm.")
        max_pool_percent: float = Field(
            default=100.0,
            gt=0,
            le=100,
            description="Maximum percentage of scoreable pool tokens to acquire.",
        )
        bootstrap_epochs: int = Field(
            default=20,
            ge=0,
            description="Passes over the five fully labelled seed sentences.",
        )
        update_passes: int = Field(
            default=1,
            ge=1,
            description="Passes over new and replayed tokens per round.",
        )
        replay_ratio: float = Field(
            default=1.0,
            ge=0,
            description="Older labelled tokens replayed per newly selected token.",
        )
        learning_rate: float = Field(
            default=5e-5,
            gt=0,
            description="AdamW learning rate for bootstrap and online updates.",
        )
        weight_decay: float = Field(
            default=0.01,
            ge=0,
            description="AdamW weight decay.",
        )
        batch_size: int = Field(default=8, ge=1, description="Training item batch size.")
        score_batch_size: int = Field(
            default=32,
            ge=1,
            description="Pool scoring and test evaluation batch size.",
        )
        max_length: int = Field(
            default=256,
            ge=4,
            description="Maximum tokenizer sequence length.",
        )
        wandb_enabled: bool = Field(
            default=False,
            description="Log settings and evaluation rows to Weights & Biases.",
        )
        wandb_project: str = Field(
            default="uq-pet-token-uq",
            description="Weights & Biases project name.",
        )
        wandb_run_name: str = Field(
            default="",
            description="Optional W&B run name; generated when empty.",
        )

        @field_validator("model_seeds", mode="before")
        @classmethod
        def parse_model_seeds(cls, value):
            if isinstance(value, str):
                values = [part.strip() for part in value.split(",") if part.strip()]
                return [int(part) for part in values]
            if isinstance(value, int):
                return [value]
            return value

        @field_validator("checkpoint")
        @classmethod
        def validate_checkpoint(cls, value):
            if not value.strip():
                raise ValueError("checkpoint must not be blank")
            return value.strip()

        @field_validator("wandb_project", "wandb_run_name")
        @classmethod
        def strip_wandb_text(cls, value):
            return value.strip()

        @model_validator(mode="after")
        def validate_run(self) -> Self:
            if any(seed < 0 for seed in self.model_seeds):
                raise ValueError("model seeds must be non-negative")
            if len(self.model_seeds) != len(set(self.model_seeds)):
                raise ValueError("model seeds must be unique")
            if self.wandb_enabled and not self.wandb_project.strip():
                raise ValueError("wandb_project is required when W&B logging is enabled")
            return self

        def resolved_wandb_run_name(self) -> str:
            if self.wandb_run_name.strip():
                return self.wandb_run_name.strip()
            checkpoint_name = self.checkpoint.rsplit("/", maxsplit=1)[-1]
            payload = self.model_dump(exclude={"wandb_run_name"})
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
            seeds = "-".join(str(seed) for seed in self.model_seeds)
            return f"{checkpoint_name}-seeds{seeds}-{digest}"

    def print_cli_help():
        console = Console()
        console.print(
            "[bold]Usage:[/] [cyan]uv run notebooks/bert_token_uq.py[/] "
            "[green]--checkpoint[/] [italic]<id>[/] [dim]\\[options][/]"
        )
        console.print(
            "[dim]No arguments runs the synthetic validation; "
            "any experiment option runs training.[/]"
        )
        console.print()
        table = Table(
            title="CLI Options",
            title_style="bold italic",
            box=box.ROUNDED,
            header_style="bold white",
            border_style="bright_blue",
            row_styles=("", "dim"),
        )
        table.add_column("Flag", style="bold cyan", no_wrap=True)
        table.add_column("Type", style="green", no_wrap=True)
        table.add_column("Default", style="yellow")
        table.add_column("Description")
        for name, field in ModelParams.model_fields.items():
            default = field.get_default(call_default_factory=True)
            annotation = str(field.annotation).replace("typing.", "")
            if annotation.startswith("<class '"):
                annotation = annotation.removeprefix("<class '").removesuffix("'>")
                annotation = annotation.rsplit(".", maxsplit=1)[-1]
            table.add_row(
                f"--{name.replace('_', '-')}",
                annotation,
                str(default),
                field.description or "",
            )
        console.print(table)

    return ModelParams, default_checkpoints, print_cli_help


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    cli_arguments = dict(mo.cli_args()) if is_script_mode else {}
    is_batch_mode = is_script_mode and bool(cli_arguments)
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
    return env_config_widget


@app.cell
def _(default_checkpoints, mo):
    params_form = (
        mo.md("""
        ## Experiment parameters

        {checkpoint}

        {model_seeds}

        {uq_metric} {k} {max_pool_percent}

        {bootstrap_epochs} {update_passes} {replay_ratio}

        {learning_rate} {weight_decay} {batch_size}

        {score_batch_size} {max_length}

        {wandb_enabled} {wandb_project}

        {wandb_run_name}
        """)
        .batch(
            checkpoint=mo.ui.dropdown(
                options=list(default_checkpoints),
                value=default_checkpoints[0],
                label="Hugging Face checkpoint",
                full_width=True,
            ),
            model_seeds=mo.ui.text(
                value="0,1,2,3,4",
                label="Model seeds (comma-separated)",
                full_width=True,
            ),
            uq_metric=mo.ui.dropdown(
                options={
                    "Predictive entropy": "entropy",
                    "Least confidence": "least_confidence",
                    "Margin": "margin",
                },
                value="Predictive entropy",
                label="UQ metric",
            ),
            k=mo.ui.number(start=1, stop=1000, step=1, value=32, label="New tokens per round"),
            max_pool_percent=mo.ui.number(
                start=0.1,
                stop=100,
                step=0.1,
                value=100,
                label="Maximum scoreable pool (%)",
            ),
            bootstrap_epochs=mo.ui.number(
                start=0, stop=100, step=1, value=20, label="Bootstrap epochs"
            ),
            update_passes=mo.ui.number(
                start=1, stop=20, step=1, value=1, label="Update passes per round"
            ),
            replay_ratio=mo.ui.number(start=0, stop=4, step=0.25, value=1.0, label="Replay ratio"),
            learning_rate=mo.ui.number(
                start=0.000001,
                stop=0.001,
                step=0.000001,
                value=0.00005,
                label="Learning rate",
            ),
            weight_decay=mo.ui.number(
                start=0, stop=0.2, step=0.001, value=0.01, label="Weight decay"
            ),
            batch_size=mo.ui.number(start=1, stop=64, step=1, value=8, label="Training batch size"),
            score_batch_size=mo.ui.number(
                start=1, stop=256, step=1, value=32, label="Scoring batch size"
            ),
            max_length=mo.ui.number(
                start=4, stop=2048, step=4, value=256, label="Maximum sequence length"
            ),
            wandb_enabled=mo.ui.checkbox(value=False, label="Enable W&B logging"),
            wandb_project=mo.ui.text(value="uq-pet-token-uq", label="W&B project", full_width=True),
            wandb_run_name=mo.ui.text(
                value="", label="W&B run name (generated when empty)", full_width=True
            ),
        )
        .form(submit_button_label="Run experiment")
    )
    params_form
    return params_form


@app.cell
def _(
    ModelParams,
    cli_arguments,
    is_batch_mode,
    is_script_mode,
    mo,
    params_form,
    print_cli_help,
    sys,
):
    if is_script_mode and "help" in cli_arguments:
        print_cli_help()
        sys.exit(0)

    mo.stop(
        not is_script_mode and params_form.value is None,
        mo.md("*Submit the form to start training.*"),
    )
    if is_batch_mode:
        model_params = ModelParams(
            **{key.replace("-", "_"): value for key, value in cli_arguments.items()}
        )
    elif is_script_mode:
        model_params = ModelParams()
    else:
        model_params = ModelParams(**params_form.value)
    return (model_params,)


@app.cell
def _(
    Path,
    RESULTS_DIR,
    download_pet_ner,
    env_config_widget,
    get_device,
    is_batch_mode,
    is_script_mode,
    load_pet_splits,
    make_learning_chart,
    model_params,
    mo,
    os,
    pl,
    run_active_learning,
    wandb,
    write_run,
):
    config = model_params.model_dump()
    config["wandb_run_name"] = model_params.resolved_wandb_run_name()
    experiment_config = {
        key: value
        for key, value in config.items()
        if key not in {"wandb_enabled", "wandb_project", "wandb_run_name"}
    }

    if is_script_mode and not is_batch_mode:
        # Script mode validates notebook execution without downloading model weights.
        demo_scoreable_tokens = 96
        demo_requested_tokens = int(demo_scoreable_tokens * config["max_pool_percent"] / 100)
        demo_rounds = demo_requested_tokens // config["k"]
        demo_token_budget = demo_rounds * config["k"]
        config.update(
            {
                "scoreable_pool_tokens": demo_scoreable_tokens,
                "token_budget": demo_token_budget,
                "rounds": demo_rounds,
                "effective_pool_percent": 100 * demo_token_budget / demo_scoreable_tokens,
            }
        )
        result_records = [
            {
                "seed": model_seed,
                "arm": arm,
                "round": round_idx,
                "total_rounds": demo_rounds,
                "n_acquired": round_idx * config["k"],
                "percent_acquired": 100 * round_idx * config["k"] / demo_scoreable_tokens,
                "scoreable_pool_tokens": demo_scoreable_tokens,
                "token_budget": demo_token_budget,
                "entity_f1": 0.10
                + model_seed * 0.001
                + round_idx * (0.03 if arm == "uncertainty" else 0.02),
                "entity_precision": 0.12,
                "entity_recall": 0.10,
                "token_accuracy": 0.70 + round_idx * 0.01,
                "train_loss": 1.0 / (round_idx + 1),
                "n_new": 0 if round_idx == 0 else config["k"],
                "n_replay": 0 if round_idx == 0 else config["k"],
            }
            for model_seed in config["model_seeds"]
            for arm in ("uncertainty", "random")
            for round_idx in range(demo_rounds + 1)
        ]
        selection_records = [
            {
                "seed": model_seed,
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
            for model_seed in config["model_seeds"]
            for arm in ("uncertainty", "random")
            for idx, (token, label) in enumerate((("check", "B-Activity"), ("form", "O")))
        ]
        dataset_summary = {
            "mode": "script demo",
            "seed_sentences": 5,
            "pool_sentences": 328,
            "scoreable_pool_tokens": demo_scoreable_tokens,
            "acquisition_budget_tokens": demo_token_budget,
            "acquisition_rounds": demo_rounds,
            "effective_pool_percent": config["effective_pool_percent"],
            "test_sentences": 84,
            "device": "not loaded",
        }
        run_dir = None
    else:
        wandb_run = None
        if config["wandb_enabled"]:
            offline = os.environ.get("WANDB_MODE", "").lower() == "offline"
            if is_script_mode and not offline:
                api_key = os.environ.get("WANDB_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "W&B logging is enabled but WANDB_API_KEY is missing; "
                        "set it in the environment or .env"
                    )
                wandb.login(key=api_key, verify=True)
            elif not is_script_mode and not offline:
                env_config_widget.require_valid()
            wandb_run = wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                config=config,
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
        wandb_rows_logged = [0]

        def update_live_chart(progress_records):
            latest = progress_records[-1]
            if not is_script_mode:
                seed_position = config["model_seeds"].index(latest["seed"]) + 1
                mo.output.replace(
                    mo.vstack(
                        [
                            mo.md(
                                f"### Live results — seed {latest['seed']} "
                                f"({seed_position} of {len(config['model_seeds'])}), "
                                f"round {latest['round']} of {latest['total_rounds']} · "
                                f"{latest['percent_acquired']:.2f}% acquired"
                            ),
                            make_learning_chart(progress_records, latest["seed"]),
                        ]
                    )
                )
            if wandb_run is not None:
                for row in progress_records[wandb_rows_logged[0] :]:
                    wandb_run.log({f"evaluation/{key}": value for key, value in row.items()})
                wandb_rows_logged[0] = len(progress_records)

        if not is_script_mode:
            mo.output.replace(
                mo.md("Training the seed model; the chart will appear after round 0.")
            )
        try:
            result_records, selection_records = run_active_learning(
                seed_examples,
                pool_inputs,
                pool_gold,
                test_examples,
                **experiment_config,
                device=device,
                progress_callback=update_live_chart,
            )
            first_result = result_records[0]
            derived_config = {
                "scoreable_pool_tokens": first_result["scoreable_pool_tokens"],
                "token_budget": first_result["token_budget"],
                "rounds": first_result["total_rounds"],
                "effective_pool_percent": 100
                * first_result["token_budget"]
                / first_result["scoreable_pool_tokens"],
            }
            config.update(derived_config)
            dataset_summary.update(
                {
                    "scoreable_pool_tokens": config["scoreable_pool_tokens"],
                    "acquisition_budget_tokens": config["token_budget"],
                    "acquisition_rounds": config["rounds"],
                    "effective_pool_percent": config["effective_pool_percent"],
                }
            )
            if wandb_run is not None:
                wandb_run.config.update(derived_config)
            results_root = Path(os.environ.get("UQ_PET_RESULTS_DIR", RESULTS_DIR))
            run_dir = write_run(
                config,
                result_records,
                selection_records,
                results_dir=results_root,
            )
            if is_script_mode:
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
            **{seed_description}**. Acquisition was capped at
            **{config["max_pool_percent"]:g}%** of the scoreable pool. This produced
            **{config["rounds"]} full rounds** of **{config["k"]} new tokens per arm**
            and an effective endpoint of **{config["effective_pool_percent"]:.2f}%**.
            Each round trained for
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
        index=["seed", "round", "n_acquired", "percent_acquired"],
        values="entity_f1",
    ).with_columns((pl.col("uncertainty") - pl.col("random")).alias("uncertainty_minus_random"))
    gap_table = (
        paired.group_by(["round", "n_acquired", "percent_acquired"])
        .agg(
            pl.col("uncertainty_minus_random").mean().alias("mean"),
            pl.col("uncertainty_minus_random").std().alias("std"),
        )
        .sort(["percent_acquired", "round"])
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
