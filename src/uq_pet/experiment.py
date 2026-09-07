"""Shared experiment configuration, execution, charts, and W&B records."""

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Self

import altair as alt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uq_pet.active_learning import run_active_learning, write_run
from uq_pet.pet_data import RESULTS_DIR, download_pet_ner, load_pet_splits
from uq_pet.token_model import UQ_METRICS, get_device

DEFAULT_CHECKPOINTS = (
    "distilbert-base-cased",
    "bert-base-cased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base",
)
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)

_WANDB_ARMS = ("uncertainty", "random")
_WANDB_COMMON_FIELDS = (
    "seed",
    "round",
    "total_rounds",
    "n_acquired",
    "percent_acquired",
    "scoreable_pool_tokens",
    "token_budget",
)
_WANDB_HIDDEN_ARM_FIELDS = ("n_new", "n_replay")
_WANDB_TRACKED_ARM_FIELDS = (
    "entity_f1",
    "entity_macro_f1",
    "entity_precision",
    "entity_recall",
    "token_accuracy",
    "train_loss",
)


class ExperimentConfig(BaseModel):
    """The single validated configuration shared by notebook and batch runs."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field(
        default=DEFAULT_CHECKPOINTS[0],
        min_length=1,
        description="Hugging Face token-classification base checkpoint.",
    )
    model_seeds: list[int] = Field(
        default_factory=lambda: list(DEFAULT_MODEL_SEEDS),
        min_length=1,
        description="Model initialization seeds.",
    )
    seed_workers: int = Field(
        default=1,
        ge=1,
        description="Concurrent seed processes on the selected device; capped by the seed count.",
    )
    uq_metric: str = Field(
        default=UQ_METRICS[0],
        description="Larger-is-more-uncertain acquisition metric.",
    )
    k: int = Field(default=32, ge=1, description="Maximum new pool tokens per round and arm.")
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
        description="Older labelled tokens replayed per newly selected token, including the final round.",
    )
    learning_rate: float = Field(
        default=5e-5,
        gt=0,
        description="AdamW learning rate for bootstrap and online updates.",
    )
    weight_decay: float = Field(default=0.01, ge=0, description="AdamW weight decay.")
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

    @field_validator("uq_metric")
    @classmethod
    def validate_uq_metric(cls, value):
        if value not in UQ_METRICS:
            raise ValueError(f"uq_metric must be one of {UQ_METRICS}")
        return value

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
        if self.wandb_enabled and not self.wandb_project:
            raise ValueError("wandb_project is required when W&B logging is enabled")
        return self

    def resolved_wandb_run_name(self) -> str:
        if self.wandb_run_name:
            return self.wandb_run_name
        checkpoint_name = self.checkpoint.rsplit("/", maxsplit=1)[-1]
        payload = self.model_dump(exclude={"wandb_run_name"})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
        seeds = "-".join(str(seed) for seed in self.model_seeds)
        return f"{checkpoint_name}-seeds{seeds}-{digest}"

    def resolved_dict(self) -> dict:
        config = self.model_dump()
        config["wandb_run_name"] = self.resolved_wandb_run_name()
        return config

    def active_learning_kwargs(self) -> dict:
        return self.model_dump(exclude={"wandb_enabled", "wandb_project", "wandb_run_name"})


def execute_experiment(
    config: ExperimentConfig,
    *,
    results_dir: Path = RESULTS_DIR,
    progress_callback: Callable[[list[dict]], None] | None = None,
) -> tuple[dict, dict, list[dict], list[dict], Path]:
    """Load PET, run both arms, persist the run, and return display-ready records."""
    data_path = download_pet_ner()
    seed_examples, pool_inputs, pool_gold, test_examples = load_pet_splits(data_path)
    device = get_device()
    results, selections = run_active_learning(
        seed_examples,
        pool_inputs,
        pool_gold,
        test_examples,
        **config.active_learning_kwargs(),
        device=device,
        progress_callback=progress_callback,
    )

    first_result = results[0]
    derived_config = {
        "scoreable_pool_tokens": first_result["scoreable_pool_tokens"],
        "token_budget": first_result["token_budget"],
        "rounds": first_result["total_rounds"],
        "effective_pool_percent": (
            100 * first_result["token_budget"] / first_result["scoreable_pool_tokens"]
        ),
    }
    run_config = {**config.resolved_dict(), **derived_config}
    dataset_summary = {
        "mode": "experiment",
        "seed_sentences": len(seed_examples),
        "pool_sentences": len(pool_inputs),
        "pool_tokens": len(pool_gold),
        "scoreable_pool_tokens": derived_config["scoreable_pool_tokens"],
        "acquisition_budget_tokens": derived_config["token_budget"],
        "acquisition_rounds": derived_config["rounds"],
        "effective_pool_percent": derived_config["effective_pool_percent"],
        "test_sentences": len(test_examples),
        "device": str(device),
    }
    run_dir = write_run(run_config, results, selections, results_dir=results_dir)
    return run_config, dataset_summary, results, selections, run_dir


def require_wandb_credentials(environment: Mapping[str, str]) -> None:
    """Reject an online W&B run before downloading data or starting training."""
    if environment.get("WANDB_MODE", "").strip().lower() == "offline":
        return
    if environment.get("WANDB_API_KEY", "").strip():
        return
    raise ValueError("WANDB_API_KEY is missing; set it unless WANDB_MODE=offline")


def summarize_label_pool_share(
    selections: pl.DataFrame, acquisition_percent: float
) -> pl.DataFrame:
    """Summarize revealed labels against the full scoreable pool in each run."""
    run_columns = ["run_id", "seed", "arm"]
    label_grid = (
        selections.select(*run_columns, "scoreable_pool_tokens")
        .unique()
        .join(selections.select("label").unique(), how="cross")
    )
    counts = (
        selections.filter(pl.col("percent_acquired") <= acquisition_percent)
        .group_by([*run_columns, "label"])
        .agg(pl.len().alias("n_acquired"))
    )
    return (
        label_grid.join(counts, on=[*run_columns, "label"], how="left", validate="1:1")
        .with_columns(pl.col("n_acquired").fill_null(0))
        .with_columns(
            (100 * pl.col("n_acquired") / pl.col("scoreable_pool_tokens")).alias("pool_share")
        )
        .group_by(["arm", "label"])
        .agg(
            pl.col("pool_share").mean().alias("pool_share_mean"),
            pl.col("pool_share").std().fill_null(0.0).alias("pool_share_std"),
            pl.col("n_acquired").mean().alias("n_acquired_mean"),
            pl.col("scoreable_pool_tokens").mean().alias("scoreable_pool_tokens_mean"),
        )
        .sort(["label", "arm"])
    )


def make_learning_chart(result_records: list[dict], seed: int, uq_metric: str):
    """Build the paired entity-F1, macro-F1, and accuracy chart for one model seed."""
    records = (
        pl.DataFrame(result_records)
        if result_records
        else pl.DataFrame(
            schema={
                "seed": pl.Int64,
                "arm": pl.String,
                "percent_acquired": pl.Float64,
                "n_acquired": pl.Int64,
                "entity_f1": pl.Float64,
                "entity_macro_f1": pl.Float64,
                "token_accuracy": pl.Float64,
            }
        )
    )
    curve = (
        records.filter(pl.col("seed") == seed)
        .with_columns(pl.col("arm").replace({"uncertainty": uq_metric}))
        .sort(["arm", "percent_acquired"])
    )
    arm_order = ["random", uq_metric]
    arm_color = alt.Color(
        "arm:N",
        title=None,
        sort=arm_order,
        scale=alt.Scale(domain=arm_order, range=["#4C78A8", "#F58518"]),
        legend=alt.Legend(orient="top"),
    )
    shared_x = alt.X(
        "percent_acquired:Q",
        title="Scoreable pool acquired (%)",
        scale=alt.Scale(domain=[0, 100]),
    )

    def metric_chart(field: str, title: str, y_title: str):
        return (
            alt.Chart(curve)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=shared_x,
                y=alt.Y(f"{field}:Q", title=y_title, scale=alt.Scale(zero=False)),
                color=arm_color,
                order=alt.Order("percent_acquired:Q"),
                tooltip=[
                    alt.Tooltip("arm:N", title="Arm"),
                    alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
                    alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                    alt.Tooltip(f"{field}:Q", title=title, format=".3f"),
                ],
            )
            .properties(title=title, width=500, height=320)
        )

    return alt.hconcat(
        metric_chart("entity_f1", "Entity F1", "F1"),
        metric_chart("entity_macro_f1", "Macro entity F1", "Macro F1"),
        metric_chart("token_accuracy", "Token accuracy", "Accuracy"),
        spacing=35,
    ).resolve_scale(color="shared")


def make_variance_chart(
    results_frame: pl.DataFrame, uq_metric: str, acquisition_percent: float | None = None
):
    """Plot seed means and clipped ±1 SD bands, including older runs without macro F1."""
    metrics = [
        (field, title, y_title)
        for field, title, y_title in (
            ("entity_f1", "Entity F1", "F1"),
            ("entity_macro_f1", "Macro entity F1", "Macro F1"),
            ("token_accuracy", "Token accuracy", "Accuracy"),
        )
        if field != "entity_macro_f1" or field in results_frame.columns
    ]
    summary = (
        results_frame.with_columns(pl.col("arm").replace({"uncertainty": uq_metric}))
        .group_by(["arm", "n_acquired", "percent_acquired"])
        .agg(
            expression
            for field, _, _ in metrics
            for expression in (
                pl.col(field).mean().alias(f"{field}_mean"),
                pl.col(field).std().fill_null(0.0).alias(f"{field}_std"),
            )
        )
        .with_columns(
            (pl.col(f"{field}_mean") + sign * pl.col(f"{field}_std"))
            .clip(0.0, 1.0)
            .alias(f"{field}_{bound}")
            for field, _, _ in metrics
            for bound, sign in (("lower", -1), ("upper", 1))
        )
        .sort(["arm", "percent_acquired"])
    )
    arm_order = ["random", uq_metric]
    base = alt.Chart(summary).encode(
        x=alt.X(
            "percent_acquired:Q",
            title="Scoreable pool acquired (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "arm:N",
            title=None,
            sort=arm_order,
            scale=alt.Scale(domain=arm_order, range=["#4C78A8", "#F58518"]),
            legend=alt.Legend(orient="top"),
        ),
    )

    def metric_chart(field, title, y_title):
        band = base.mark_area(opacity=0.18).encode(
            y=alt.Y(f"{field}_lower:Q", title=y_title, scale=alt.Scale(zero=False)),
            y2=alt.Y2(f"{field}_upper:Q"),
        )
        mean_line = base.mark_line(strokeWidth=3).encode(
            y=alt.Y(f"{field}_mean:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("percent_acquired:Q", title="Pool acquired", format=".2f"),
                alt.Tooltip("n_acquired:Q", title="Acquired", format=".0f"),
                alt.Tooltip(f"{field}_mean:Q", title="Mean", format=".3f"),
                alt.Tooltip(f"{field}_std:Q", title="Std. dev.", format=".3f"),
            ],
        )
        layers = [band, mean_line]
        if acquisition_percent is not None:
            layers.append(
                alt.Chart(pl.DataFrame({"selected_percent": [float(acquisition_percent)]}))
                .mark_rule(color="#E45756", strokeDash=[7, 5], strokeWidth=2)
                .encode(
                    x=alt.X("selected_percent:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("selected_percent:Q", title="Coverage budget (%)", format=".0f")
                    ],
                )
            )
        return alt.layer(*layers).properties(title=title, width=500, height=320)

    return alt.hconcat(*(metric_chart(*metric) for metric in metrics), spacing=35).resolve_scale(
        color="shared"
    )


def configure_wandb_metrics(wandb_run, uq_metric: str) -> None:
    """Keep bookkeeping out of auto-panels and use acquisition as the x-axis."""
    for field in _WANDB_COMMON_FIELDS:
        wandb_run.define_metric(f"evaluation/{field}", hidden=True)
    for field in _WANDB_HIDDEN_ARM_FIELDS:
        for arm in (uq_metric, "random"):
            wandb_run.define_metric(f"evaluation/{field}/{arm}", hidden=True)
    for field in _WANDB_TRACKED_ARM_FIELDS:
        for arm in (uq_metric, "random"):
            wandb_run.define_metric(
                f"evaluation/{field}/{arm}",
                step_metric="evaluation/percent_acquired",
            )


def make_wandb_evaluation_log(rows: list[dict], uq_metric: str) -> dict:
    """Build one W&B step with a separate scalar series for each arm."""
    if len(rows) != len(_WANDB_ARMS):
        raise ValueError("a W&B evaluation step requires exactly one row per arm")

    rows_by_arm = {row["arm"]: row for row in rows}
    if set(rows_by_arm) != set(_WANDB_ARMS):
        raise ValueError("a W&B evaluation step requires uncertainty and random rows")

    first = rows_by_arm[_WANDB_ARMS[0]]
    for field in _WANDB_COMMON_FIELDS:
        if any(rows_by_arm[arm][field] != first[field] for arm in _WANDB_ARMS[1:]):
            raise ValueError(f"W&B evaluation rows disagree on {field}")

    payload = {f"evaluation/{field}": first[field] for field in _WANDB_COMMON_FIELDS}
    arm_fields = set(first) - set(_WANDB_COMMON_FIELDS) - {"arm"}
    for arm in _WANDB_ARMS:
        if set(rows_by_arm[arm]) - set(_WANDB_COMMON_FIELDS) - {"arm"} != arm_fields:
            raise ValueError("W&B evaluation rows have different metric fields")
        for field in sorted(arm_fields):
            display_arm = uq_metric if arm == "uncertainty" else arm
            payload[f"evaluation/{field}/{display_arm}"] = rows_by_arm[arm][field]
    return payload


def make_wandb_comparison_media(wandb_module, result_records: list[dict], uq_metric: str) -> dict:
    """Build one table-free interactive comparison panel per model seed."""
    return {
        f"final_comparison/seed_{seed}_{uq_metric}_vs_random": wandb_module.Html(
            make_learning_chart(result_records, seed, uq_metric).to_html(),
            inject=False,
        )
        for seed in sorted({row["seed"] for row in result_records})
    }
