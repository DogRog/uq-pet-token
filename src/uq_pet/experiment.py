"""Shared experiment configuration, execution, summaries, and W&B records."""

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Self

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


class RandomSearchConfig(ExperimentConfig):
    """Fixed random-sweep budget and settings, validated before any training."""

    num_configs: int = Field(ge=1)
    sweep_name: str = "bert-token-uq-random"
    sweeps_dir: Path = RESULTS_DIR / "random_search"
    sampler_seed: int = Field(default=0, ge=0)

    @field_validator("sweep_name")
    @classmethod
    def validate_sweep_name(cls, value):
        if not value or not all(c.isalnum() or c in "-_" for c in value):
            raise ValueError(
                "sweep_name must contain only letters, numbers, hyphens or underscores"
            )
        return value

    def experiment_config(self) -> ExperimentConfig:
        """Strip sweep settings and disable W&B for the paired sweep runs."""
        return ExperimentConfig.model_validate(
            {
                **self.model_dump(include=set(ExperimentConfig.model_fields)),
                "wandb_enabled": False,
                "wandb_run_name": "",
            }
        )


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
    run_config = {**config.resolved_dict(), **derived_config, "evaluation_split": "test"}
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
