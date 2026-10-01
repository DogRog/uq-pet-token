"""Validated settings shared by notebook, experiment, and search entry points."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uq_pet.pet_data import RESULTS_DIR
from uq_pet.token_model import UQ_METRICS

DEFAULT_CHECKPOINTS = (
    "distilbert-base-cased",
    "bert-base-cased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base",
)
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)
Dataset = Literal["pet", "conll2003"]


DATASET_DEFAULTS = {"dataset": "pet", "dataset_percent": 100.0}


def omit_default_dataset(settings: dict) -> dict:
    """Leave the full PET pool implicit so plans saved before these fields still resume."""
    return {
        key: value
        for key, value in settings.items()
        if key not in DATASET_DEFAULTS or value != DATASET_DEFAULTS[key]
    }


class ExperimentConfig(BaseModel):
    """The single validated configuration shared by notebook and batch runs."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field(
        default=DEFAULT_CHECKPOINTS[0],
        min_length=1,
        description="Hugging Face token-classification base checkpoint.",
    )
    dataset: Dataset = Field(
        default="pet",
        description="PET, or CoNLL-2003 with seed and pool from train and the full test split.",
    )
    dataset_percent: float = Field(
        default=100.0,
        gt=0,
        le=100,
        description="Percentage of the dataset's training pool kept for acquisition.",
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
    precision: Literal["auto", "fp32", "bf16"] = Field(
        default="auto",
        description="Auto uses BF16 on supported CUDA GPUs and FP32 elsewhere.",
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
        default=256,
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
        if self.dataset != "pet":
            checkpoint_name = f"{checkpoint_name}-{self.dataset}"
        if self.dataset_percent != 100:
            checkpoint_name = f"{checkpoint_name}-{self.dataset_percent:g}pct"
        payload = omit_default_dataset(self.model_dump(exclude={"wandb_run_name"}))
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
        seeds = "-".join(str(seed) for seed in self.model_seeds)
        return f"{checkpoint_name}-seeds{seeds}-{digest}"

    def resolved_dict(self) -> dict:
        config = self.model_dump()
        config["wandb_run_name"] = self.resolved_wandb_run_name()
        return config

    def active_learning_kwargs(self) -> dict:
        return self.model_dump(
            exclude={"dataset_percent", "wandb_enabled", "wandb_project", "wandb_run_name"}
        )


class RandomSearchConfig(ExperimentConfig):
    """Fixed random-sweep budget and settings, validated before any training."""

    mode: Literal["compare", "tune-random"] = "compare"
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
        """Strip sweep settings while preserving the requested experiment logging."""
        return ExperimentConfig.model_validate(
            self.model_dump(include=set(ExperimentConfig.model_fields))
        )


class FixedComparisonConfig(RandomSearchConfig):
    """Compare one supplied configuration without sampling hyperparameters."""

    mode: Literal["compare-fixed"] = "compare-fixed"
    num_configs: Literal[1] = 1
    uq_metrics: list[str] = Field(default_factory=lambda: list(UQ_METRICS), min_length=1)

    @field_validator("uq_metrics")
    @classmethod
    def validate_metrics(cls, value):
        if len(set(value)) != len(value) or any(metric not in UQ_METRICS for metric in value):
            raise ValueError("uq-metrics must be unique supported metrics")
        return value


class RandomBaselineSearchConfig(RandomSearchConfig):
    """Tune random acquisition on validation and save the winning configuration."""

    mode: Literal["tune-random"] = "tune-random"
    num_configs: int = Field(default=50, ge=1)
    sweep_name: str = "distilbert-random-baseline-50"
    sweeps_dir: Path = RESULTS_DIR / "random_baseline_search"
    validation_sentences: int = Field(default=66, ge=1)
    validation_seed: int = Field(default=1729, ge=0)
    objective: Literal["random_validation_entity_f1_auc", "random_validation_final_entity_f1"] = (
        "random_validation_entity_f1_auc"
    )


class SupervisedConfig(BaseModel):
    """Fully labelled sentence training: validation random search, then fixed-epoch test runs."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field(default=DEFAULT_CHECKPOINTS[0], min_length=1)
    dataset: Dataset = "pet"
    dataset_percent: float = Field(
        default=100.0,
        gt=0,
        le=100,
        description="Percentage of the dataset's training pool kept for tuning and test runs.",
    )
    model_seeds: list[int] = Field(
        default_factory=lambda: list(DEFAULT_MODEL_SEEDS),
        min_length=1,
        description="Model initialization seeds; each also orders the pool sentences.",
    )
    seed_workers: int = Field(
        default=1,
        ge=1,
        description="Concurrent seed processes on the selected device; capped by the seed count.",
    )
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    epochs: int = Field(default=20, ge=1, description="Fixed passes; no early stopping.")
    num_configs: int = Field(default=30, ge=1, description="Sampled tuning configurations.")
    sampler_seed: int = Field(default=0, ge=0)
    batch_sizes: list[int] = Field(default_factory=lambda: [8, 16, 32], min_length=1)
    weight_decays: list[float] = Field(default_factory=lambda: [0.0, 0.01], min_length=1)
    score_batch_size: int = Field(default=256, ge=1)
    max_length: int = Field(default=256, ge=4)
    validation_sentences: int = Field(default=66, ge=1)
    validation_seed: int = Field(default=1729, ge=0)
    sentence_percents: list[float] = Field(
        default_factory=lambda: [100.0],
        min_length=1,
        description="Pool sentence percentages trained on for the test runs; 100 is the upper bound.",
    )
    sweep_name: str = ""
    sweeps_dir: Path = RESULTS_DIR / "supervised"
    wandb_enabled: bool = Field(
        default=False,
        description="Log each tuning trial and test budget to Weights & Biases.",
    )
    wandb_project: str = Field(
        default="uq-pet-token-supervised",
        description="Weights & Biases project name.",
    )

    @field_validator("model_seeds", mode="before")
    @classmethod
    def parse_model_seeds(cls, value):
        return ExperimentConfig.parse_model_seeds(value)

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(cls, value):
        return ExperimentConfig.validate_checkpoint(value)

    @field_validator("wandb_project")
    @classmethod
    def strip_wandb_project(cls, value):
        return value.strip()

    @field_validator("sweep_name")
    @classmethod
    def validate_sweep_name(cls, value):
        if value and not all(c.isalnum() or c in "-_" for c in value):
            raise ValueError(
                "sweep_name must contain only letters, numbers, hyphens or underscores"
            )
        return value

    @model_validator(mode="after")
    def validate_search(self) -> Self:
        if any(seed < 0 for seed in self.model_seeds):
            raise ValueError("model seeds must be non-negative")
        for name in ("model_seeds", "batch_sizes", "weight_decays"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if any(value <= 0 for value in self.batch_sizes):
            raise ValueError("batch sizes must be positive")
        if any(value < 0 for value in self.weight_decays):
            raise ValueError("weight decays must be non-negative")
        percents = self.sentence_percents
        if percents != sorted(set(percents)) or not all(0 <= p <= 100 for p in percents):
            raise ValueError("sentence_percents must be sorted, unique, and within [0, 100]")
        if self.wandb_enabled and not self.wandb_project:
            raise ValueError("wandb_project is required when W&B logging is enabled")
        if not self.sweep_name:
            dataset = "" if self.dataset == "pet" else f"-{self.dataset}"
            if self.dataset_percent != 100:
                dataset += f"-{self.dataset_percent:g}pct".replace(".", "p")
            self.sweep_name = self.checkpoint.rsplit("/", maxsplit=1)[-1] + dataset + "-supervised"
        return self
