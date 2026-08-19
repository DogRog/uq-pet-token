"""Dataset identity, paths, and YAML-backed experiment configuration."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# These values define the stable train/test split used by both experiment branches.
SEED = 3407
TEST_SIZE = 0.2
SEED_SPLIT_SEED = 42
N_SEED_EXAMPLES = 5


def project_root() -> Path:
    """Repo root: $UQ_PET_ROOT if set, else two levels above this module."""
    env = os.environ.get("UQ_PET_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATASET_PATH = RAW_DATA_DIR / "PETv1.1-entities.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

NER_DATASET_URL = (
    "https://raw.githubusercontent.com/patriziobellan86/PETv1.1/master/PETv1.1-entities.jsonl"
)

NER_TAGS = [
    "O",
    "B-Actor",
    "I-Actor",
    "B-Activity",
    "I-Activity",
    "B-Activity Data",
    "I-Activity Data",
    "B-Further Specification",
    "I-Further Specification",
    "B-XOR Gateway",
    "I-XOR Gateway",
    "B-Condition Specification",
    "I-Condition Specification",
    "B-AND Gateway",
    "I-AND Gateway",
]


@dataclass
class ArmConfig:
    """One selection strategy and the keyword parameters accepted by that strategy."""

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def resolved_label(self) -> str:
        """Display name: explicit label, else strategy plus parameter values."""
        if self.label:
            return self.label
        if not self.params:
            return self.strategy
        values = ",".join(str(self.params[k]) for k in sorted(self.params))
        return f"{self.strategy}:{values}"


def arm_to_dict(arm: ArmConfig) -> dict[str, Any]:
    """Serialize an arm to the same flat shape accepted in YAML."""
    data: dict[str, Any] = {"strategy": arm.strategy, **arm.params}
    if arm.label:
        data["label"] = arm.label
    return data


def parse_arm(raw: dict | str) -> ArmConfig:
    """Build an arm from a flat mapping or a bare strategy name."""
    if isinstance(raw, str):
        return ArmConfig(strategy=raw)
    if not isinstance(raw, dict):
        raise TypeError(f"each arm must be a mapping or strategy name, got {raw!r}")
    params = dict(raw)
    strategy = params.pop("strategy", None)
    if not strategy:
        raise ValueError(f"arm {raw!r} is missing the required 'strategy' key")
    return ArmConfig(strategy=strategy, params=params, label=params.pop("label", None))


@dataclass
class UQConfig:
    """How the seed-trained token classifier grades the unlabelled pool.

    ``bootstrap_epochs=0`` deliberately exposes an untrained-classifier ablation. It
    is valid, but the default trains the randomly initialized PET classification head
    before interpreting its probabilities as task uncertainty.
    """

    bootstrap_epochs: int = 20
    score_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.bootstrap_epochs < 0:
            raise ValueError(f"uq.bootstrap_epochs must be >= 0, got {self.bootstrap_epochs}")
        if self.score_batch_size < 1:
            raise ValueError(f"uq.score_batch_size must be >= 1, got {self.score_batch_size}")


@dataclass
class TrainConfig:
    """Fine-tuning recipe shared by seed fitting and continuation training."""

    checkpoint: str = "distilbert-base-cased"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 256
    eval_batch_size: int = 32
    early_stopping_patience: int = 0
    val_fraction: float = 0.0
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"train.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {self.batch_size}")
        if self.eval_batch_size < 1:
            raise ValueError(f"train.eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if self.max_length < 1:
            raise ValueError(f"train.max_length must be >= 1, got {self.max_length}")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError(f"train.warmup_fraction must be in [0, 1], got {self.warmup_fraction}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"train.early_stopping_patience must be >= 0, got {self.early_stopping_patience}"
            )
        if not 0.0 <= self.val_fraction < 0.5:
            raise ValueError(f"train.val_fraction must be in [0, 0.5), got {self.val_fraction}")
        if bool(self.early_stopping_patience) != bool(self.val_fraction):
            raise ValueError(
                "train.early_stopping_patience and train.val_fraction must be set "
                f"together, got patience={self.early_stopping_patience} and "
                f"val_fraction={self.val_fraction}"
            )

    @property
    def early_stopping(self) -> bool:
        return self.early_stopping_patience > 0 and self.val_fraction > 0


@dataclass
class ExperimentConfig:
    """A seed-trained BERT UQ sweep over budgets, arms, and model seeds."""

    budget_pct: list[float] = field(default_factory=lambda: [10.0])
    arms: list[ArmConfig] = field(
        default_factory=lambda: [
            ArmConfig(strategy="random"),
            ArmConfig(strategy="mean_token_entropy"),
        ]
    )
    tie_seed: int = 0
    n_seed: int = N_SEED_EXAMPLES
    seed_split_seed: int = SEED_SPLIT_SEED
    model_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    uq: UQConfig = field(default_factory=UQConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if isinstance(self.budget_pct, (int, float)):
            self.budget_pct = [float(self.budget_pct)]
        if not self.budget_pct:
            raise ValueError("budget_pct must not be empty")
        bad = [b for b in self.budget_pct if not 0 < b <= 100]
        if bad:
            raise ValueError(f"every budget_pct must be in (0, 100], got {bad}")
        if len(set(self.budget_pct)) != len(self.budget_pct):
            raise ValueError(f"budget_pct must be unique, got {self.budget_pct}")
        self.budget_pct = sorted(float(b) for b in self.budget_pct)

        self.arms = [a if isinstance(a, ArmConfig) else parse_arm(a) for a in self.arms]
        if not self.arms:
            raise ValueError("arms must not be empty")
        labels = self.arm_labels()
        if len(set(labels)) != len(labels):
            raise ValueError(f"arm labels must be unique, got {labels} — set an explicit label")
        if self.n_seed < 1:
            raise ValueError(f"n_seed must be >= 1, got {self.n_seed}")
        if not self.model_seeds:
            raise ValueError("model_seeds must not be empty")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError(f"model_seeds must be unique, got {self.model_seeds}")

    def arm_labels(self) -> list[str]:
        return [arm.resolved_label() for arm in self.arms]


def load_config(path: str | Path) -> ExperimentConfig:
    """Deserialize YAML; unknown keys raise instead of being silently ignored."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    uq = UQConfig(**(raw.pop("uq", None) or {}))
    train = TrainConfig(**(raw.pop("train", None) or {}))
    arms = [parse_arm(a) for a in raw.pop("arms", None) or []] or None
    if arms is not None:
        raw["arms"] = arms
    return ExperimentConfig(uq=uq, train=train, **raw)


def config_to_yaml(cfg: ExperimentConfig) -> str:
    """Serialize a fully resolved config to loadable YAML."""
    data = asdict(cfg)
    data["arms"] = [arm_to_dict(arm) for arm in cfg.arms]
    return yaml.safe_dump(data, sort_keys=False)
