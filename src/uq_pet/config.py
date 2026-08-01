"""Shared constants, project paths, and experiment configuration dataclasses.

Split of responsibility: anything in this module's *constants* section defines the
identity of the experiment's data; anything in a dataclass is a knob you may set from
a YAML run-config in configs/.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --- Cache-identity constants -------------------------------------------------
# The LLM score cache is keyed by `idx` = position in the *pool* split. That position
# is a function of these four values, the raw PET file, and the exact `datasets` pin
# in pyproject.toml. They are deliberately NOT YAML knobs: changing one silently
# invalidates every cached record without changing the cache filename. If you must
# change one, delete data/processed/llm_scores/*.jsonl in the same commit.
SEED = 3407
TEST_SIZE = 0.2
FEW_SHOT_SPLIT_SEED = 42
N_FEW_SHOT_EXAMPLES = 5


def project_root() -> Path:
    """Repo root: $UQ_PET_ROOT if set, else two levels up from src/uq_pet/config.py."""
    env = os.environ.get("UQ_PET_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATASET_PATH = RAW_DATA_DIR / "PETv1.1-entities.jsonl"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LLM_SCORES_DIR = PROCESSED_DATA_DIR / "llm_scores"
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

ARMS = ("uncertainty", "random")
SCORE_VARIANTS = ("pure", "filtered")


@dataclass
class LLMConfig:
    """One repeated-sampling pass over a split through an OpenAI-compatible gateway.

    The defaults reproduce the existing cache at
    data/processed/llm_scores/nhr_gemma_pool_dist.jsonl — 328 records, K=5 choices
    each with token logprobs.

    Note on cache naming: the filename comes from `cache_prefix`/`cache_suffix`, not
    from the sampling parameters, so editing `temperature` alone would point at a
    stale file. The guard against that is content validation, not naming:
    `llm.load_cache()` drops every record whose `model`/`params` disagree with the
    config in hand. That is strictly stronger than a derived filename — it also
    catches a gateway that silently served a different model — but it does mean you
    should bump `cache_suffix` when you deliberately change the sampling recipe.
    """

    model: str = "RedHatAI/gemma-4-31B-it-FP8-block"
    base_url: str = "https://hub.nhr.fau.de/api/llmgw/v1"
    api_key_env: str = "NHR_FAU_API_KEY"
    cache_prefix: str = "nhr_gemma"
    cache_suffix: str = "dist"
    n_samples: int = 5
    temperature: float = 1.0
    seed: int | None = 0
    max_tokens: int = 256
    logprobs: bool = True
    top_logprobs: int | None = None
    workers: int = 8
    timeout: float = 120.0
    max_retries: int = 5
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"llm.n_samples must be >= 1, got {self.n_samples}")
        if self.temperature < 0:
            raise ValueError(f"llm.temperature must be >= 0, got {self.temperature}")
        if self.workers < 1:
            raise ValueError(f"llm.workers must be >= 1, got {self.workers}")
        if self.max_retries < 1:
            raise ValueError(f"llm.max_retries must be >= 1, got {self.max_retries}")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("llm.top_logprobs requires llm.logprobs to be true")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"llm.limit must be >= 1 or null, got {self.limit}")

    def sampling_params(self) -> dict[str, Any]:
        """The kwargs passed to chat.completions.create, also stored on each record.

        With the defaults this equals the `params` dict in the existing cached
        records exactly, so those records validate without a migration.
        """
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "seed": self.seed,
            "n": self.n_samples,
            "max_tokens": self.max_tokens,
            "logprobs": self.logprobs,
        }
        if self.top_logprobs is not None:
            params["top_logprobs"] = self.top_logprobs
        return params

    def cache_path(self, split: str = "pool") -> Path:
        return LLM_SCORES_DIR / f"{self.cache_prefix}_{split}_{self.cache_suffix}.jsonl"


@dataclass
class TrainConfig:
    """Fixed fine-tuning recipe — identical for both arms and every seed."""

    checkpoint: str = "distilbert-base-cased"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 256
    eval_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"train.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError(
                f"train.warmup_fraction must be in [0, 1], got {self.warmup_fraction}"
            )


@dataclass
class ExperimentConfig:
    """One budget, two arms, N training seeds.

    Every arm gets the same number of sentences (`budget_pct` of the pool); the only
    experimental variable is *which* sentences. `train_seeds` exists because a
    ~32-sentence fine-tune is noisy enough that a single-seed gap is not a result.
    """

    budget_pct: float = 10.0
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    score: str = "filtered"
    selection_seed: int = 42
    tie_seed: int = 0
    train_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    llm: LLMConfig = field(default_factory=LLMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not 0 < self.budget_pct <= 100:
            raise ValueError(f"budget_pct must be in (0, 100], got {self.budget_pct}")
        if self.score not in SCORE_VARIANTS:
            raise ValueError(f"Unknown score '{self.score}' (expected one of {SCORE_VARIANTS})")
        if not self.arms:
            raise ValueError("arms must not be empty")
        unknown = [a for a in self.arms if a not in ARMS]
        if unknown:
            raise ValueError(f"Unknown arms {unknown} (expected a subset of {list(ARMS)})")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError(f"arms must be unique, got {self.arms}")
        if not self.train_seeds:
            raise ValueError("train_seeds must not be empty")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError(f"train_seeds must be unique, got {self.train_seeds}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Deserialize a YAML file into an ExperimentConfig; unknown keys raise TypeError."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    llm = LLMConfig(**(raw.pop("llm", None) or {}))
    train = TrainConfig(**(raw.pop("train", None) or {}))
    return ExperimentConfig(llm=llm, train=train, **raw)


def config_to_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False)
