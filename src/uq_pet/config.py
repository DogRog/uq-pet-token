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
# in pyproject.toml.
#
# SEED and TEST_SIZE are deliberately NOT YAML knobs: changing one silently
# invalidates every cached record without changing the cache filename. If you must
# change one, delete data/processed/llm_scores/*.jsonl in the same commit.
#
# The two few-shot values *are* overridable — they are the defaults of
# `ExperimentConfig.n_few_shot` / `few_shot_seed`. That is safe only because a
# non-default pair adds a tag to the cache filename (see `few_shot_tag`), so a run
# with different few-shot examples gets its own cache instead of poisoning this one.
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


@dataclass
class ArmConfig:
    """One arm of the comparison: a selection strategy plus that strategy's parameters.

    Every metric variant is registered under its own name, so an arm is usually just
    that name:

        arms:
          - random
          - avg_neg_logprob_filtered
          - vote_entropy

    The long form is a flat mapping, for when a metric takes parameters or the arm
    needs a display name — `strategy` names a metric registered in `uncertainty`
    ("random" is the control, and is one of them), an optional `label` renames it, and
    every remaining key is a parameter of that metric:

        arms:
          - strategy: random
            seed: 7
          - strategy: avg_neg_logprob_pure
            label: anlp

    Which parameters a strategy accepts is defined by the metric function itself, so
    that validation lives in `uncertainty.validate_arm` and this module stays free of
    package imports. `main()` calls it before any work begins.
    """

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def resolved_label(self) -> str:
        """Display name: explicit label, else strategy plus its parameter values."""
        if self.label:
            return self.label
        if not self.params:
            return self.strategy
        values = ",".join(str(self.params[k]) for k in sorted(self.params))
        return f"{self.strategy}:{values}"


def arm_to_dict(arm: ArmConfig) -> dict[str, Any]:
    """Serialize back to the flat YAML form that `load_config` accepts.

    dataclasses.asdict would emit the nested {strategy, params, label} shape instead,
    which load_config rejects — that would silently break every run's config.yaml
    snapshot and the round-trip it is meant to support.
    """
    data: dict[str, Any] = {"strategy": arm.strategy, **arm.params}
    if arm.label:
        data["label"] = arm.label
    return data


def parse_arm(raw: dict | str) -> ArmConfig:
    """Build an ArmConfig from one flat YAML mapping (or a bare strategy name)."""
    if isinstance(raw, str):
        return ArmConfig(strategy=raw)
    if not isinstance(raw, dict):
        raise TypeError(f"each arm must be a mapping or a strategy name, got {raw!r}")

    params = dict(raw)
    strategy = params.pop("strategy", None)
    if not strategy:
        raise ValueError(f"arm {raw!r} is missing the required 'strategy' key")
    label = params.pop("label", None)
    return ArmConfig(strategy=strategy, params=params, label=label)


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

    def cache_path(self, split: str = "pool", tag: str = "") -> Path:
        """Cache file for one split. `tag` separates caches whose split differs.

        It is empty for the default few-shot settings, so the existing cache keeps its
        name; see `ExperimentConfig.few_shot_tag`.
        """
        return LLM_SCORES_DIR / f"{self.cache_prefix}_{split}{tag}_{self.cache_suffix}.jsonl"


@dataclass
class TrainConfig:
    """Fixed fine-tuning recipe — identical for both arms and every seed.

    `epochs` is a *fixed* budget by default, which is the cheap and reproducible
    choice but bakes one stopping point into every cell. Set
    `early_stopping_patience` (with `val_fraction`) to let each cell stop on its own
    instead; `epochs` then becomes the ceiling. The validation split comes out of the
    selected sentences, not out of the test set and not out of the unselected pool —
    an active-learning run only ever has its own budget to spend, and the test split
    has to stay untouched to keep the reported numbers honest.
    """

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
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError(f"train.warmup_fraction must be in [0, 1], got {self.warmup_fraction}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"train.early_stopping_patience must be >= 0, got {self.early_stopping_patience}"
            )
        if not 0.0 <= self.val_fraction < 0.5:
            raise ValueError(f"train.val_fraction must be in [0, 0.5), got {self.val_fraction}")
        # Either knob alone is a silent no-op: a validation split nothing reads, or a
        # patience with no signal to be patient about.
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
    """A budget sweep x arms x training seeds.

    Every arm gets the same number of sentences at a given budget; the only
    experimental variable is *which* sentences. `train_seeds` exists because a
    ~32-sentence fine-tune is noisy enough that a single-seed gap is not a result.

    Note the seeds vary training, not selection: all seeds share one selected set per
    (budget, arm), so the error bars cover training noise but not selection noise.

    `n_few_shot` / `few_shot_seed` choose the demonstrations in the LLM prompt. They
    also decide which sentences are held out of the pool, so changing either changes
    both the prompt and the meaning of every cache index — which is why a non-default
    pair writes to its own cache file (`few_shot_tag`).
    """

    budget_pct: list[float] = field(default_factory=lambda: [10.0])
    arms: list[ArmConfig] = field(
        default_factory=lambda: [
            ArmConfig(strategy="random"),
            ArmConfig(strategy="avg_neg_logprob_filtered"),
        ]
    )
    tie_seed: int = 0
    n_few_shot: int = N_FEW_SHOT_EXAMPLES
    few_shot_seed: int = FEW_SHOT_SPLIT_SEED
    train_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    llm: LLMConfig = field(default_factory=LLMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # A bare scalar is a natural way to write a single budget; accept it.
        if isinstance(self.budget_pct, (int, float)):
            self.budget_pct = [float(self.budget_pct)]
        if not self.budget_pct:
            raise ValueError("budget_pct must not be empty")
        bad = [b for b in self.budget_pct if not 0 < b <= 100]
        if bad:
            raise ValueError(f"every budget_pct must be in (0, 100], got {bad}")
        if len(set(self.budget_pct)) != len(self.budget_pct):
            raise ValueError(f"budget_pct must be unique, got {self.budget_pct}")
        # Sorted so a learning curve's x axis is monotone whatever the file's order.
        self.budget_pct = sorted(float(b) for b in self.budget_pct)

        self.arms = [a if isinstance(a, ArmConfig) else parse_arm(a) for a in self.arms]
        if not self.arms:
            raise ValueError("arms must not be empty")
        labels = [a.resolved_label() for a in self.arms]
        if len(set(labels)) != len(labels):
            raise ValueError(f"arm labels must be unique, got {labels} — set an explicit label")

        if self.n_few_shot < 1:
            raise ValueError(f"n_few_shot must be >= 1, got {self.n_few_shot}")
        if not self.train_seeds:
            raise ValueError("train_seeds must not be empty")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError(f"train_seeds must be unique, got {self.train_seeds}")

    def arm_labels(self) -> list[str]:
        return [a.resolved_label() for a in self.arms]

    def few_shot_tag(self) -> str:
        """Cache-filename tag for a non-default few-shot split, "" for the default.

        The score cache is keyed by position in the pool, and the pool is whatever the
        few-shot split leaves behind — so two runs with different `n_few_shot` or
        `few_shot_seed` must not share a file. Returning "" for the defaults is what
        keeps the existing nhr_gemma_pool_dist.jsonl reachable.
        """
        if (self.n_few_shot, self.few_shot_seed) == (N_FEW_SHOT_EXAMPLES, FEW_SHOT_SPLIT_SEED):
            return ""
        return f"_fs{self.n_few_shot}s{self.few_shot_seed}"


def load_config(path: str | Path) -> ExperimentConfig:
    """Deserialize a YAML file into an ExperimentConfig; unknown keys raise TypeError."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    llm = LLMConfig(**(raw.pop("llm", None) or {}))
    train = TrainConfig(**(raw.pop("train", None) or {}))
    arms = [parse_arm(a) for a in raw.pop("arms", None) or []] or None
    if arms is not None:
        raw["arms"] = arms
    return ExperimentConfig(llm=llm, train=train, **raw)


def config_to_yaml(cfg: ExperimentConfig) -> str:
    data = asdict(cfg)
    data["arms"] = [arm_to_dict(a) for a in cfg.arms]  # asdict would emit the nested form
    return yaml.safe_dump(data, sort_keys=False)
