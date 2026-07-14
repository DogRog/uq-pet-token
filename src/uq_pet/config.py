"""Shared constants, project paths, and experiment configuration dataclasses."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

SEED = 3407
TEST_SIZE = 0.2
FEW_SHOT_EXAMPLE_INDEX = 12  # index into the pool split used as the prompt example


def project_root() -> Path:
    """Repo root: $UQ_PET_ROOT if set, else two levels up from src/uq_pet/."""
    env = os.environ.get("UQ_PET_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT = "ner_v1"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATASET_PATH = RAW_DATA_DIR / "PETv1.1-entities.jsonl"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LLM_SCORES_DIR = PROCESSED_DATA_DIR / "llm_scores"
RESULTS_DIR = PROJECT_ROOT / "results"

NER_DATASET_URL = (
    "https://raw.githubusercontent.com/patriziobellan86/PETv1.1/master/"
    "PETv1.1-entities.jsonl"
)

NER_TAGS = [
    "O", "B-Actor", "I-Actor", "B-Activity", "I-Activity",
    "B-Activity Data", "I-Activity Data", "B-Further Specification",
    "I-Further Specification", "B-XOR Gateway", "I-XOR Gateway",
    "B-Condition Specification", "I-Condition Specification",
    "B-AND Gateway", "I-AND Gateway"
]

@dataclass
class LLMScoreConfig:
    """Configuration for the LLM repeated-sampling pass over the pool.

    `backend` picks how samples are generated: "openrouter" calls the API
    (text only, black-box metrics); "mlx" (Apple Silicon) and "hf"
    (transformers, CUDA when available) run the model in-process and also
    record per-token predictive entropies (white-box metrics).
    `max_concurrency`/`max_retries` only apply to "openrouter";
    `batch_size` (sentences decoded per batched generate call) only to "hf".
    `prompt` names a template in prompts/<name>.txt; each prompt gets its
    own score cache.
    """

    backend: str = "openrouter"
    model: str = "meta-llama/llama-3-8b-instruct"
    prompt: str = DEFAULT_PROMPT
    num_samples: int = 5
    temperature: float = 0.7
    max_tokens: int = 1500
    seed: int = SEED
    max_concurrency: int = 8
    max_retries: int = 3
    batch_size: int = 8

    def __post_init__(self):
        if self.backend not in ("openrouter", "mlx", "hf"):
            raise ValueError(f"Unknown llm.backend '{self.backend}' (expected 'openrouter', 'mlx' or 'hf')")

    def prompt_path(self) -> Path:
        return PROMPTS_DIR / f"{self.prompt}.txt"

    def cache_path(self) -> Path:
        safe_model = self.model.replace("/", "_")
        name = f"{safe_model}_k{self.num_samples}_t{self.temperature}_seed{self.seed}"
        # The default prompt keeps the historical filename so old caches stay valid.
        if self.prompt != DEFAULT_PROMPT:
            name += f"_{self.prompt}"
        return LLM_SCORES_DIR / f"{name}.jsonl"


@dataclass
class TrainConfig:
    """Fixed fine-tuning recipe used for every grid cell."""

    checkpoint: str = "distilbert-base-cased"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 256


@dataclass
class ExperimentConfig:
    """Full grid: budgets x strategies x seeds.

    `workers` is how many grid cells train concurrently (worker processes
    sharing one GPU); cells are independent and each seeds itself, so results
    do not depend on it. 1 keeps everything in-process.
    """

    budgets: list[int] = field(default_factory=lambda: [10, 25, 50, 100])
    strategies: list[str] = field(default_factory=lambda: [
        "random",
        "uncertainty:mean_token_entropy",
        "uncertainty:sequence_entropy",
        "uncertainty:jaccard_distance",
    ])
    repeats: int = 5
    workers: int = 1
    llm: LLMScoreConfig = field(default_factory=LLMScoreConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Deserialize a YAML file into an ExperimentConfig; unknown keys raise TypeError."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    llm = LLMScoreConfig(**(raw.pop("llm", None) or {}))
    train = TrainConfig(**(raw.pop("train", None) or {}))
    return ExperimentConfig(llm=llm, train=train, **raw)


def config_to_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False)
