import pytest
import yaml

from uq_pet.config import (
    CONFIGS_DIR,
    ExperimentConfig,
    LLMConfig,
    TrainConfig,
    config_to_yaml,
    load_config,
    project_root,
)


def test_project_root_is_repo_root():
    """Regression lock: this was parents[1], which resolved to <repo>/src."""
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "uq_pet").is_dir()


def test_project_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UQ_PET_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


# --- Cache-identity locks -----------------------------------------------------
# These two pin LLMConfig's defaults to the 328-record cache that already exists.


def test_default_cache_path_matches_existing_cache():
    assert LLMConfig().cache_path("pool").name == "nhr_gemma_pool_dist.jsonl"


def test_default_sampling_params_match_cached_params():
    assert LLMConfig().sampling_params() == {
        "temperature": 1.0,
        "seed": 0,
        "n": 5,
        "max_tokens": 256,
        "logprobs": True,
    }


def test_cache_path_varies_by_split_and_suffix():
    cfg = LLMConfig(cache_prefix="x", cache_suffix="vote")
    assert cfg.cache_path("test").name == "x_test_vote.jsonl"


def test_top_logprobs_appears_only_when_set():
    assert "top_logprobs" not in LLMConfig().sampling_params()
    assert LLMConfig(top_logprobs=3).sampling_params()["top_logprobs"] == 3


# --- Loading ------------------------------------------------------------------


def _write(tmp_path, data) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_load_config_round_trip(tmp_path):
    cfg = ExperimentConfig(budget_pct=25, train_seeds=[7])
    path = tmp_path / "rt.yaml"
    path.write_text(config_to_yaml(cfg))
    assert load_config(path) == cfg


def test_load_config_nested_partial_override(tmp_path):
    path = _write(tmp_path, {"budget_pct": 5, "train": {"epochs": 2}})
    cfg = load_config(path)
    assert cfg.budget_pct == 5
    assert cfg.train.epochs == 2
    assert cfg.train.checkpoint == TrainConfig().checkpoint  # untouched fields keep defaults
    assert cfg.llm == LLMConfig()


def test_load_config_unknown_key_raises(tmp_path):
    with pytest.raises(TypeError):
        load_config(_write(tmp_path, {"budgets": [10, 25]}))


def test_load_config_unknown_nested_key_raises(tmp_path):
    with pytest.raises(TypeError):
        load_config(_write(tmp_path, {"llm": {"backend": "openrouter"}}))


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    assert isinstance(load_config(path), ExperimentConfig)


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"temperature": -1.0},
        {"workers": 0},
        {"max_retries": 0},
        {"limit": 0},
        {"logprobs": False, "top_logprobs": 3},
    ],
)
def test_llm_config_validation(kwargs):
    with pytest.raises(ValueError):
        LLMConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_pct": 0},
        {"budget_pct": 101},
        {"score": "entropy"},
        {"arms": []},
        {"arms": ["uncertainty", "uncertainty"]},
        {"arms": ["magic"]},
        {"train_seeds": []},
        {"train_seeds": [0, 0]},
    ],
)
def test_experiment_config_validation(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"warmup_fraction": 1.5}])
def test_train_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
