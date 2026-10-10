import pytest

from experiments.config import RunConfig, load_configs


def load_config(path):
    """The single config of a file without "models"."""
    [config] = load_configs(path)
    return config


@pytest.fixture
def write_config(tmp_path):
    def _write(text, name="my-run"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(text)
        return path
    return _write


def test_minimal_config_uses_defaults(write_config):
    config = load_config(write_config("provider: nhr\nmodel: org/Model-1"))

    assert config == RunConfig(name="my-run", provider="nhr", model="org/Model-1")
    assert (config.reasoning_effort, config.max_documents, config.repeats) == (None, None, 1)


def test_full_config(write_config):
    config = load_config(write_config(
        "provider: openai\nmodel: gpt-6-sol\ntask: md\nreasoning_effort: high\nmax_documents: 5\nrepeats: 3"
    ))

    assert config == RunConfig(
        name="my-run", provider="openai", model="gpt-6-sol", task="md", reasoning_effort="high", max_documents=5,
        repeats=3,
    )


def test_settings_are_what_changes_predictions(write_config):
    config = load_config(write_config("provider: nhr\nmodel: m\nreasoning_effort: low\nrepeats: 2"))
    assert config.settings() == {"dataset": "pet", "task": "end-to-end", "provider": "nhr", "model": "m", "reasoning_effort": "low"}


@pytest.mark.parametrize("text, message", [
    ("model: m", "'provider' is required"),
    ("provider: nhr", "'model' is required"),
    ("provider: azure\nmodel: m", "unknown provider"),
    ("provider: nhr\nmodel: m\nreasoning: high", "unknown keys"),
    ("provider: nhr\nmodel: m\nreasoning_effort: max", "reasoning_effort"),
    ("provider: nhr\nmodel: m\ntask: re", "task must be one of"),
    ("provider: nhr\nmodel: m\ndataset: conll2003", "dataset must be one of"),
    ("provider: nhr\nmodel: m\ndataset: quishpi", "quishpi has no relations"),
    ("provider: nhr\nmodel: m\nmax_documents: 0", "max_documents"),
    ("provider: nhr\nmodel: m\nrepeats: 0", "repeats"),
    ("provider: nhr\nmodel: m\nrepeats: '3'", "repeats"),
    ("provider: nhr\nmodel: 1.5", "'model' must be text"),
    ("", "expected keys"),
    ("- provider: nhr", "expected keys"),
])
def test_invalid_configs(write_config, text, message):
    with pytest.raises(ValueError, match=message):
        load_config(write_config(text))


def test_dataset(write_config):
    config = load_config(write_config("provider: nhr\nmodel: m\ndataset: medical\ntask: md-test-sentences"))
    assert (config.dataset, config.task) == ("medical", "md-test-sentences")
    assert config.settings()["dataset"] == "medical"


def test_shipped_configs_are_valid():
    import pathlib
    paths = sorted((pathlib.Path(__file__).parent.parent / "configs").rglob("*.yaml"))
    assert paths
    configs = {path: load_configs(path) for path in paths}
    # the names are the output folders in res/pred/
    names = [c.name for path_configs in configs.values() for c in path_configs]
    assert len(set(names)) == len(names)
    # configs/<dataset>/<task>.yaml
    for path, path_configs in configs.items():
        if path.parent.name != "configs":
            assert all((c.dataset, c.task) == (path.parent.name, path.stem) for c in path_configs), path


MODELS = """task: md
provider: nhr
repeats: 5
models:
  - name: gemma
    model: org/gemma-1
    reasoning_effort: [none, low]
  - name: qwen
    model: org/qwen-1
    reasoning_effort: high
  - name: magistral
    model: org/magistral-1
"""


def test_one_config_per_model_and_reasoning_effort(write_config):
    configs = load_configs(write_config(MODELS, name="many"))

    assert [(c.name, c.model, c.reasoning_effort) for c in configs] == [
        ("gemma-md", "org/gemma-1", "none"),
        ("gemma-md-reasoning-low", "org/gemma-1", "low"),
        ("qwen-md-reasoning-high", "org/qwen-1", "high"),
        ("magistral-md-reasoning-default", "org/magistral-1", None),
    ]
    assert all((c.task, c.provider, c.repeats) == ("md", "nhr", 5) for c in configs)


def test_names_start_with_the_dataset_unless_pet(write_config):
    text = MODELS.replace("task: md", "dataset: quishpi\ntask: md-test-sentences")
    names = [c.name for c in load_configs(write_config(text))]
    assert names[:2] == ["quishpi-gemma-md-test-sentences", "quishpi-gemma-md-test-sentences-reasoning-low"]


@pytest.mark.parametrize("models, message", [
    ("models: []", "must be a list"),
    ("models:\n  - model: m", "'name' is required"),
    ("models:\n  - name: a\n    reasoning_effort: low", "models entry 1: 'model' is required"),
    ("models:\n  - name: a\n    model: m\n    reasoning_effort: []", "empty list"),
    ("models:\n  - name: a\n    model: m\n    reasoning_effort: [low, max]", "models entry 1: reasoning_effort"),
    ("models:\n  - name: a\n    model: m\n    task: md", "unknown keys"),
    ("model: m\nmodels:\n  - name: a\n    model: m", "per entry"),
])
def test_invalid_models(write_config, models, message):
    with pytest.raises(ValueError, match=message):
        load_configs(write_config("provider: nhr\n" + models))


def test_none_is_the_reasoning_effort_not_a_missing_value(write_config):
    # YAML reads null and ~ as missing, but "none" stays text
    assert load_config(write_config("provider: openai\nmodel: m\nreasoning_effort: none")).reasoning_effort == "none"
    assert load_config(write_config("provider: openai\nmodel: m\nreasoning_effort: null")).reasoning_effort is None


def test_comments_and_quotes(write_config):
    config = load_config(write_config('# a comment\nprovider: "nhr"  # inline\nmodel: \'org/Model-1\'\n'))
    assert (config.provider, config.model) == ("nhr", "org/Model-1")
