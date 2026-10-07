import pytest

from experiments.config import RunConfig, load_config


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
    assert config.settings() == {"task": "end-to-end", "provider": "nhr", "model": "m", "reasoning_effort": "low"}


@pytest.mark.parametrize("text, message", [
    ("model: m", "'provider' is required"),
    ("provider: nhr", "'model' is required"),
    ("provider: azure\nmodel: m", "unknown provider"),
    ("provider: nhr\nmodel: m\nreasoning: high", "unknown keys"),
    ("provider: nhr\nmodel: m\nreasoning_effort: max", "reasoning_effort"),
    ("provider: nhr\nmodel: m\ntask: re", "task must be one of"),
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


def test_shipped_configs_are_valid():
    import pathlib
    paths = sorted((pathlib.Path(__file__).parent.parent / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)


def test_none_is_the_reasoning_effort_not_a_missing_value(write_config):
    # YAML reads null and ~ as missing, but "none" stays text
    assert load_config(write_config("provider: openai\nmodel: m\nreasoning_effort: none")).reasoning_effort == "none"
    assert load_config(write_config("provider: openai\nmodel: m\nreasoning_effort: null")).reasoning_effort is None


def test_comments_and_quotes(write_config):
    config = load_config(write_config('# a comment\nprovider: "nhr"  # inline\nmodel: \'org/Model-1\'\n'))
    assert (config.provider, config.model) == ("nhr", "org/Model-1")
