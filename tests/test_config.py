from dataclasses import asdict

import pytest
import yaml

from uq_pet.config import (
    ArmConfig,
    ExperimentConfig,
    TrainConfig,
    UQConfig,
    config_to_yaml,
    load_config,
    parse_arm,
)


def write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_are_the_primary_bert_experiment_contract():
    cfg = ExperimentConfig()
    assert cfg.n_seed == 5
    assert cfg.model_seeds == [0, 1, 2]
    assert cfg.arm_labels() == ["random", "mean_token_entropy"]
    assert cfg.train.checkpoint == "distilbert-base-cased"
    assert cfg.uq.bootstrap_epochs == 20


def test_zero_bootstrap_epochs_is_an_explicit_valid_ablation():
    assert UQConfig(bootstrap_epochs=0).bootstrap_epochs == 0


@pytest.mark.parametrize("value", [-1, -20])
def test_negative_bootstrap_epochs_are_rejected(value):
    with pytest.raises(ValueError, match="bootstrap_epochs"):
        UQConfig(bootstrap_epochs=value)


def test_arm_short_and_long_forms():
    assert parse_arm("margin") == ArmConfig("margin")
    assert parse_arm({"strategy": "confident", "metric": "margin", "label": "easy"}) == (
        ArmConfig("confident", {"metric": "margin"}, "easy")
    )


def test_resolved_label_includes_parameters_when_no_label_is_given():
    assert ArmConfig("confident", {"metric": "margin"}).resolved_label() == "confident:margin"


def test_load_config_reads_bert_uq_fields(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            {
                "budget_pct": [25, 5],
                "arms": ["random", "margin"],
                "n_seed": 10,
                "seed_split_seed": 7,
                "model_seeds": [4],
                "uq": {"bootstrap_epochs": 3, "score_batch_size": 8},
                "train": {"checkpoint": "bert-base-cased", "epochs": 2},
            },
        )
    )
    assert cfg.budget_pct == [5.0, 25.0]
    assert cfg.n_seed == 10
    assert cfg.seed_split_seed == 7
    assert cfg.model_seeds == [4]
    assert cfg.uq == UQConfig(bootstrap_epochs=3, score_batch_size=8)
    assert cfg.train.checkpoint == "bert-base-cased"


def test_snapshot_round_trips(tmp_path):
    expected = ExperimentConfig(
        budget_pct=[7, 14],
        arms=[ArmConfig("random", {"seed": 9}), ArmConfig("margin")],
        model_seeds=[3],
        uq=UQConfig(bootstrap_epochs=0),
        train=TrainConfig(epochs=2),
    )
    path = tmp_path / "resolved.yaml"
    path.write_text(config_to_yaml(expected))
    assert load_config(path) == expected


def test_snapshot_arm_shape_is_flat():
    data = yaml.safe_load(config_to_yaml(ExperimentConfig()))
    assert data["arms"][0] == {"strategy": "random"}
    assert "params" not in data["arms"][0]
    assert set(data) == set(asdict(ExperimentConfig()))


def test_unknown_config_key_raises(tmp_path):
    with pytest.raises(TypeError):
        load_config(write_config(tmp_path, {"grader": "bert"}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_pct": []},
        {"budget_pct": [0]},
        {"budget_pct": [10, 10]},
        {"arms": []},
        {"n_seed": 0},
        {"model_seeds": []},
        {"model_seeds": [0, 0]},
    ],
)
def test_invalid_experiment_shapes_raise(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_duplicate_resolved_arm_labels_raise():
    with pytest.raises(ValueError, match="labels must be unique"):
        ExperimentConfig(arms=[ArmConfig("random"), ArmConfig("random")])
