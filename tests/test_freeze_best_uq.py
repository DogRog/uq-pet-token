"""Check that tuning winners are frozen into best-UQ configs without training."""

import json

import pytest

import freeze_best_uq


def tune_config(tmp_path, **overrides):
    path = tmp_path / "medical" / "tune_random" / "distilbert.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "checkpoint": "distilbert-base-cased",
        "dataset": "medical",
        "sweep_name": "distilbert-medical-random-baseline-100",
        "sweeps_dir": str(tmp_path / "sweeps"),
        "wandb_enabled": True,
        "wandb_project": "distilbert-medical-random-baseline",
        "mode": "tune-random",
        "num_configs": 100,
        "validation_sentences": 51,
        **overrides,
    }
    path.write_text(json.dumps(settings))
    return path


def test_missing_winner_writes_nothing(tmp_path):
    assert freeze_best_uq.freeze(tune_config(tmp_path)) is None
    assert not (tmp_path / "medical" / "best_uq").exists()


def test_winner_is_frozen_and_overwrites_a_stale_config(tmp_path):
    sweep = tmp_path / "sweeps" / "distilbert-medical-random-baseline-100"
    sweep.mkdir(parents=True)
    winner = {"checkpoint": "distilbert-base-cased", "dataset": "medical", "learning_rate": 2e-4}
    (sweep / "best_config.json").write_text(json.dumps(winner))
    out = tmp_path / "medical" / "best_uq"
    out.mkdir(parents=True)
    (out / "distilbert.json").write_text('{"learning_rate": 9e-5}')
    path = freeze_best_uq.freeze(tune_config(tmp_path))
    assert path == out / "distilbert.json"
    assert json.loads(path.read_text()) == {
        **winner,
        "wandb_enabled": True,
        "wandb_project": "distilbert-medical-best-uq",
        "seed_workers": 5,
    }


def test_test_tuned_oracle_is_never_frozen(tmp_path):
    config = tune_config(tmp_path, tuning_split="test", objective="random_test_entity_f1_auc")
    with pytest.raises(ValueError, match="ADR 0007"):
        freeze_best_uq.freeze(config)


def test_only_tuning_configs_are_frozen(tmp_path):
    config = tune_config(tmp_path)
    elsewhere = config.rename(tmp_path / "distilbert.json")
    with pytest.raises(ValueError, match="tune_random"):
        freeze_best_uq.freeze(elsewhere)


def test_each_dataset_folder_holds_matching_configs():
    configs = freeze_best_uq.PROJECT_ROOT / "configs"
    for dataset in ("pet", "quishpi", "medical"):
        tuning_paths = sorted((configs / dataset / "tune_random").glob("*.json"))
        assert len(tuning_paths) == 5
        for stage in ("supervised", "random_search"):
            names = sorted(path.name for path in (configs / dataset / stage).glob("*.json"))
            assert names == [path.name for path in tuning_paths]
            for path in (configs / dataset / stage).glob("*.json"):
                assert json.loads(path.read_text()).get("dataset", "pet") == dataset
    for path in configs.glob("*/tune_random/*.json"):
        dataset = path.parent.parent.name
        supervised = path.parent.parent / "supervised" / path.name
        tuning, baseline = json.loads(path.read_text()), json.loads(supervised.read_text())
        assert tuning.get("dataset", "pet") == dataset
        assert baseline["checkpoint"] == tuning["checkpoint"]
        assert baseline.get("validation_sentences", 66) == tuning["validation_sentences"]
        assert baseline.get("validation_seed", 1729) == tuning["validation_seed"]


def test_unchanged_winner_keeps_the_file(tmp_path):
    sweep = tmp_path / "sweeps" / "distilbert-medical-random-baseline-100"
    sweep.mkdir(parents=True)
    (sweep / "best_config.json").write_text('{"checkpoint": "distilbert-base-cased"}')
    out = tmp_path / "medical" / "best_uq"
    out.mkdir(parents=True)
    saved = (
        '{"seed_workers": 5, "wandb_enabled": true,\n'
        ' "wandb_project": "distilbert-medical-best-uq", "checkpoint": "distilbert-base-cased"}'
    )
    (out / "distilbert.json").write_text(saved)
    freeze_best_uq.freeze(tune_config(tmp_path))
    assert (out / "distilbert.json").read_text() == saved
