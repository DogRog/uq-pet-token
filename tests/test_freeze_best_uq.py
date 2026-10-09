"""Check that tuning winners are frozen into best-UQ configs without training."""

import json

import pytest

import freeze_best_uq


def tune_config(tmp_path, **overrides):
    path = tmp_path / "distilbert_medical_tune_random_100.json"
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
    out = tmp_path / "best_uq"
    out.mkdir()
    assert freeze_best_uq.freeze(tune_config(tmp_path), out) is None
    assert not any(out.iterdir())


def test_winner_is_frozen_and_overwrites_a_stale_config(tmp_path):
    sweep = tmp_path / "sweeps" / "distilbert-medical-random-baseline-100"
    sweep.mkdir(parents=True)
    winner = {"checkpoint": "distilbert-base-cased", "dataset": "medical", "learning_rate": 2e-4}
    (sweep / "best_config.json").write_text(json.dumps(winner))
    out = tmp_path / "best_uq"
    out.mkdir()
    (out / "distilbert_medical.json").write_text('{"learning_rate": 9e-5}')
    path = freeze_best_uq.freeze(tune_config(tmp_path), out)
    assert path == out / "distilbert_medical.json"
    assert json.loads(path.read_text()) == {
        **winner,
        "wandb_enabled": True,
        "wandb_project": "distilbert-medical-best-uq",
        "seed_workers": 5,
    }


def test_test_tuned_oracle_is_never_frozen(tmp_path):
    config = tune_config(tmp_path, tuning_split="test", objective="random_test_entity_f1_auc")
    with pytest.raises(ValueError, match="ADR 0007"):
        freeze_best_uq.freeze(config, tmp_path)


def test_every_tuning_config_maps_to_its_best_uq_name():
    for path in (freeze_best_uq.PROJECT_ROOT / "configs" / "tune_random").glob("*.json"):
        name = path.stem.removesuffix("_tune_random_100")
        assert name != path.stem
        supervised = freeze_best_uq.PROJECT_ROOT / "configs" / "supervised" / f"{name}.json"
        tuning, baseline = json.loads(path.read_text()), json.loads(supervised.read_text())
        assert baseline["checkpoint"] == tuning["checkpoint"]
        assert baseline.get("dataset", "pet") == tuning.get("dataset", "pet")
        assert baseline.get("validation_sentences", 66) == tuning["validation_sentences"]
        assert baseline.get("validation_seed", 1729) == tuning["validation_seed"]


def test_unchanged_winner_keeps_the_file(tmp_path):
    sweep = tmp_path / "sweeps" / "distilbert-medical-random-baseline-100"
    sweep.mkdir(parents=True)
    (sweep / "best_config.json").write_text('{"checkpoint": "distilbert-base-cased"}')
    out = tmp_path / "best_uq"
    out.mkdir()
    saved = (
        '{"seed_workers": 5, "wandb_enabled": true,\n'
        ' "wandb_project": "distilbert-medical-best-uq", "checkpoint": "distilbert-base-cased"}'
    )
    (out / "distilbert_medical.json").write_text(saved)
    freeze_best_uq.freeze(tune_config(tmp_path), out)
    assert (out / "distilbert_medical.json").read_text() == saved
