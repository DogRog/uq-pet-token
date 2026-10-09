"""Check fixed multi-metric execution without starting model training."""

import json

import pytest

import run_uq_metrics as runner


def test_fixed_plan_preserves_every_winner(monkeypatch, capsys):
    monkeypatch.setattr(runner.search.random, "Random", lambda *_: pytest.fail("Must not sample"))
    for path in sorted((runner.PROJECT_ROOT / "configs" / "best_uq").glob("*.json")):
        assert runner.main(["--config", str(path), "--dry-run"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["sampling"] == "none"
        assert plan["num_configs"] == len(plan["configurations"]) == 1
        assert plan["uq_metrics"] == list(runner.UQ_METRICS)
        actual = {**plan["fixed_config"], **plan["configurations"][0]["parameters"]}
        for key, value in json.loads(path.read_text()).items():
            if key not in {"uq_metric", "seed_workers", *runner.search.WANDB_FIELDS}:
                assert actual[key] == value


def test_shared_execution_saves_grouped_results_and_resumes(monkeypatch, tmp_path):
    search = runner.search
    monkeypatch.setattr(search, "get_device", lambda: "cpu")
    monkeypatch.setattr(search, "load_splits", lambda *_: (([{}], [{}], {}, [{}]), {}))
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        output = {}
        for metric in kwargs["uq_metrics"]:
            rows = [
                dict(
                    seed=seed,
                    round=r,
                    total_rounds=1,
                    percent_acquired=r * 100,
                    arm=arm,
                    entity_f1=0.5,
                )
                for seed in kwargs["model_seeds"]
                for r in range(2)
                for arm in ("uncertainty", "random")
            ]
            kwargs["progress_callback"](metric, rows)
            output[metric] = (rows, [{"token": "selected"}])
        return output

    monkeypatch.setattr(search, "run_metric_comparisons", run)
    args = [
        "--config-json",
        '{"checkpoint":"bert-base-cased","k":19}',
        "--sweeps-dir",
        str(tmp_path),
        "--uq-metrics",
        "margin",
        "entropy",
    ]
    assert runner.main(args) == 0
    assert len(calls) == 1
    assert calls[0]["seed_workers"] == 2
    assert calls[0]["k"] == 19
    assert calls[0]["uq_metrics"] == ["margin", "entropy"]
    assert not calls[0]["random_only"]
    root = tmp_path / "bert-base-cased-best-uq"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    for metric in ("margin", "entropy"):
        slot = root / "runs" / "config_0000" / metric
        completed = json.loads((slot / "completed.json").read_text())
        for name in ("config.json", "results.csv", "selections.json"):
            assert (root / completed["run_dir"] / name).is_file()
    monkeypatch.setattr(search, "load_splits", lambda *_: pytest.fail("Resume loaded data"))
    assert runner.main([*args, "--seed-workers", "3"]) == 0
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        runner.main([*args[:1], '{"checkpoint":"bert-base-cased","k":20}', *args[2:]])


@pytest.mark.parametrize(
    "args",
    [
        ["--config-json", '{"k":0}'],
        ["--config-json", "[]"],
        ["--config-json", "{}", "--uq-metrics", "entropy", "entropy"],
        ["--config-json", "{}", "--seed-workers", "0"],
    ],
)
def test_invalid_settings_never_launch(monkeypatch, args):
    monkeypatch.setattr(runner.search, "load_splits", lambda *_: pytest.fail("Must not train"))
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2


def test_fixed_mode_rejects_multiple_configurations():
    with pytest.raises(ValueError):
        runner.search.FixedComparisonConfig(num_configs=2)


def test_acquisition_flags_enter_the_plan_and_defaults_stay_implicit(capsys):
    config = str(runner.PROJECT_ROOT / "configs" / "best_uq" / "distilbert.json")
    assert runner.main(["--config", config, "--dry-run"]) == 0
    plain = json.loads(capsys.readouterr().out)["fixed_config"]
    assert "gumbel_noise" not in plain and "max_per_word_form" not in plain
    args = ["--config", config, "--gumbel-noise", "--max-per-word-form", "1", "--dry-run"]
    assert runner.main(args) == 0
    fixed = json.loads(capsys.readouterr().out)["fixed_config"]
    assert fixed == {**plain, "gumbel_noise": True, "max_per_word_form": 1}


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({}, "distilbert-base-cased-best-uq"),
        ({"dataset": "medical"}, "distilbert-base-cased-medical-best-uq"),
        (
            {"dataset": "quishpi", "gumbel_noise": True},
            "distilbert-base-cased-quishpi-best-uq-gumbel",
        ),
        ({"dataset_percent": 12.5}, "distilbert-base-cased-12p5pct-best-uq"),
    ],
)
def test_default_result_group_names_the_dataset(monkeypatch, settings, name):
    seen = {}
    monkeypatch.setattr(
        runner.search, "run", lambda args, _: seen.update(json.loads(args.config_json))
    )
    assert runner.main(["--config-json", json.dumps(settings), "--dry-run"]) == 0
    assert seen["sweep_name"] == name
