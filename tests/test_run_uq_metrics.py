"""Check multi-metric dispatch without starting model training."""

import json
from types import SimpleNamespace

import pytest

import run_uq_metrics as runner


def test_dispatch_preserves_fixed_settings_and_metric_order(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert (
        runner.main(
            [
                "--config-json",
                '{"checkpoint":"bert-base-cased","learning_rate":2.658981434590006e-05}',
                "--uq-metrics",
                "margin",
                "entropy",
            ]
        )
        == 0
    )
    configs = [json.loads(command[-1]) for command, _ in calls]
    assert [config["uq_metric"] for config in configs] == ["margin", "entropy"]
    assert all(config["checkpoint"] == "bert-base-cased" for config in configs)
    assert all(config["learning_rate"] == 2.658981434590006e-05 for config in configs)
    assert all(kwargs["cwd"] == runner.PROJECT_ROOT for _, kwargs in calls)


def test_failure_stops_remaining_runs(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["--config-json", "{}"]) == 7
    assert len(calls) == 1


def test_dry_run_and_invalid_settings_never_launch(monkeypatch, capsys):
    def unexpected_run(*args, **kwargs):
        pytest.fail("Must not start training")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)
    config_path = runner.PROJECT_ROOT / "configs" / "best_uq" / "bert_base.json"
    assert runner.main(["--config", str(config_path), "--dry-run"]) == 0
    configs = json.loads(capsys.readouterr().out)
    assert [config["uq_metric"] for config in configs] == list(runner.UQ_METRICS)
    for args in (
        ["--config-json", '{"k":0}'],
        ["--config-json", "[]"],
        ["--config-json", "{}", "--uq-metrics", "entropy", "entropy"],
    ):
        with pytest.raises(SystemExit) as error:
            runner.main(args)
        assert error.value.code == 2
