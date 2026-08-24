from pathlib import Path
from types import SimpleNamespace

import pytest

import uq_pet.experiment as experiment
from uq_pet.experiment import (
    ExperimentConfig,
    configure_wandb_metrics,
    execute_experiment,
    make_wandb_evaluation_log,
)


def result_row(arm, round_idx, accuracy):
    return {
        "seed": 0,
        "arm": arm,
        "round": round_idx,
        "total_rounds": 1,
        "n_acquired": round_idx * 16,
        "percent_acquired": round_idx * 10.0,
        "scoreable_pool_tokens": 160,
        "token_budget": 16,
        "entity_f1": accuracy - 0.1,
        "entity_precision": accuracy - 0.05,
        "entity_recall": accuracy - 0.15,
        "token_accuracy": accuracy,
        "train_loss": 1.0 - accuracy,
        "n_new": 0 if round_idx == 0 else 16,
        "n_replay": 0 if round_idx == 0 else 16,
    }


def test_wandb_evaluation_log_separates_arm_metrics_in_one_step():
    payload = make_wandb_evaluation_log(
        [result_row("uncertainty", 1, 0.8), result_row("random", 1, 0.7)],
        "least_confidence",
    )

    assert payload["evaluation/round"] == 1
    assert payload["evaluation/token_accuracy/least_confidence"] == 0.8
    assert payload["evaluation/token_accuracy/random"] == 0.7
    assert "evaluation/token_accuracy" not in payload
    assert "evaluation/arm" not in payload


def test_wandb_evaluation_log_rejects_a_half_finished_round():
    with pytest.raises(ValueError, match="exactly one row per arm"):
        make_wandb_evaluation_log([result_row("uncertainty", 1, 0.8)], "entropy")


def test_wandb_metric_configuration_hides_bookkeeping_and_uses_acquisition_x_axis():
    calls = []
    configure_wandb_metrics(
        SimpleNamespace(define_metric=lambda *args, **kwargs: calls.append((args, kwargs))),
        "least_confidence",
    )

    assert (("evaluation/round",), {"hidden": True}) in calls
    assert (("evaluation/n_new/random",), {"hidden": True}) in calls
    assert (
        ("evaluation/token_accuracy/least_confidence",),
        {"step_metric": "evaluation/percent_acquired"},
    ) in calls


def test_wandb_comparison_uses_one_html_panel_and_no_table_per_seed(monkeypatch):
    records = [
        result_row(arm, round_idx, accuracy)
        for arm, accuracies in (("uncertainty", (0.6, 0.8)), ("random", (0.6, 0.7)))
        for round_idx, accuracy in enumerate(accuracies)
    ]
    chart_calls = []
    html_calls = []

    def make_learning_chart(result_records, seed, uq_metric):
        chart_calls.append((result_records, seed, uq_metric))
        return SimpleNamespace(to_html=lambda: "<html>comparison</html>")

    def html(data, *, inject):
        html_calls.append((data, inject))
        return "html-media"

    monkeypatch.setattr(experiment, "make_learning_chart", make_learning_chart)
    media = experiment.make_wandb_comparison_media(
        SimpleNamespace(Html=html), records, "least_confidence"
    )

    assert media == {"final_comparison/seed_0_least_confidence_vs_random": "html-media"}
    assert chart_calls == [(records, 0, "least_confidence")]
    assert html_calls == [("<html>comparison</html>", False)]


def test_execute_experiment_uses_shared_config_and_persists_derived_settings(monkeypatch, tmp_path):
    config = ExperimentConfig(model_seeds=0, max_pool_percent=50, wandb_enabled=True)
    captured = {}
    rows = [
        {
            **result_row("uncertainty", 0, 0.6),
            "scoreable_pool_tokens": 200,
            "token_budget": 100,
            "total_rounds": 5,
        },
        {
            **result_row("random", 0, 0.6),
            "scoreable_pool_tokens": 200,
            "token_budget": 100,
            "total_rounds": 5,
        },
    ]

    monkeypatch.setattr(experiment, "download_pet_ner", lambda: Path("pet.jsonl"))
    monkeypatch.setattr(
        experiment,
        "load_pet_splits",
        lambda path: ([{"tokens": ["seed"]}], [{"tokens": ["pool"]}], {(0, 0): 0}, []),
    )
    monkeypatch.setattr(experiment, "get_device", lambda: "cpu")

    def run_active_learning(*args, **kwargs):
        captured["active_learning_kwargs"] = kwargs
        return rows, [{"token": "pool"}]

    def write_run(run_config, results, selections, *, results_dir):
        captured["run_config"] = run_config
        captured["results_dir"] = results_dir
        return results_dir / "run"

    monkeypatch.setattr(experiment, "run_active_learning", run_active_learning)
    monkeypatch.setattr(experiment, "write_run", write_run)

    run_config, summary, results, selections, run_dir = execute_experiment(
        config, results_dir=tmp_path
    )

    assert "wandb_enabled" not in captured["active_learning_kwargs"]
    assert captured["active_learning_kwargs"]["model_seeds"] == [0]
    assert captured["run_config"]["effective_pool_percent"] == 50
    assert summary["acquisition_rounds"] == 5
    assert results == rows
    assert selections == [{"token": "pool"}]
    assert run_dir == tmp_path / "run"
