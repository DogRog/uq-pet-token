import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "bert_token_uq.py"
NOTEBOOK_SPEC = importlib.util.spec_from_file_location("bert_token_uq", NOTEBOOK_PATH)
assert NOTEBOOK_SPEC is not None and NOTEBOOK_SPEC.loader is not None
notebook = importlib.util.module_from_spec(NOTEBOOK_SPEC)
NOTEBOOK_SPEC.loader.exec_module(notebook)

_, wandb_defs = notebook.wandb_logging.run()
configure_wandb_metrics = wandb_defs["configure_wandb_metrics"]
make_wandb_comparison_charts = wandb_defs["make_wandb_comparison_charts"]
make_wandb_evaluation_log = wandb_defs["make_wandb_evaluation_log"]


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
        [result_row("uncertainty", 1, 0.8), result_row("random", 1, 0.7)]
    )

    assert payload["evaluation/round"] == 1
    assert payload["evaluation/token_accuracy/uncertainty"] == 0.8
    assert payload["evaluation/token_accuracy/random"] == 0.7
    assert "evaluation/token_accuracy" not in payload
    assert "evaluation/arm" not in payload


def test_wandb_evaluation_log_rejects_a_half_finished_round():
    with pytest.raises(ValueError, match="exactly one row per arm"):
        make_wandb_evaluation_log([result_row("uncertainty", 1, 0.8)])


def test_wandb_metric_configuration_hides_bookkeeping_and_uses_acquisition_x_axis():
    calls = []
    configure_wandb_metrics(
        SimpleNamespace(define_metric=lambda *args, **kwargs: calls.append((args, kwargs)))
    )

    assert (("evaluation/round",), {"hidden": True}) in calls
    assert (("evaluation/n_new/random",), {"hidden": True}) in calls
    assert (
        ("evaluation/token_accuracy/uncertainty",),
        {"step_metric": "evaluation/percent_acquired"},
    ) in calls


def test_wandb_comparison_chart_contains_both_arm_lines():
    calls = []

    def line_series(**kwargs):
        calls.append(kwargs)
        return kwargs

    records = [
        result_row(arm, round_idx, accuracy)
        for arm, accuracies in (("uncertainty", (0.6, 0.8)), ("random", (0.6, 0.7)))
        for round_idx, accuracy in enumerate(accuracies)
    ]
    charts = make_wandb_comparison_charts(
        SimpleNamespace(plot=SimpleNamespace(line_series=line_series)), records
    )

    accuracy_chart = charts["final_comparison/seed_0_token_accuracy"]
    assert accuracy_chart["xs"] == [[0.0, 10.0], [0.0, 10.0]]
    assert accuracy_chart["ys"] == [[0.6, 0.8], [0.6, 0.7]]
    assert accuracy_chart["keys"] == ["Uncertainty", "Random"]
    assert accuracy_chart["split_table"] is True
    assert len(calls) == 3
