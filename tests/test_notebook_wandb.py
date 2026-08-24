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
make_wandb_comparison_media = wandb_defs["make_wandb_comparison_media"]
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


def test_wandb_comparison_uses_one_html_panel_and_no_table_per_seed():
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

    media = make_wandb_comparison_media(
        SimpleNamespace(Html=html), make_learning_chart, records, "least_confidence"
    )

    assert media == {"final_comparison/seed_0_least_confidence_vs_random": "html-media"}
    assert chart_calls == [(records, 0, "least_confidence")]
    assert html_calls == [("<html>comparison</html>", False)]
