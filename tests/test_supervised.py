"""Verify the supervised grid, validation-only tuning, test isolation, and resume."""

import argparse
import json

import polars as pl
import pytest

from uq_pet import supervised
from uq_pet.config import SupervisedConfig
from utils.charts import make_variance_chart


def invoke(config, *extra):
    parser = argparse.ArgumentParser()
    supervised.configure_parser(parser)
    supervised.run(parser.parse_args(["--config-json", config.model_dump_json(), *extra]), parser)


def test_grid_is_full_factorial_ordered_and_named_by_checkpoint():
    config = SupervisedConfig(checkpoint="microsoft/deberta-v3-base")
    plan = supervised.grid_plan(config)
    assert config.sweep_name == "deberta-v3-base-supervised"
    assert len(plan["configurations"]) == 5 * 3 * 2
    assert plan["configurations"][0] == {
        "config_id": "config_0000",
        "parameters": {"learning_rate": 1e-5, "batch_size": 8, "weight_decay": 0.0},
    }
    parameters = [json.dumps(entry["parameters"]) for entry in plan["configurations"]]
    assert len(set(parameters)) == len(parameters)
    assert plan == supervised.grid_plan(config)
    assert "sentence_percents" not in json.dumps(plan)
    for bad in ({"sentence_percents": [100, 50]}, {"batch_sizes": [8, 8]}, {"epochs": 0}):
        with pytest.raises(ValueError):
            SupervisedConfig(**bad)


def test_sentence_order_is_seeded_and_budgets_nest():
    order = supervised.sentence_order(20, 3)
    assert sorted(order) == list(range(20))
    assert order == supervised.sentence_order(20, 3)
    assert order != supervised.sentence_order(20, 4)
    assert set(order[:5]) <= set(order[:10])


def test_pool_items_exclude_truncated_words():
    pool = [{"tokens": ["a", "b", "c"]}, {"tokens": ["d"]}]
    gold = {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4}
    items = supervised.pool_sentence_items([0, 1], pool, gold, {(0, 0), (0, 1)})
    assert items == [{"tokens": ["a", "b", "c"], "targets": {0: 1, 1: 2}}]


def test_train_and_evaluate_uses_seed_and_selected_sentences(monkeypatch):
    seen = {}
    monkeypatch.setattr(supervised, "set_seed", lambda seed: seen.setdefault("seed", seed))
    monkeypatch.setattr(
        supervised, "scoreable_token_keys", lambda *a, **kw: {(0, 0), (1, 0), (1, 1), (2, 0)}
    )

    class FakeAdamW:
        def __init__(self, parameters, **kwargs):
            seen["optimizer"] = kwargs

    monkeypatch.setattr(supervised.torch.optim, "AdamW", FakeAdamW)

    class FakeModel:
        def parameters(self):
            return []

    monkeypatch.setattr(
        supervised, "load_token_classifier", lambda checkpoint, device, labels: (FakeModel(), "tok")
    )

    def fake_train(model, optimizer, tokenizer, items, **kwargs):
        seen["items"] = items
        seen["train"] = kwargs
        return 0.25

    monkeypatch.setattr(supervised, "train_items", fake_train)
    monkeypatch.setattr(
        supervised, "evaluate_model", lambda *a, **kw: {"entity_f1": 0.5, "token_accuracy": 0.9}
    )
    seed_examples = [{"tokens": ["s"], "ner_tags": [0]}]
    pool = [{"tokens": ["a"]}, {"tokens": ["b", "c"]}, {"tokens": ["d"]}]
    gold = {(0, 0): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4}
    metrics = supervised.train_and_evaluate(
        seed_examples,
        pool,
        gold,
        [1],
        [],
        checkpoint="x",
        labels=["O", "B-X", "I-X", "B-Y", "I-Y"],
        seed=7,
        epochs=20,
        learning_rate=3e-5,
        batch_size=16,
        weight_decay=0.01,
        score_batch_size=4,
        max_length=32,
        device="cpu",
        precision="fp32",
    )
    assert seen["seed"] == 7
    assert seen["optimizer"] == {"lr": 3e-5, "weight_decay": 0.01}
    assert seen["train"]["passes"] == 20 and seen["train"]["batch_size"] == 16
    assert seen["items"] == [
        {"tokens": ["s"], "targets": {0: 0}},
        {"tokens": ["b", "c"], "targets": {0: 2, 1: 3}},
    ]
    assert metrics["n_sentences"] == 1
    assert metrics["n_acquired"] == 2 and metrics["scoreable_pool_tokens"] == 4
    assert metrics["percent_acquired"] == 50
    assert metrics["train_loss"] == 0.25


@pytest.fixture
def fake_supervised(tmp_path, monkeypatch):
    config = SupervisedConfig(
        model_seeds=[0, 1],
        learning_rates=[1e-5, 5e-5],
        batch_sizes=[8],
        weight_decays=[0.0],
        validation_sentences=2,
        sentence_percents=[50, 100],
        sweeps_dir=tmp_path,
        sweep_name="example",
    )
    seed = [{"tokens": ["seed"], "ner_tags": [0]}]
    pool = [
        {"pool_idx": idx, "tokens": [str(idx)], "document_name": "doc", "sentence_id": idx}
        for idx in range(6)
    ]
    gold = {(idx, 0): 0 for idx in range(6)}
    test = [{"tokens": ["test"], "ner_tags": [0]}]
    monkeypatch.setattr(supervised, "load_splits", lambda *_: ((seed, pool, gold, test), {}))
    monkeypatch.setattr(supervised, "get_device", lambda: "cpu")
    calls = []

    def fake_train(seed_examples, pool_inputs, pool_gold, indices, evaluation, **kwargs):
        stage = "test" if evaluation is test else "validation"
        if stage == "validation":
            assert len(pool_inputs) == 4 and len(evaluation) == 2
            assert indices == list(range(4))
        calls.append((stage, kwargs["learning_rate"], kwargs["seed"], list(indices)))
        # The larger learning rate wins validation; tests must not pick the first trial.
        score = 0.8 if kwargs["learning_rate"] == 5e-5 else 0.3
        return {
            "n_sentences": len(indices),
            "n_acquired": len(indices),
            "percent_acquired": 100 * len(indices) / len(pool_inputs),
            "scoreable_pool_tokens": len(pool_inputs),
            "entity_f1": score + kwargs["seed"] / 100,
            "token_accuracy": 0.9,
            "train_loss": 0.1,
        }

    monkeypatch.setattr(supervised, "train_and_evaluate", fake_train)
    return config, tmp_path / "example", calls


def test_tune_selects_on_validation_then_tests_frozen_winner(fake_supervised, tmp_path):
    config, root, calls = fake_supervised
    invoke(config)
    tuning = [call for call in calls if call[0] == "validation"]
    testing = [call for call in calls if call[0] == "test"]
    assert len(tuning) == 4 and len(testing) == 4
    assert all(call[1] == 5e-5 for call in testing)
    half = [call[3] for call in testing if len(call[3]) == 3]
    full = [call[3] for call in testing if len(call[3]) == 6]
    assert len(half) == 2 and all(indices == list(range(6)) for indices in full)
    selection = json.loads((root / "selection.json").read_text())
    assert selection["config_id"] == "config_0001"
    assert selection["validation_score"] == pytest.approx(0.805)
    assert json.loads((root / "best_config.json").read_text())["learning_rate"] == 5e-5
    summary = json.loads((root / "summary.json").read_text())
    assert summary["tuning_complete"]
    assert [row["slot"] for row in summary["test_runs"]] == ["test/pool_100pct", "test/pool_50pct"]
    assert all(row["status"] == "complete" for row in summary["test_runs"])

    sweep, rows = supervised.load_test_results(tmp_path, config.checkpoint)
    assert sweep == "example"
    assert rows.height == 4 and set(rows["arm"]) == {"supervised"}
    assert {"seed", "n_acquired", "percent_acquired", "sentence_percent", "entity_f1"} <= set(
        rows.columns
    )
    assert supervised.load_test_results(tmp_path, "other-checkpoint") is None

    calls.clear()
    invoke(config)
    assert calls == []
    invoke(config.model_copy(update={"sentence_percents": [0, 50, 100]}))
    assert [(call[0], len(call[3])) for call in calls] == [("test", 0), ("test", 0)]


def test_resume_rejects_changed_plan_and_test_requires_winner(fake_supervised):
    config, root, calls = fake_supervised
    with pytest.raises(SystemExit):
        invoke(config, "--stage", "test")
    assert calls == []
    invoke(config, "--stage", "tune")
    assert not (root / "test").exists()
    with pytest.raises(SystemExit):
        invoke(config.model_copy(update={"learning_rates": [2e-5]}))


def test_failed_trial_is_recorded_and_retried(fake_supervised, monkeypatch):
    config, root, calls = fake_supervised
    working = supervised.train_and_evaluate

    def fail(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(supervised, "train_and_evaluate", fail)
    with pytest.raises(RuntimeError):
        invoke(config)
    failure = json.loads((root / "tuning" / "config_0000" / "failure.json").read_text())
    assert "out of memory" in failure["error"]
    monkeypatch.setattr(supervised, "train_and_evaluate", working)
    invoke(config)
    assert not (root / "tuning" / "config_0000" / "failure.json").exists()


def test_dry_run_loads_nothing(fake_supervised, monkeypatch, capsys):
    config, root, _ = fake_supervised
    monkeypatch.setattr(supervised, "load_splits", lambda *_: pytest.fail("loaded data"))
    invoke(config, "--dry-run")
    assert json.loads(capsys.readouterr().out)["sentence_percents"] == [50, 100]
    assert not root.exists()


def test_variance_chart_draws_supervised_reference():
    results = pl.DataFrame(
        {
            "seed": [0, 0, 0, 0],
            "arm": ["random", "uncertainty", "random", "uncertainty"],
            "n_acquired": [0, 0, 10, 10],
            "percent_acquired": [0.0, 0.0, 100.0, 100.0],
            "entity_f1": [0.2, 0.2, 0.6, 0.7],
        }
    )
    upper = pl.DataFrame(
        {"seed": [0, 1], "arm": ["supervised"] * 2, "n_acquired": [10, 10]}
    ).with_columns(pl.lit(100.0).alias("percent_acquired"), pl.Series("entity_f1", [0.8, 0.9]))
    plain = make_variance_chart(results, "entropy").to_dict()
    with_upper = make_variance_chart(results, "entropy", supervised=upper).to_dict()
    assert "supervised" not in json.dumps(plain)
    assert "supervised" in json.dumps(with_upper)
    curve = pl.concat([upper, upper.with_columns(pl.lit(50.0).alias("percent_acquired"))])
    assert "supervised" in json.dumps(
        make_variance_chart(results, "entropy", supervised=curve).to_dict()
    )
