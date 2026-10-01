"""Verify supervised sampling, validation-only tuning, test isolation, W&B, and resume."""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest
import torch

from test_concurrent_seeds import tiny_experiment as tiny_experiment
from test_random_search import fake_wandb as fake_wandb
from uq_pet import supervised
from uq_pet.config import SupervisedConfig
from uq_pet.pet_data import NER_TAGS
from uq_pet.search import LEARNING_RATE_BOUNDS
from utils.charts import make_variance_chart


def invoke(config, *extra):
    parser = argparse.ArgumentParser()
    supervised.configure_parser(parser)
    supervised.run(parser.parse_args(["--config-json", config.model_dump_json(), *extra]), parser)


def test_plan_samples_log_uniform_learning_rates_and_is_named_by_checkpoint():
    config = SupervisedConfig(checkpoint="microsoft/deberta-v3-base")
    plan = supervised.sample_plan(config)
    assert config.sweep_name == "deberta-v3-base-supervised"
    assert len(plan["configurations"]) == plan["num_configs"] == 30
    assert [entry["config_id"] for entry in plan["configurations"][:2]] == [
        "config_0000",
        "config_0001",
    ]
    for entry in plan["configurations"]:
        parameters = entry["parameters"]
        assert set(parameters) == set(supervised.TUNED_FIELDS)
        assert LEARNING_RATE_BOUNDS[0] <= parameters["learning_rate"] <= LEARNING_RATE_BOUNDS[1]
        assert parameters["batch_size"] in config.batch_sizes
        assert parameters["weight_decay"] in config.weight_decays
    assert len({entry["parameters"]["learning_rate"] for entry in plan["configurations"]}) == 30
    assert plan["search_space"]["learning_rate"] == {
        "distribution": "log_uniform",
        "low": LEARNING_RATE_BOUNDS[0],
        "high": LEARNING_RATE_BOUNDS[1],
    }
    assert plan == supervised.sample_plan(config)
    assert plan != supervised.sample_plan(config.model_copy(update={"sampler_seed": 1}))
    assert "sentence_percents" not in json.dumps(plan)
    assert "wandb" not in json.dumps(plan)
    for bad in (
        {"sentence_percents": [100, 50]},
        {"batch_sizes": [8, 8]},
        {"epochs": 0},
        {"num_configs": 0},
        {"learning_rates": [1e-5]},
        {"wandb_enabled": True, "wandb_project": " "},
    ):
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
        num_configs=2,
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
        # The larger learning rate wins validation, whichever trial sampled it.
        score = 0.3 + 1000 * kwargs["learning_rate"]
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
    best = max(
        supervised.sample_plan(config)["configurations"],
        key=lambda entry: entry["parameters"]["learning_rate"],
    )
    best_rate = best["parameters"]["learning_rate"]
    tuning = [call for call in calls if call[0] == "validation"]
    testing = [call for call in calls if call[0] == "test"]
    assert len(tuning) == 4 and len(testing) == 4
    assert all(call[1] == best_rate for call in testing)
    half = [call[3] for call in testing if len(call[3]) == 3]
    full = [call[3] for call in testing if len(call[3]) == 6]
    assert len(half) == 2 and all(indices == list(range(6)) for indices in full)
    selection = json.loads((root / "selection.json").read_text())
    assert selection["config_id"] == best["config_id"]
    assert selection["validation_score"] == pytest.approx(0.305 + 1000 * best_rate)
    assert json.loads((root / "best_config.json").read_text())["learning_rate"] == best_rate
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
        invoke(config.model_copy(update={"sampler_seed": 1}))


def test_wandb_logs_each_trial_and_test_budget(fake_supervised, fake_wandb):
    config, root, _ = fake_supervised
    invoke(config.model_copy(update={"wandb_enabled": True, "wandb_project": "supervised"}))
    assert fake_wandb.logins == [{"key": "test-key", "verify": True}]
    names = [run.settings["name"] for run in fake_wandb.runs]
    assert names == [
        "example-config_0000",
        "example-config_0000",
        "example-config_0001",
        "example-config_0001",
        "example-test-50pct",
        "example-test-100pct",
    ]
    summaries = [
        r for r in fake_wandb.runs if r.settings["job_type"] == "supervised_tuning_summary"
    ]
    trials = [r for r in fake_wandb.runs if r not in summaries]
    for run in trials:
        assert run.settings["project"] == "supervised"
        assert run.settings["group"] == "example"
        assert run.exit_codes == [0]
        assert [log["seed"] for log in run.logs[:2]] == [0, 1]
        assert set(run.logs[2]) == {"mean_entity_f1", "std_entity_f1"}
    tuning, test = trials[0], trials[2]
    assert tuning.settings["job_type"] == "supervised_tuning"
    assert tuning.settings["config"]["evaluation_split"] == "validation"
    assert "learning_rate" in tuning.settings["config"]
    assert test.settings["job_type"] == "supervised_test"
    assert test.settings["config"]["evaluation_split"] == "test"
    assert test.settings["config"]["sentence_percent"] == 50
    assert not (root / "plan.json").read_text().count("wandb")

    assert len(fake_wandb.sweeps) == 1
    definition, target = fake_wandb.sweeps[0]
    assert target == {"entity": "test", "project": "supervised"}
    assert definition["metric"] == {"name": "validation_entity_f1", "goal": "maximize"}
    assert set(definition["parameters"]) == set(supervised.TUNED_FIELDS)
    assert definition["parameters"]["learning_rate"]["distribution"] == "log_uniform_values"
    plan = supervised.sample_plan(config)
    for run, entry in zip(summaries, plan["configurations"], strict=True):
        assert run.settings["settings"]["sweep_id"] == "sweep-1"
        assert run.settings["config"]["run_role"] == "configuration_summary"
        assert run.settings["config"]["learning_rate"] == entry["parameters"]["learning_rate"]
        completed = json.loads(
            (root / "tuning" / entry["config_id"] / "completed.json").read_text()
        )
        assert run.logs == [
            {
                "validation_entity_f1": completed["score"],
                "std_validation_entity_f1": completed["std_entity_f1"],
            }
        ]
        assert run.exit_codes == [0]


def test_wandb_sweep_backfills_completed_trials_once(fake_supervised, fake_wandb):
    config, root, calls = fake_supervised
    invoke(config, "--stage", "tune")
    online = config.model_copy(update={"wandb_enabled": True, "wandb_project": "supervised"})
    invoke(online, "--stage", "tune")
    assert len(calls) == 4  # Nothing retrained; both completed trials were published.
    assert [run.settings["job_type"] for run in fake_wandb.runs] == [
        "supervised_tuning_summary"
    ] * 2
    state = json.loads((root / "wandb_sweeps.json").read_text())["test/supervised"]
    assert sorted(state["published"]) == ["config_0000", "config_0001"]
    invoke(online, "--stage", "tune")
    assert len(fake_wandb.runs) == 2 and len(fake_wandb.sweeps) == 1


def test_offline_wandb_creates_no_sweep(fake_supervised, fake_wandb, monkeypatch):
    config, root, _ = fake_supervised
    monkeypatch.setenv("WANDB_MODE", "offline")
    invoke(config.model_copy(update={"wandb_enabled": True}), "--stage", "tune")
    assert not fake_wandb.sweeps
    assert not (root / "wandb_sweeps.json").exists()
    assert len(fake_wandb.runs) == 2


def test_supervised_workspace_plots_parameters_against_validation_f1():
    from uq_pet.utils.wandb_tuning import supervised_chart_workspace

    workspace = supervised_chart_workspace("test", "project", "example")
    columns = workspace.sections[0].panels[0].columns
    assert [column.metric.name for column in columns] == [
        "learning_rate",
        "batch_size",
        "weight_decay",
        "validation_entity_f1",
    ]


def test_wandb_missing_credentials_fails_before_loading_data(fake_supervised, monkeypatch):
    config, _, _ = fake_supervised
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(supervised, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(supervised, "load_splits", lambda *_: pytest.fail("loaded data"))
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        invoke(config.model_copy(update={"wandb_enabled": True}))


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


def test_spawned_seeds_match_sequential_training(tiny_experiment):
    (examples, pool, gold, _), kwargs = tiny_experiment
    settings = {
        "checkpoint": kwargs["checkpoint"],
        "labels": NER_TAGS,
        "epochs": 2,
        "learning_rate": 1e-3,
        "batch_size": 2,
        "weight_decay": 0.0,
        "score_batch_size": 2,
        "max_length": 8,
        "device": torch.device("cpu"),
        "precision": "fp32",
    }
    jobs = {
        seed: ((examples, pool, gold, [0, 2], examples), {**settings, "seed": seed})
        for seed in (11, 3, 8)
    }
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sequential = dict(supervised.train_seeds(jobs, None))
        executor = supervised.seed_executor(2)
        try:
            concurrent = dict(supervised.train_seeds(jobs, executor))
        finally:
            executor.shutdown()
    finally:
        torch.set_num_threads(old_threads)
    assert supervised.seed_executor(1) is None
    assert concurrent == sequential
    assert sequential[11] != sequential[3]


def test_concurrent_seeds_save_in_seed_order_and_keep_the_plan(fake_supervised, monkeypatch):
    config, root, _ = fake_supervised
    workers = []

    def thread_executor(count):
        workers.append(count)
        return ThreadPoolExecutor(count)

    # Seed 0 finishes last, so completion order differs from seed order.
    seed_one_done = threading.Event()
    fake_train = supervised.train_and_evaluate

    def ordered_train(*args, **kwargs):
        if kwargs["seed"] == 0:
            assert seed_one_done.wait(timeout=10)
        metrics = fake_train(*args, **kwargs)
        if kwargs["seed"] == 1:
            seed_one_done.set()
        return metrics

    monkeypatch.setattr(supervised, "seed_executor", thread_executor)
    monkeypatch.setattr(supervised, "train_and_evaluate", ordered_train)
    invoke(config, "--stage", "tune", "--seed-workers", "4")
    seed_one_done.clear()
    invoke(config.model_copy(update={"seed_workers": 2}))
    assert workers == [2, 2]  # Capped by the two seeds; changing workers resumes the plan.
    for marker in [*root.glob("tuning/*/completed.json"), *root.glob("test/*/completed.json")]:
        rows = pl.read_csv(root / json.loads(marker.read_text())["run_dir"] / "results.csv")
        assert rows["seed"].to_list() == [0, 1]
    selections = json.loads(
        next((root / "test" / "pool_50pct").glob("*/selections.json")).read_text()
    )
    assert [row["seed"] for row in selections] == [0, 0, 0, 1, 1, 1]
    with pytest.raises(SystemExit):
        invoke(config, "--seed-workers", "0")


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
