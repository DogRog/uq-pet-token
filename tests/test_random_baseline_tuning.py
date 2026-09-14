"""Verify validation-only baseline selection, final isolation, and resumable execution."""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from test_random_search import fake_wandb as fake_wandb
from uq_pet.experiment import RandomBaselineSearchConfig
from uq_pet.pet_data import split_tuning_pool

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("tune_random", SCRIPTS / "bert_token_uq_search.py")
tune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tune)


def run_tuning(config, *, dry_run=False):
    parser = argparse.ArgumentParser()
    tune.configure_parser(parser)
    args = ["--config-json", config.model_dump_json()]
    if dry_run:
        args.append("--dry-run")
    tune.run(parser.parse_args(args), parser)


def rows(score, *, arm="random", seeds=(0, 1)):
    return [
        {
            "seed": seed,
            "arm": arm,
            "round": idx,
            "total_rounds": 2,
            "percent_acquired": percent,
            "n_acquired": percent,
            "scoreable_pool_tokens": 100,
            "token_budget": 100,
            "entity_f1": score if idx else 0.1,
        }
        for seed in seeds
        for idx, percent in enumerate((0, 30, 100))
    ]


def test_validation_score_normalizes_irregular_intervals_and_averages_seeds():
    data = rows(0.5, seeds=(0,)) + rows(0.9, seeds=(1,))
    assert tune.random_validation_score(data, "random_validation_entity_f1_auc") == pytest.approx(
        0.61
    )
    assert tune.random_validation_score(data, "random_validation_final_entity_f1") == pytest.approx(
        0.7
    )
    for bad in ([], data[:-1], rows(0.5, arm="uncertainty"), rows(float("nan"))):
        with pytest.raises(ValueError):
            tune.random_validation_score(bad, "random_validation_entity_f1_auc")


def test_split_is_label_free_disjoint_reindexed_and_reproducible():
    pool = [{"pool_idx": idx, "tokens": [str(idx), "word"]} for idx in range(10)]
    gold = {(idx, word): idx + word for idx in range(10) for word in range(2)}
    state = random.getstate()
    inputs, labels, validation, manifest = split_tuning_pool(
        pool,
        gold,
        validation_sentences=3,
        validation_seed=9,
    )
    assert random.getstate() == state
    assert len(inputs) == 7 and len(validation) == 3
    assert [row["pool_idx"] for row in inputs] == list(range(7))
    assert all("ner_tags" not in row for row in inputs)
    train = manifest["acquisition_original_pool_indices"]
    val = manifest["validation_original_pool_indices"]
    assert not set(train) & set(val) and sorted(train + val) == list(range(10))
    for idx, original in enumerate(train):
        assert labels[idx, 0] == gold[original, 0]
    assert (inputs, labels, validation, manifest) == split_tuning_pool(
        pool,
        gold,
        validation_sentences=3,
        validation_seed=9,
    )
    with pytest.raises(ValueError):
        split_tuning_pool(pool, gold, validation_sentences=10, validation_seed=9)


def test_plan_locks_objective_metric_and_split_but_allows_worker_changes():
    config = RandomBaselineSearchConfig()
    state = random.getstate()
    plan = tune.tuning_plan(config)
    assert random.getstate() == state
    assert len(plan["configurations"]) == 50
    assert (
        len({json.dumps(row["parameters"], sort_keys=True) for row in plan["configurations"]}) == 50
    )
    assert plan == tune.tuning_plan(config.model_copy(update={"seed_workers": 5}))
    for update in (
        {"uq_metric": "margin"},
        {"validation_seed": 3},
        {"objective": "random_validation_final_entity_f1"},
    ):
        assert plan != tune.tuning_plan(config.model_copy(update=update))
    assert RandomBaselineSearchConfig(model_batch_size=2, wandb_enabled=True).wandb_enabled
    with pytest.raises(ValueError):
        RandomBaselineSearchConfig(model_batch_size=5)


@pytest.fixture
def fake_search(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "SEARCH_SPACE", {"k": (8, 16)})
    monkeypatch.setattr(tune, "PROJECT_ROOT", tmp_path)
    config = RandomBaselineSearchConfig(
        num_configs=2,
        sweeps_dir=tmp_path,
        sweep_name="example",
        validation_sentences=2,
        model_seeds=[0, 1],
    )
    root = tmp_path / "example"
    dataset = tmp_path / "data.json"
    dataset.write_text("fixed dataset")
    seed = [{"tokens": ["seed"], "ner_tags": [0]}]
    pool = [{"pool_idx": idx, "tokens": [str(idx)]} for idx in range(6)]
    gold = {(idx, 0): 0 for idx in range(6)}
    test = [{"tokens": ["test"], "ner_tags": [0]}]
    monkeypatch.setattr(tune, "download_pet_ner", lambda: dataset)
    monkeypatch.setattr(tune, "load_pet_splits", lambda _: (seed, pool, gold, test))
    monkeypatch.setattr(tune, "get_device", lambda: "cpu")
    calls = []

    def run_random(s, p, g, evaluation, **kwargs):
        assert s is seed
        assert evaluation is not test and len(evaluation) == 2
        assert len(p) == 4 and len(g) == 4
        assert all("ner_tags" not in row for row in p)
        assert not {r["tokens"][0] for r in p} & {r["tokens"][0] for r in evaluation}
        calls.append(("validation", kwargs))
        output = rows(0.8 if kwargs["k"] == 16 else 0.3)
        # Publish complete rounds in the shared engine stub below.
        return output, [{"arm": "random", "token": "selected"}]

    def run_pair(s, p, g, evaluation, **kwargs):
        assert s is seed and p is pool and g is gold and evaluation is test
        assert kwargs["k"] == 16
        frozen = json.loads((root / "best_config.json").read_text())
        assert frozen["k"] == 16
        assert all(
            row["status"] == "complete"
            for row in tune.write_summary(root, tune.tuning_plan(config))["trials"]
        )
        random_kwargs = next(kw for stage, kw in calls if stage == "validation" and kw["k"] == 16)
        assert {
            key: val
            for key, val in kwargs.items()
            if key not in {"progress_callback", "random_only"}
        } == {
            key: val
            for key, val in random_kwargs.items()
            if key not in {"progress_callback", "random_only"}
        }
        calls.append(("test", kwargs))
        output = rows(0.4) + rows(0.6, arm="uncertainty")
        # Publish complete rounds in the shared engine stub below.
        return output, [{"arm": "uncertainty", "token": "selected"}]

    def run_comparisons(*args, **kwargs):
        runner = run_random if kwargs["random_only"] else run_pair
        output, selections = runner(*args, **kwargs)
        output.sort(key=lambda row: (row["seed"], row["round"], row["arm"] == "random"))
        width = 1 if kwargs["random_only"] else 2
        metric = kwargs["uq_metrics"][0]
        for end in range(width, len(output) + 1, width):
            kwargs["progress_callback"](metric, output[:end])
        return {metric: (output, selections)}

    monkeypatch.setattr(tune, "run_metric_comparisons", run_comparisons)
    return config, root, calls


def test_tune_then_freeze_then_test_and_completed_resume(fake_search, monkeypatch):
    config, root, calls = fake_search
    run_tuning(config)
    assert [stage for stage, _ in calls] == ["validation", "validation", "test"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    assert summary["final"]["mean_final_test_entity_f1_gap"] == pytest.approx(0.2)
    assert len(summary["final"]["per_seed"]) == 2
    for entry in [*summary["trials"], summary["final"]]:
        path = root / entry["run_dir"]
        assert (path / "selections.json").exists()
        assert (path / "results.csv").exists()
    monkeypatch.setattr(
        tune, "download_pet_ner", lambda: pytest.fail("complete resume loaded data")
    )
    run_tuning(config.model_copy(update={"seed_workers": 2}))
    assert len(calls) == 3
    with pytest.raises(SystemExit, match="2"):
        run_tuning(config.model_copy(update={"validation_seed": 999}))


def test_failed_tuning_never_selects_winner_or_tests_and_resumes(fake_search, monkeypatch):
    config, root, calls = fake_search
    original = tune.run_metric_comparisons
    attempts = 0

    def fail_second(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(tune, "run_metric_comparisons", fail_second)
    with pytest.raises(RuntimeError):
        run_tuning(config)
    assert not (root / "best_config.json").exists()
    assert [stage for stage, _ in calls] == ["validation"]
    summary = json.loads((root / "summary.json").read_text())
    assert [r["status"] for r in summary["trials"]] == ["complete", "failed"]
    saved = summary["trials"][0]["run_dir"]
    monkeypatch.setattr(tune, "run_metric_comparisons", original)
    run_tuning(config)
    assert len(calls) == 3
    assert json.loads((root / "summary.json").read_text())["trials"][0]["run_dir"] == saved


def test_dry_run_has_no_data_or_files(fake_search, monkeypatch):
    config, root, calls = fake_search
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("dry run loaded data"))
    run_tuning(config, dry_run=True)
    assert not root.exists()
    assert calls == []


def test_final_failure_reuses_frozen_winner_and_never_retunes(fake_search, monkeypatch):
    config, root, calls = fake_search
    original = tune.run_metric_comparisons
    monkeypatch.setattr(
        tune,
        "run_metric_comparisons",
        lambda *a, **kw: (
            original(*a, **kw)
            if kw["random_only"]
            else (_ for _ in ()).throw(RuntimeError("test failed"))
        ),
    )
    with pytest.raises(RuntimeError, match="test failed"):
        run_tuning(config)
    frozen = (root / "best_config.json").read_text()
    assert json.loads((root / "summary.json").read_text())["final"]["status"] == "failed"
    monkeypatch.setattr(tune, "run_metric_comparisons", original)
    monkeypatch.setattr(
        tune,
        "run_metric_comparisons",
        lambda *a, **kw: pytest.fail("retuned winner") if kw["random_only"] else original(*a, **kw),
    )
    run_tuning(config)
    assert (root / "best_config.json").read_text() == frozen
    assert [stage for stage, _ in calls] == ["validation", "validation", "test"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    (root / summary["trials"][0]["run_dir"] / "results.csv").unlink()
    with pytest.raises(ValueError, match="missing results.csv"):
        run_tuning(config)


def test_tuning_supports_model_batching_and_wandb_with_separate_validation_test_logs(
    fake_search, fake_wandb
):
    config, root, calls = fake_search
    config = config.model_copy(update={"model_batch_size": 2, "wandb_enabled": True})
    run_tuning(config)
    assert all(kwargs["model_batch_size"] == 2 for _, kwargs in calls)
    assert len(fake_wandb.runs) == 6  # Two tuning configs and one final, each with two seeds.
    validation_runs = [
        r for r in fake_wandb.runs if r.settings["config"]["evaluation_split"] == "validation"
    ]
    test_runs = [r for r in fake_wandb.runs if r.settings["config"]["evaluation_split"] == "test"]
    assert len(validation_runs) == 4 and len(test_runs) == 2
    for run in validation_runs:
        assert run.settings["config"]["random_only"]
        assert all("evaluation/entity_f1/random" in log for log in run.logs)
        assert all("evaluation/entity_f1/entropy" not in log for log in run.logs)
        assert all("/entropy" not in args[0] for args, _ in run.metrics)
    for run in test_runs:
        assert not run.settings["config"]["random_only"]
        assert all("evaluation/entity_f1/entropy" in log for log in run.logs)
    for run in fake_wandb.runs:
        assert [row["evaluation/round"] for row in run.logs] == [0, 1, 2]
        assert run.exit_codes == [0]
    # Changing logging preferences doesn't invalidate a completed scientific plan.
    run_tuning(config.model_copy(update={"wandb_enabled": False}))
    assert len(calls) == 3
    assert json.loads((root / "best_config.json").read_text())["model_batch_size"] == 2


def test_tuning_wandb_missing_credentials_fails_before_loading_data(fake_search, monkeypatch):
    config, _, _ = fake_search
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        run_tuning(config.model_copy(update={"wandb_enabled": True}))


def test_tuning_wandb_offline_resume_retains_winner_and_closes_failed_runs(
    fake_search, fake_wandb, monkeypatch
):
    config, root, calls = fake_search
    config = config.model_copy(update={"wandb_enabled": True})
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY")
    original = tune.run_metric_comparisons

    def fail_final(*args, **kwargs):
        result = original(*args, **kwargs)
        if not kwargs["random_only"]:
            raise RuntimeError("test interrupted")
        return result

    monkeypatch.setattr(tune, "run_metric_comparisons", fail_final)
    with pytest.raises(RuntimeError, match="test interrupted"):
        run_tuning(config)
    assert not fake_wandb.logins
    assert [r.exit_codes for r in fake_wandb.runs] == [[0]] * 4 + [[1]] * 2
    frozen = (root / "best_config.json").read_bytes()
    monkeypatch.setattr(tune, "run_metric_comparisons", original)
    run_tuning(config.model_copy(update={"wandb_enabled": False, "wandb_project": "changed"}))
    assert (root / "best_config.json").read_bytes() == frozen
    assert [stage for stage, _ in calls] == ["validation", "validation", "test", "test"]
