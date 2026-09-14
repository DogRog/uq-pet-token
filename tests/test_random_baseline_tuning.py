"""Verify validation-only baseline selection, final isolation, and resumable execution."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from uq_pet.experiment import RandomBaselineSearchConfig
from uq_pet.pet_data import split_tuning_pool

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "tune_random", SCRIPTS / "bert_token_uq_tune_random.py"
)
tune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tune)


def rows(score, *, arm="random", seeds=(0, 1)):
    return [
        {
            "seed": seed,
            "arm": arm,
            "round": idx,
            "total_rounds": 2,
            "percent_acquired": percent,
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
    with pytest.raises(ValueError):
        RandomBaselineSearchConfig(model_batch_size=2)


@pytest.fixture
def fake_search(tmp_path, monkeypatch):
    monkeypatch.setattr("bert_token_uq_search.SEARCH_SPACE", {"k": (8, 16)})
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
        kwargs["progress_callback"](output)
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
        assert {key: val for key, val in kwargs.items() if key != "progress_callback"} == {
            key: val for key, val in random_kwargs.items() if key != "progress_callback"
        }
        calls.append(("test", kwargs))
        output = rows(0.4) + rows(0.6, arm="uncertainty")
        kwargs["progress_callback"](output)
        return output, [{"arm": "uncertainty", "token": "selected"}]

    monkeypatch.setattr(tune, "run_random_selection", run_random)
    monkeypatch.setattr(tune, "run_active_learning", run_pair)
    return config, root, calls


def test_tune_then_freeze_then_test_and_completed_resume(fake_search, monkeypatch):
    config, root, calls = fake_search
    tune.run(config)
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
    tune.run(config.model_copy(update={"seed_workers": 2}))
    assert len(calls) == 3
    with pytest.raises(ValueError, match="differs"):
        tune.run(config.model_copy(update={"validation_seed": 999}))


def test_failed_tuning_never_selects_winner_or_tests_and_resumes(fake_search, monkeypatch):
    config, root, calls = fake_search
    original = tune.run_random_selection
    attempts = 0

    def fail_second(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(tune, "run_random_selection", fail_second)
    with pytest.raises(RuntimeError):
        tune.run(config)
    assert not (root / "best_config.json").exists()
    assert [stage for stage, _ in calls] == ["validation"]
    summary = json.loads((root / "summary.json").read_text())
    assert [r["status"] for r in summary["trials"]] == ["complete", "failed"]
    saved = summary["trials"][0]["run_dir"]
    monkeypatch.setattr(tune, "run_random_selection", original)
    tune.run(config)
    assert len(calls) == 3
    assert json.loads((root / "summary.json").read_text())["trials"][0]["run_dir"] == saved


def test_dry_run_has_no_data_or_files(fake_search, monkeypatch):
    config, root, calls = fake_search
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("dry run loaded data"))
    tune.run(config, dry_run=True)
    assert not root.exists()
    assert calls == []


def test_final_failure_reuses_frozen_winner_and_never_retunes(fake_search, monkeypatch):
    config, root, calls = fake_search
    original = tune.run_active_learning
    monkeypatch.setattr(
        tune,
        "run_active_learning",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test failed")),
    )
    with pytest.raises(RuntimeError, match="test failed"):
        tune.run(config)
    frozen = (root / "best_config.json").read_text()
    assert json.loads((root / "summary.json").read_text())["final"]["status"] == "failed"
    monkeypatch.setattr(tune, "run_active_learning", original)
    monkeypatch.setattr(
        tune, "run_random_selection", lambda *a, **kw: pytest.fail("retuned winner")
    )
    tune.run(config)
    assert (root / "best_config.json").read_text() == frozen
    assert [stage for stage, _ in calls] == ["validation", "validation", "test"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    (root / summary["trials"][0]["run_dir"] / "results.csv").unlink()
    with pytest.raises(ValueError, match="missing results.csv"):
        tune.run(config)
