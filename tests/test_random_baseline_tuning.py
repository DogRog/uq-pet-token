"""Verify validation-only baseline selection, final isolation, and resumable execution."""

import argparse
import json
import math
import random

import pytest

from test_random_search import fake_wandb as fake_wandb
from uq_pet import search as tune
from uq_pet.config import RandomBaselineSearchConfig
from uq_pet.pet_data import split_tuning_pool


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
        {"validation_seed": 3},
        {"objective": "random_validation_final_entity_f1"},
    ):
        assert plan != tune.tuning_plan(config.model_copy(update=update))
    assert not any(key.startswith("final_") for key in plan)
    assert plan == tune.tuning_plan(config.model_copy(update={"uq_metric": "margin"}))
    assert RandomBaselineSearchConfig(wandb_enabled=True).wandb_enabled


@pytest.fixture
def fake_search(tmp_path, monkeypatch):
    original_plan = tune.sample_plan

    def fixed_candidates(config):
        plan = original_plan(config)
        # Keep winner-selection tests independent of random sampling. The second
        # candidate must score better, so choosing the first trial cannot pass.
        for candidate, k in zip(plan["configurations"], (8, 16), strict=True):
            candidate["parameters"]["k"] = k
        plan["search_space"]["k"] = [8, 16]
        return plan

    monkeypatch.setattr(tune, "sample_plan", fixed_candidates)
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

    def run_comparisons(*args, **kwargs):
        assert kwargs["random_only"], "tune-random must never launch UQ or evaluate test"
        output, selections = run_random(*args, **kwargs)
        output.sort(key=lambda row: (row["seed"], row["round"], row["arm"] == "random"))
        width = 1 if kwargs["random_only"] else 2
        metric = kwargs["uq_metrics"][0]
        for end in range(width, len(output) + 1, width):
            kwargs["progress_callback"](metric, output[:end])
        return {metric: (output, selections)}

    monkeypatch.setattr(tune, "run_metric_comparisons", run_comparisons)
    return config, root, calls


def test_tune_freezes_winner_and_stops_without_uq(fake_search, monkeypatch):
    config, root, calls = fake_search
    run_tuning(config)
    assert [stage for stage, _ in calls] == ["validation", "validation"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and "final" not in summary
    assert not (root / "final").exists()
    frozen = json.loads((root / "best_config.json").read_text())
    assert frozen["k"] == calls[1][1]["k"] == 16
    assert frozen["learning_rate"] == calls[1][1]["learning_rate"]
    assert "uq_metric" not in frozen
    selection = json.loads((root / "selection.json").read_text())
    assert selection["config_id"] == "config_0001"
    assert selection["selection_split"] == "validation"
    for entry in summary["trials"]:
        path = root / entry["run_dir"]
        assert (path / "selections.json").exists()
        assert (path / "results.csv").exists()
        metadata = json.loads((path / "config.json").read_text())
        assert metadata["random_only"] and "uq_metric" not in metadata
    monkeypatch.setattr(
        tune, "download_pet_ner", lambda: pytest.fail("complete resume loaded data")
    )
    run_tuning(config.model_copy(update={"seed_workers": 2, "uq_metric": "margin"}))
    assert len(calls) == 2
    with pytest.raises(SystemExit, match="2"):
        run_tuning(config.model_copy(update={"validation_seed": 999}))


def test_failed_tuning_never_selects_winner_and_resumes(fake_search, monkeypatch):
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
    assert not (root / "final").exists()
    summary = json.loads((root / "summary.json").read_text())
    assert [r["status"] for r in summary["trials"]] == ["complete", "failed"]
    saved = summary["trials"][0]["run_dir"]
    monkeypatch.setattr(tune, "run_metric_comparisons", original)
    run_tuning(config)
    assert len(calls) == 2
    assert json.loads((root / "summary.json").read_text())["trials"][0]["run_dir"] == saved


def test_dry_run_has_no_data_or_files(fake_search, monkeypatch):
    config, root, calls = fake_search
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("dry run loaded data"))
    run_tuning(config, dry_run=True)
    assert not root.exists() and calls == []


def test_resume_finishes_winner_publication_without_retuning(fake_search, monkeypatch):
    config, root, calls = fake_search
    run_tuning(config)
    (root / "selection.json").unlink()
    monkeypatch.setattr(tune, "run_metric_comparisons", lambda *a, **kw: pytest.fail("retuned"))
    run_tuning(config)
    assert len(calls) == 2
    assert json.loads((root / "summary.json").read_text())["complete"]


def test_tuning_wandb_logs_only_random_validation(fake_search, fake_wandb):
    config, root, calls = fake_search
    config = config.model_copy(update={"wandb_enabled": True})
    run_tuning(config)
    assert len(fake_wandb.runs) == 6  # Four seed logs, two configuration summaries.
    summaries = [r for r in fake_wandb.runs if r.settings["job_type"] == "random_tuning_summary"]
    assert len(summaries) == 2
    assert len(fake_wandb.sweeps) == 1
    assert fake_wandb.sweeps[0][0]["metric"] == {"name": config.objective, "goal": "maximize"}
    for run in summaries:
        assert run.settings["settings"]["sweep_id"] == "sweep-1"
        assert run.logs[0][config.objective] == pytest.approx(
            tune.random_validation_score(
                rows(0.8 if run.settings["config"]["k"] == 16 else 0.3), config.objective
            )
        )
        assert "random_validation_final_entity_f1" in run.logs[0]
    for run in [r for r in fake_wandb.runs if r not in summaries]:
        assert run.settings["config"]["evaluation_split"] == "validation"
        assert run.settings["config"]["random_only"]
        assert all("evaluation/entity_f1/random" in log for log in run.logs)
        assert all("evaluation/entity_f1/entropy" not in log for log in run.logs)
        assert all("/entropy" not in args[0] for args, _ in run.metrics)
        assert [row["evaluation/round"] for row in run.logs] == [0, 1, 2]
        assert run.exit_codes == [0]
    run_tuning(config.model_copy(update={"wandb_enabled": False}))
    assert len(calls) == 2


def test_tuning_wandb_missing_credentials_fails_before_loading_data(fake_search, monkeypatch):
    config, _, _ = fake_search
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        run_tuning(config.model_copy(update={"wandb_enabled": True}))


def test_legacy_sweep_resumes_without_touching_historical_final(fake_search, monkeypatch):
    config, root, calls = fake_search
    run_tuning(config)
    current = json.loads((root / "plan.json").read_text())
    legacy = {
        **current,
        "version": 2,
        "final_uq_metric": "entropy",
        "final_evaluation_split": "test",
        "final_pool": "original_328_sentences",
    }
    params = legacy["configurations"][0]["parameters"]
    params["learning_rate"] = math.nextafter(params["learning_rate"], math.inf)
    tune.write_json(root / "plan.json", legacy)
    (root / "final").mkdir()
    historical = root / "final" / "completed.json"
    historical.write_text('{"historical": true}')
    frozen = (root / "best_config.json").read_bytes()
    monkeypatch.setattr(
        tune, "download_pet_ner", lambda: pytest.fail("completed resume loaded data")
    )
    run_tuning(config)
    assert len(calls) == 2
    assert (root / "best_config.json").read_bytes() == frozen
    assert historical.read_text() == '{"historical": true}'
    assert json.loads((root / "plan.v2.json").read_text()) == legacy
    assert (
        json.loads((root / "plan.json").read_text())["configurations"] == legacy["configurations"]
    )
    assert json.loads((root / "summary.json").read_text())["complete"]


def test_plan_comparison_rejects_material_rate_changes():
    plan = tune.tuning_plan(RandomBaselineSearchConfig())
    changed = json.loads(json.dumps(plan))
    changed["configurations"][0]["parameters"]["learning_rate"] *= 1.00001
    assert tune.scientific_plan(plan) != tune.scientific_plan(changed)


def test_publish_completed_search_creates_sweep_without_training(
    fake_search, fake_wandb, monkeypatch
):
    config, root, calls = fake_search
    run_tuning(config)
    monkeypatch.setattr(tune, "download_pet_ner", lambda: pytest.fail("publication loaded data"))
    online = config.model_copy(update={"wandb_enabled": True, "wandb_project": "random-tuning"})
    parser = argparse.ArgumentParser()
    tune.configure_parser(parser)
    args = parser.parse_args(["--config-json", online.model_dump_json(), "--publish-wandb-only"])
    tune.run(args, parser)
    assert len(calls) == 2
    assert len(fake_wandb.runs) == 2 and len(fake_wandb.sweeps) == 1
    state = json.loads((root / "wandb_sweeps.json").read_text())["test/random-tuning"]
    assert len(state["published"]) == 2
    tune.run(args, parser)
    assert len(fake_wandb.runs) == 2 and len(fake_wandb.sweeps) == 1


def test_offline_tuning_does_not_create_remote_sweep(fake_search, fake_wandb, monkeypatch):
    config, _, calls = fake_search
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_tuning(config.model_copy(update={"wandb_enabled": True}))
    assert len(calls) == 2
    assert not fake_wandb.sweeps
    assert len(fake_wandb.runs) == 4


def test_tuning_workspace_axes_include_objective_and_f1():
    from uq_pet.utils.wandb_tuning import AXES, chart_workspace

    workspace = chart_workspace("test", "project", "example", "random_validation_entity_f1_auc")
    columns = workspace.sections[0].panels[0].columns
    assert [column.metric.name for column in columns] == [
        *AXES,
        "random_validation_final_entity_f1",
        "random_validation_entity_f1_auc",
    ]
    assert columns[0].log
