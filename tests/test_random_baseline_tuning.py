"""Verify random-only baseline selection on validation or test, isolation, and resumes."""

import argparse
import json
import math
import random

import pytest

from test_random_search import fake_wandb as fake_wandb
from uq_pet import search as tune
from uq_pet.config import RandomBaselineSearchConfig
from uq_pet.data_prep import split_tuning_pool


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
    assert tune.random_tuning_score(data, "random_validation_entity_f1_auc") == pytest.approx(0.61)
    assert tune.random_tuning_score(data, "random_validation_final_entity_f1") == pytest.approx(0.7)
    for bad in ([], data[:-1], rows(0.5, arm="uncertainty"), rows(float("nan"))):
        with pytest.raises(ValueError):
            tune.random_tuning_score(bad, "random_validation_entity_f1_auc")


def test_best_trial_text_uses_the_winner_tie_break():
    assert tune.best_trial_text({}) == ""
    scores = {"config_0003": 0.7, "config_0001": 0.7, "config_0000": 0.6}
    assert tune.best_trial_text(scores) == " · best config_0001 0.7000"


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
    seed = [{"tokens": ["seed"], "ner_tags": [0]}]
    pool = [{"pool_idx": idx, "tokens": [str(idx)]} for idx in range(6)]
    gold = {(idx, 0): 0 for idx in range(6)}
    test = [{"tokens": ["test"], "ner_tags": [0]}]
    monkeypatch.setattr(
        tune, "load_splits", lambda *_: ((seed, pool, gold, test), {"dataset_sha256": "fixed"})
    )
    monkeypatch.setattr(tune, "get_device", lambda: "cpu")
    calls = []

    def run_random(s, p, g, evaluation, **kwargs):
        assert s is seed
        assert all("ner_tags" not in row for row in p)
        if evaluation is test:
            # Test tuning (the appendix oracle) trains on the full pool.
            assert p is pool and g is gold
            calls.append(("test", kwargs))
        else:
            assert len(evaluation) == 2
            assert len(p) == 4 and len(g) == 4
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
        predictions = [[["O"] * len(row["tokens"]) for row in args[3]]] * len(output)
        return {metric: (output, selections, predictions)}

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
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("complete resume loaded data"))
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
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("dry run loaded data"))
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
            tune.random_tuning_score(
                rows(0.8 if run.settings["config"]["k"] == 16 else 0.3), config.objective
            )
        )
        assert "random_validation_final_entity_f1" in run.logs[0]
        assert run.settings["config"]["evaluation_split"] == "validation"
    assert len(fake_wandb.tag_updates) == 1 and fake_wandb.tag_updates[0][1] == ["winner"]
    for run in [r for r in fake_wandb.runs if r not in summaries]:
        assert run.settings["name"].endswith("-validation-random")
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
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("unexpected download"))
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
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("completed resume loaded data"))
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
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("publication loaded data"))
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


@pytest.mark.parametrize("split", ["validation", "test"])
def test_tuning_workspace_ranks_configurations_by_objective(split):
    from uq_pet.utils.wandb_tuning import AXES, chart_workspace

    objective = f"random_{split}_entity_f1_auc"
    workspace = chart_workspace("test", "project", "example", objective)
    assert workspace.name == f"example — {split} tuning"
    bars, coordinates = workspace.sections[0].panels
    assert [metric.name for metric in bars.metrics] == [objective]
    assert bars.title.endswith("the top bar wins")
    (order,) = workspace.runset_settings.order
    assert order.item.name == objective and not order.ascending
    columns = coordinates.columns
    assert [column.metric.name for column in columns] == [
        *AXES,
        f"random_{split}_final_entity_f1",
        objective,
    ]
    assert columns[0].log
    workspace._to_model()


def test_test_tuning_requires_a_test_objective():
    with pytest.raises(ValueError, match="does not score the test split"):
        RandomBaselineSearchConfig(tuning_split="test")
    with pytest.raises(ValueError, match="does not score the validation split"):
        RandomBaselineSearchConfig(objective="random_test_entity_f1_auc")


def test_test_tuning_plan_drops_the_holdout_and_keeps_validation_plans_unchanged():
    validation = tune.tuning_plan(RandomBaselineSearchConfig())
    assert validation["tuning_evaluation_split"] == "validation"
    assert {"validation_sentences", "validation_seed"} <= set(validation)
    test = tune.tuning_plan(
        RandomBaselineSearchConfig(tuning_split="test", objective="random_test_entity_f1_auc")
    )
    assert test["tuning_evaluation_split"] == "test"
    assert not {"validation_sentences", "validation_seed"} & set(test)
    assert test["configurations"] == validation["configurations"]
    assert tune.winner_filename(test) == "oracle_config.json"
    assert tune.winner_filename(validation) == "best_config.json"


def test_test_tuning_scores_the_full_pool_on_test_and_names_the_oracle(fake_search, monkeypatch):
    config, root, calls = fake_search
    config = config.model_copy(
        update={"tuning_split": "test", "objective": "random_test_entity_f1_auc"}
    )
    run_tuning(config)
    assert [stage for stage, _ in calls] == ["test", "test"]
    assert not (root / "best_config.json").exists()
    frozen = json.loads((root / "oracle_config.json").read_text())
    assert frozen["k"] == 16
    selection = json.loads((root / "selection.json").read_text())
    assert selection["selection_split"] == "test"
    assert selection["test_score"] == pytest.approx(
        tune.random_tuning_score(rows(0.8), "random_test_entity_f1_auc")
    )
    assert "validation_score" not in selection
    split = json.loads((root / "split.json").read_text())
    assert split["tuning_split"] == "test" and "validation_sentences" not in split
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    for entry in summary["trials"]:
        metadata = json.loads((root / entry["run_dir"] / "config.json").read_text())
        assert metadata["evaluation_split"] == "test" and metadata["random_only"]
        assert metadata["test_sentences"] == 1 and "validation_sentences" not in metadata
    monkeypatch.setattr(tune, "load_splits", lambda *_: pytest.fail("complete resume loaded data"))
    run_tuning(config)
    assert len(calls) == 2


def test_test_tuning_wandb_labels_test_and_tags_the_winner(fake_search, fake_wandb):
    config, root, calls = fake_search
    config = config.model_copy(
        update={
            "tuning_split": "test",
            "objective": "random_test_entity_f1_auc",
            "wandb_enabled": True,
            "wandb_project": "pet",
        }
    )
    run_tuning(config)
    definition = fake_wandb.sweeps[0][0]
    assert definition["metric"] == {"name": "random_test_entity_f1_auc", "goal": "maximize"}
    assert "test-tuned random oracle" in definition["description"]
    summaries = [r for r in fake_wandb.runs if r.settings["job_type"] == "random_tuning_summary"]
    assert len(summaries) == 2
    for run in summaries:
        assert run.settings["config"]["evaluation_split"] == "test"
        assert set(run.logs[0]) == {"random_test_entity_f1_auc", "random_test_final_entity_f1"}
    for run in [r for r in fake_wandb.runs if r not in summaries]:
        assert run.settings["name"].endswith("-test-random")
        assert run.settings["config"]["evaluation_split"] == "test"
    state = json.loads((root / "wandb_sweeps.json").read_text())["test/pet"]
    assert state["winner"] == "config_0001"
    winner_run = state["published"]["config_0001"]
    assert fake_wandb.tag_updates == [(f"test/pet/{winner_run}", ["winner"])]
    run_tuning(config)
    assert len(fake_wandb.tag_updates) == 1
