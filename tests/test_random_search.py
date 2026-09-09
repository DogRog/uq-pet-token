import argparse
import importlib.util
import json
import random
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

SEARCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bert_token_uq_search.py"
SPEC = importlib.util.spec_from_file_location("bert_token_uq_search", SEARCH_PATH)
assert SPEC is not None and SPEC.loader is not None
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)
mean_entity_f1_gap_auc = search.mean_entity_f1_gap_auc


@pytest.fixture(autouse=True)
def isolated_project_env(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "PROJECT_ROOT", tmp_path)


def test_mean_entity_f1_gap_auc_pairs_arms_and_averages_seeds():
    gaps = {
        0: (0.0, 0.2, 0.2),
        1: (0.0, 0.1, 0.3),
    }
    results = []
    for seed, seed_gaps in gaps.items():
        for round_idx, (percent, gap) in enumerate(zip((0.0, 50.0, 100.0), seed_gaps, strict=True)):
            results.extend(
                [
                    {
                        "seed": seed,
                        "round": round_idx,
                        "arm": "uncertainty",
                        "percent_acquired": percent,
                        "entity_f1": 0.5 + gap,
                    },
                    {
                        "seed": seed,
                        "round": round_idx,
                        "arm": "random",
                        "percent_acquired": percent,
                        "entity_f1": 0.5,
                    },
                ]
            )

    assert mean_entity_f1_gap_auc(results) == pytest.approx(0.1375)


def test_mean_entity_f1_gap_auc_rejects_half_finished_round():
    with pytest.raises(ValueError, match="uncertainty and random"):
        mean_entity_f1_gap_auc(
            [
                {
                    "seed": 0,
                    "round": 0,
                    "arm": "uncertainty",
                    "percent_acquired": 0.0,
                    "entity_f1": 0.5,
                }
            ]
        )


def test_plan_is_unique_reproducible_local_and_covers_all_metrics():
    config = search.RandomSearchConfig(num_configs=30)
    state = random.getstate()
    plan = search.sample_plan(config)
    assert random.getstate() == state
    assert plan == search.sample_plan(config)
    assert plan != search.sample_plan(config.model_copy(update={"sampler_seed": 1}))
    assert plan == search.sample_plan(config.model_copy(update={"seed_workers": 5}))
    assert (
        len({json.dumps(row["parameters"], sort_keys=True) for row in plan["configurations"]}) == 30
    )
    assert plan["uq_metrics"] == list(search.UQ_METRICS)
    assert plan["evaluation_split"] == "test"
    assert "uq_metric" not in plan["search_space"]
    for row in plan["configurations"]:
        assert all(value in search.SEARCH_SPACE[key] for key, value in row["parameters"].items())
    assert plan["fixed_config"]["model_seeds"] == [0, 1, 2, 3, 4]


def test_file_config_and_explicit_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"num_configs": 30, "sweep_name": "example", "sampler_seed": 8}))
    parser = argparse.ArgumentParser()
    search.configure_parser(parser)
    config = search.load_search_config(
        parser.parse_args(["--config", str(path), "--num-configs", "2"]), parser
    )
    assert config.num_configs == 2
    assert config.sweep_name == "example"
    assert config.sampler_seed == 8
    assert set(config.experiment_config().model_dump()) == set(search.ExperimentConfig.model_fields)


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "{}",
        '{"num_configs":0}',
        '{"num_configs":999999}',
        '{"num_configs":1,"sampler":"tpe"}',
        '{"num_configs":1,"unknown":true}',
        '{"num_configs":1,"sampler_seed":-1}',
        '{"num_configs":1,"sweep_name":"../outside"}',
    ],
)
def test_invalid_config_has_no_side_effects(tmp_path, payload):
    parser = argparse.ArgumentParser()
    search.configure_parser(parser)
    with pytest.raises(SystemExit, match="2"):
        search.run(
            parser.parse_args(["--config-json", payload, "--sweeps-dir", str(tmp_path)]), parser
        )
    assert list(tmp_path.iterdir()) == []


def run_search(tmp_path, **overrides):
    parser = argparse.ArgumentParser()
    search.configure_parser(parser)
    args = parser.parse_args(
        [
            "--config-json",
            json.dumps(
                {
                    "num_configs": 2,
                    "sweep_name": "example",
                    "sweeps_dir": str(tmp_path),
                    "model_seeds": [0, 1],
                    **overrides,
                }
            ),
        ]
    )
    search.run(args, parser)


@pytest.fixture
def fake_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SEARCH_SPACE", {"k": (8, 16)})
    seed, pool, gold, test = (
        [{"tokens": ["seed"]}],
        [{"tokens": ["pool"]}],
        {(0, 0): 0},
        [{"tokens": ["test"]}],
    )
    monkeypatch.setattr(search, "download_pet_ner", lambda: "unused")
    monkeypatch.setattr(search, "load_pet_splits", lambda _: (seed, pool, gold, test))
    monkeypatch.setattr(search, "get_device", lambda: "cpu")
    calls = []

    def fake_run(seed_examples, pool_inputs, pool_gold, test_examples, **kwargs):
        assert seed_examples is seed
        assert pool_inputs is pool and pool_gold is gold
        assert test_examples is test
        plan = json.loads((tmp_path / "example" / "plan.json").read_text())
        assert len(plan["configurations"]) == 2  # full plan precedes the first model call
        assert plan["uq_metrics"] == list(search.UQ_METRICS)
        calls.append(kwargs)
        comparisons = {}
        for metric in kwargs["uq_metrics"]:
            gap = {"entropy": 0.2, "least_confidence": -0.1, "margin": 0.0}[metric]
            results = []
            for seed_id in kwargs["model_seeds"]:
                for round_id, percent in enumerate((0, 100)):
                    for arm in ("uncertainty", "random"):
                        results.append(
                            {
                                "seed": seed_id,
                                "round": round_id,
                                "total_rounds": 1,
                                "arm": arm,
                                "percent_acquired": percent,
                                "n_acquired": round_id,
                                "scoreable_pool_tokens": 1,
                                "token_budget": 1,
                                "entity_f1": 0.5
                                + (gap if arm == "uncertainty" and round_id else 0),
                            }
                        )
                    kwargs["progress_callback"](metric, list(results))
            comparisons[metric] = (results, [{"seed": 0, "arm": "random", "token": "pool"}])
        return comparisons

    monkeypatch.setattr(search, "run_metric_comparisons", fake_run)
    return calls


def test_sweep_evaluates_original_test_and_saves_every_pair(tmp_path, fake_experiment):
    run_search(tmp_path)
    root = tmp_path / "example"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    assert len(fake_experiment) == 2
    assert len(summary["comparisons"]) == 6
    assert [row["wins"] for row in summary["by_metric"]] == [2, 0, 0]
    assert [row["losses"] for row in summary["by_metric"]] == [0, 2, 0]
    assert [row["ties"] for row in summary["by_metric"]] == [0, 0, 2]
    assert all(call["uq_metrics"] == list(search.UQ_METRICS) for call in fake_experiment)
    assert all(call["precision"] == "fp32" for call in fake_experiment)
    for comparison in summary["comparisons"]:
        run_dir = root / comparison["run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        assert config["evaluation_split"] == "test"
        assert config["test_sentences"] == 1
        assert (run_dir / "selections.json").is_file()
        assert len(search.pl.read_csv(run_dir / "results.csv")) == 8
        assert len(search.pl.read_csv(run_dir.parent / "progress.csv")) == 8
    assert not (root / "best_config.json").exists()


def test_complete_resume_does_not_load_data_and_rejects_plan_changes(
    tmp_path, fake_experiment, monkeypatch
):
    run_search(tmp_path)
    monkeypatch.setattr(
        search, "download_pet_ner", lambda: pytest.fail("complete resume must not load data")
    )
    run_search(tmp_path, seed_workers=2)
    assert len(fake_experiment) == 2
    for overrides in ({"num_configs": 1}, {"sampler_seed": 9}, {"checkpoint": "roberta-base"}):
        with pytest.raises(SystemExit, match="2"):
            run_search(tmp_path, **overrides)
    assert len(fake_experiment) == 2


def test_failed_run_is_reported_and_resume_keeps_completed_runs(
    tmp_path, fake_experiment, monkeypatch
):
    original = search.run_metric_comparisons
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("model failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "run_metric_comparisons", fail_second)
    with pytest.raises(RuntimeError, match="model failed"):
        run_search(tmp_path)
    root = tmp_path / "example"
    first = json.loads((root / "summary.json").read_text())
    assert not first["complete"]
    assert [row["status"] for row in first["comparisons"]] == [
        "complete",
        "complete",
        "complete",
        "failed",
        "failed",
        "failed",
    ]
    saved = first["comparisons"][0]["run_dir"]
    monkeypatch.setattr(search, "run_metric_comparisons", original)
    run_search(tmp_path)
    final = json.loads((root / "summary.json").read_text())
    assert final["complete"]
    assert final["comparisons"][0]["run_dir"] == saved
    assert len(fake_experiment) == 2


def test_corrupt_completed_output_is_not_silently_skipped(tmp_path, fake_experiment):
    run_search(tmp_path)
    root = tmp_path / "example"
    summary = json.loads((root / "summary.json").read_text())
    (root / summary["comparisons"][0]["run_dir"] / "results.csv").unlink()
    with pytest.raises(ValueError, match="missing results.csv"):
        run_search(tmp_path)


def test_partial_export_failure_resumes_only_missing_metrics(
    tmp_path, fake_experiment, monkeypatch
):
    original = search.write_run
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("disk unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "write_run", fail_second)
    with pytest.raises(OSError, match="disk unavailable"):
        run_search(tmp_path)
    root = tmp_path / "example"
    before = json.loads((root / "summary.json").read_text())
    assert [row["status"] for row in before["comparisons"]] == [
        "complete",
        "failed",
        "failed",
        "pending",
        "pending",
        "pending",
    ]
    saved_dir = before["comparisons"][0]["run_dir"]
    saved_bytes = (root / saved_dir / "results.csv").read_bytes()
    monkeypatch.setattr(search, "write_run", original)
    run_search(tmp_path)
    assert fake_experiment[1]["uq_metrics"] == ["least_confidence", "margin"]
    after = json.loads((root / "summary.json").read_text())
    assert after["complete"]
    assert after["comparisons"][0]["run_dir"] == saved_dir
    assert (root / saved_dir / "results.csv").read_bytes() == saved_bytes


def test_resume_rejects_effective_precision_changes(tmp_path, fake_experiment, monkeypatch):
    run_search(tmp_path)
    monkeypatch.setattr(search, "resolve_precision", lambda *args: "bf16")
    with pytest.raises(SystemExit, match="2"):
        run_search(tmp_path)
    assert len(fake_experiment) == 2


def test_resume_rejects_model_group_size_changes(tmp_path, fake_experiment):
    run_search(tmp_path, model_batch_size=2)
    with pytest.raises(SystemExit, match="2"):
        run_search(tmp_path, model_batch_size=1)
    assert len(fake_experiment) == 2


@pytest.fixture
def fake_wandb(monkeypatch):
    class Run:
        def __init__(self, settings):
            self.settings = settings
            self.logs = []
            self.metrics = []
            self.exit_codes = []

        def log(self, payload):
            self.logs.append(payload)

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def finish(self, *, exit_code):
            self.exit_codes.append(exit_code)

        def get_project_url(self):
            return "https://wandb.ai/test/pet"

        def get_url(self):
            return f"https://wandb.ai/test/pet/runs/{self.settings['name']}"

    class SDK:
        runs = []
        logins = []

        def init(self, **settings):
            run = Run(settings)
            self.runs.append(run)
            return run

        def login(self, **kwargs):
            self.logins.append(kwargs)

    sdk = SDK()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return sdk


def test_sweep_wandb_logs_each_seed_metric_and_prints_project_link(
    tmp_path, fake_experiment, fake_wandb, monkeypatch
):
    output = StringIO()
    monkeypatch.setattr(search, "CONSOLE", Console(file=output, width=240, color_system=None))
    run_search(tmp_path, wandb_enabled=True, wandb_project="pet", wandb_run_name="trial")
    assert fake_wandb.logins == [{"key": "test-key", "verify": True}]
    assert len(fake_wandb.runs) == 2 * 3 * 2  # configurations x metrics x seeds
    assert "W&B project: https://wandb.ai/test/pet" in output.getvalue()
    assert "W&B run (entropy, seed 0):" in output.getvalue()
    for run in fake_wandb.runs:
        config = run.settings["config"]
        metric = config["uq_metric"]
        assert config["wandb_enabled"]
        assert config["model_seeds"] == [config["seed"]]
        assert config["effective_precision"] == "fp32"
        assert config["evaluation_split"] == "test"
        assert run.settings["reinit"] == "create_new"
        assert run.settings["group"] == "example"
        assert run.settings["name"].startswith("trial-config_")
        assert [row["evaluation/round"] for row in run.logs] == [0, 1]
        assert all(row["evaluation/seed"] == config["seed"] for row in run.logs)
        assert all(f"evaluation/entity_f1/{metric}" in row for row in run.logs)
        assert all("evaluation/entity_f1/random" in row for row in run.logs)
        assert run.exit_codes == [0]
    summary = json.loads((tmp_path / "example" / "summary.json").read_text())
    for comparison in summary["comparisons"]:
        config = json.loads(
            (tmp_path / "example" / comparison["run_dir"] / "config.json").read_text()
        )
        assert config["wandb_enabled"] is True
        assert config["wandb_project"] == "pet"


def test_wandb_disabled_never_initializes_or_logs_in(tmp_path, fake_experiment, fake_wandb):
    run_search(tmp_path)
    assert not fake_wandb.runs
    assert not fake_wandb.logins


@pytest.mark.parametrize("exported_key", [None, "shell-key"])
def test_wandb_loads_project_dotenv_without_overriding_shell(
    tmp_path, fake_experiment, fake_wandb, monkeypatch, exported_key
):
    (tmp_path / ".env").write_text("WANDB_API_KEY=file-key\n")
    if exported_key is None:
        monkeypatch.delenv("WANDB_API_KEY")
    else:
        monkeypatch.setenv("WANDB_API_KEY", exported_key)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    run_search(tmp_path, wandb_enabled=True)
    assert fake_wandb.logins == [{"key": exported_key or "file-key", "verify": True}]


def test_wandb_offline_mode_can_be_loaded_from_dotenv(
    tmp_path, fake_experiment, fake_wandb, monkeypatch
):
    (tmp_path / ".env").write_text("WANDB_MODE=offline\n")
    monkeypatch.delenv("WANDB_API_KEY")
    run_search(tmp_path, wandb_enabled=True)
    assert not fake_wandb.logins
    assert len(fake_wandb.runs) == 12


def test_wandb_offline_does_not_require_credentials_or_print_online_links(
    tmp_path, fake_experiment, fake_wandb, monkeypatch
):
    output = StringIO()
    monkeypatch.setattr(search, "CONSOLE", Console(file=output, width=240, color_system=None))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY")
    run_search(tmp_path, wandb_enabled=True)
    assert len(fake_wandb.runs) == 12
    assert not fake_wandb.logins
    assert "W&B offline:" in output.getvalue()
    assert "W&B project:" not in output.getvalue()


def test_wandb_missing_credentials_fail_before_data_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(search, "get_device", lambda: "cpu")
    monkeypatch.setattr(search, "download_pet_ner", lambda: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        run_search(tmp_path, wandb_enabled=True)


def test_wandb_closes_all_initialized_runs_on_training_failure(
    tmp_path, fake_experiment, fake_wandb, monkeypatch
):
    original = search.run_metric_comparisons

    def fail_after_progress(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("training failed")

    monkeypatch.setattr(search, "run_metric_comparisons", fail_after_progress)
    with pytest.raises(RuntimeError, match="training failed"):
        run_search(tmp_path, wandb_enabled=True)
    assert len(fake_wandb.runs) == 6
    assert all(run.exit_codes == [1] for run in fake_wandb.runs)


def test_logging_can_be_enabled_when_resuming_legacy_plan(
    tmp_path, fake_experiment, fake_wandb, monkeypatch
):
    original = search.run_metric_comparisons
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "run_metric_comparisons", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_search(tmp_path)
    path = tmp_path / "example" / "plan.json"
    plan = json.loads(path.read_text())
    plan["fixed_config"].update(wandb_enabled=False, wandb_project="old", wandb_run_name="")
    path.write_text(json.dumps(plan))
    old_plan = path.read_bytes()
    monkeypatch.setattr(search, "run_metric_comparisons", original)
    run_search(tmp_path, wandb_enabled=True, wandb_project="pet")
    assert path.read_bytes() == old_plan
    assert len(fake_wandb.runs) == 6  # only unfinished configuration is logged
    assert all(run.settings["project"] == "pet" for run in fake_wandb.runs)


def test_unsupported_bf16_fails_before_loading_data_or_creating_a_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        search, "download_pet_ner", lambda: pytest.fail("must fail before download")
    )
    with pytest.raises(ValueError, match="native BF16"):
        run_search(tmp_path, precision="bf16")
    assert list(tmp_path.iterdir()) == []


def test_progress_tracks_each_seed_and_resets_between_configurations(
    tmp_path, fake_experiment, monkeypatch
):
    displays = []
    original_progress = search.Progress

    def make_progress(*args, **kwargs):
        display = original_progress(*args, **kwargs, disable=True)
        displays.append(display)
        return display

    monkeypatch.setattr(search, "Progress", make_progress)
    original_run = search.run_metric_comparisons
    snapshots = []
    starts = []

    def inspect_run(*args, **kwargs):
        display = displays[0]
        starts.append([task.completed for task in display.tasks])
        callback = kwargs["progress_callback"]

        def inspect_progress(metric, rows):
            callback(metric, rows)
            snapshots.append(
                [(task.completed, task.total, task.description) for task in display.tasks]
            )

        return original_run(*args, **{**kwargs, "progress_callback": inspect_progress})

    monkeypatch.setattr(search, "run_metric_comparisons", inspect_run)
    run_search(tmp_path)
    assert starts == [[index, 0, 0] for index in (0, 3)]
    # Seed 0 is finished while seed 1 is still waiting, then seed 1 catches up.
    assert snapshots[1][1] == (1, 1, "Seed 0 · entropy · round 1/1")
    assert snapshots[1][2] == (0, 1, "Seed 1 · waiting / bootstrap")
    assert snapshots[1][0][0] == pytest.approx(0.5)
    assert snapshots[2][1] == snapshots[1][1]
    assert snapshots[2][2] == (0, 1, "Seed 1 · entropy · round 0/1")
    assert snapshots[2][0][0] == pytest.approx(0.75)
    assert displays[0].tasks[0].completed == displays[0].tasks[0].total == 6
    assert displays[0].tasks[0].description == "Configurations 2/2"
