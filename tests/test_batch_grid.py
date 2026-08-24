import csv
import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

GRID_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bert_token_uq_grid.py"
PARALLEL_GRID_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "parallel_bert_token_uq_grid.py"
)
GRID_SPEC = importlib.util.spec_from_file_location("bert_token_uq_grid", GRID_PATH)
assert GRID_SPEC is not None and GRID_SPEC.loader is not None
grid = importlib.util.module_from_spec(GRID_SPEC)
GRID_SPEC.loader.exec_module(grid)

SEARCH_SPACE = grid.SEARCH_SPACE
build_runs = grid.build_runs
params_to_cli_args = grid.params_to_cli_args
require_wandb_credentials = grid.require_wandb_credentials
sample_search_configs = grid.sample_search_configs
write_combined_results = grid.write_combined_results


def test_search_config_sampling_is_unique_and_reproducible():
    first = sample_search_configs(12, random.Random(7))
    second = sample_search_configs(12, random.Random(7))

    assert first == second
    assert len(first) == 12
    assert len({tuple(config.items()) for config in first}) == 12
    assert SEARCH_SPACE["uq_metric"] == ("entropy", "least_confidence", "margin")
    assert SEARCH_SPACE["k"] == (8, 16, 32, 64)


def test_sampled_configs_are_crossed_with_every_checkpoint_and_seed():
    configs = sample_search_configs(2, random.Random(3))
    runs = build_runs(
        configs,
        ("model-a", "model-b"),
        (0, 1, 2),
        wandb_enabled=False,
        wandb_project="test-project",
    )

    assert len(runs) == 12
    assert len({run["run_id"] for run in runs}) == 12
    assert {run["params"]["checkpoint"] for run in runs} == {"model-a", "model-b"}
    assert {run["params"]["model_seeds"] for run in runs} == {0, 1, 2}
    assert all(run["params"]["wandb_enabled"] is False for run in runs)


def test_cli_args_use_marimo_names_and_boolean_values():
    args = params_to_cli_args({"model_seeds": 4, "learning_rate": 2e-5, "wandb_enabled": False})

    assert args == [
        "--model-seeds",
        "4",
        "--learning-rate",
        "2e-05",
        "--wandb-enabled",
        "false",
    ]


def test_online_wandb_launch_requires_api_key():
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({})
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({"WANDB_API_KEY": "   "})


def test_wandb_preflight_accepts_key_or_offline_mode():
    require_wandb_credentials({"WANDB_API_KEY": "configured"})
    require_wandb_credentials({"WANDB_MODE": "offline"})


def test_combined_results_include_run_configuration(tmp_path):
    run_output = tmp_path / "run-output"
    run_output.mkdir()
    (run_output / "results.csv").write_text("seed,arm,round,entity_f1\n0,random,0,0.1\n")
    completed_runs = [
        {
            "run_id": "model-seed0",
            "params": {"checkpoint": "model", "model_seeds": 0},
            "output_dir": str(run_output),
        }
    ]

    output_path = write_combined_results(tmp_path, completed_runs)

    assert output_path == tmp_path / "combined_results.csv"
    with output_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "run_id": "model-seed0",
            "config_checkpoint": "model",
            "config_model_seeds": "0",
            "seed": "0",
            "arm": "random",
            "round": "0",
            "entity_f1": "0.1",
        }
    ]


def test_parallel_launcher_dry_run_partitions_all_checkpoints():
    environment = os.environ.copy()
    environment["COLUMNS"] = "240"
    completed = subprocess.run(
        [sys.executable, str(PARALLEL_GRID_PATH), "--count", "1", "--seeds", "0"],
        cwd=GRID_PATH.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Workers" in completed.stdout
    assert "Total runs" in completed.stdout
    assert "worker-a" in completed.stdout
    assert "worker-b" in completed.stdout
    assert "distilbert-base-cased" in completed.stdout
    assert "answerdotai/ModernBERT-base" in completed.stdout
    assert "Dry run complete" in completed.stdout
