import json

import pandas as pd
import pytest

import uq_pet.main as pipeline
from uq_pet.config import ArmConfig, ExperimentConfig, load_config
from uq_pet.main import build_parser, make_run_dir, stage_select, summarize, write_outputs

ENTROPY = "mean_token_entropy"


def scored_records(examples):
    """Strictly increasing entropy with pool index."""
    records = []
    for idx, example in enumerate(examples):
        probability = 0.01 + idx * 0.1
        records.append(
            {
                "idx": idx,
                "n_tokens": len(example["tokens"]),
                "probabilities": [[1 - probability, probability] for _ in example["tokens"]],
            }
        )
    return records


def make_results(uncertainty_f1, random_f1, budget=10.0):
    rows = []
    for arm, values in ((ENTROPY, uncertainty_f1), ("random", random_f1)):
        for seed, value in enumerate(values):
            rows.append(
                {
                    "budget_pct": budget,
                    "arm": arm,
                    "seed": seed,
                    "n_seed": 5,
                    "n_selected": 32,
                    "n_train": 37,
                    "entity_f1": value,
                    "entity_precision": value,
                    "entity_recall": value,
                    "token_accuracy": value,
                    "per_type_f1": {"Actor": value, "Activity": 0.5},
                }
            )
    return pd.DataFrame(rows)


def test_parser_requires_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_flags():
    args = build_parser().parse_args(["--config", "c.yaml", "--dry-run", "--no-plot", "-v"])
    assert args.dry_run and args.no_plot and args.verbose


def test_make_run_dir_snapshot_round_trips(tmp_path):
    cfg = ExperimentConfig(budget_pct=[7, 14], model_seeds=[4])
    run_dir = make_run_dir(cfg, tmp_path / "myrun.yaml", tmp_path / "results")
    assert run_dir.name.startswith("myrun_")
    assert load_config(run_dir / "config.yaml") == cfg


def test_make_run_dir_never_overwrites_a_collision(tmp_path, monkeypatch):
    real_datetime = pipeline.datetime

    class FixedDateTime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 8, 6, 12, 0, 0, 123456)

    monkeypatch.setattr(pipeline, "datetime", FixedDateTime)
    make_run_dir(ExperimentConfig(), tmp_path / "c.yaml", tmp_path)
    with pytest.raises(FileExistsError):
        make_run_dir(ExperimentConfig(), tmp_path / "c.yaml", tmp_path)


def test_stage_select_has_seed_in_every_cell_key(sample_examples):
    cfg = ExperimentConfig(
        budget_pct=[50],
        arms=[ArmConfig("random"), ArmConfig(ENTROPY)],
        model_seeds=[7],
    )
    cells = stage_select(cfg, scored_records(sample_examples), sample_examples, model_seed=7)
    assert set(cells) == {(50.0, "random", 7), (50.0, ENTROPY, 7)}
    assert {len(value) for value in cells.values()} == {2}


def test_uncertainty_selections_are_nested_per_seed(sample_examples):
    cfg = ExperimentConfig(budget_pct=[25, 75], arms=[ArmConfig(ENTROPY)], model_seeds=[3])
    cells = stage_select(cfg, scored_records(sample_examples), sample_examples, model_seed=3)
    assert all(example in cells[75.0, ENTROPY, 3] for example in cells[25.0, ENTROPY, 3])


def test_default_random_draw_changes_with_model_seed(sample_examples):
    cfg = ExperimentConfig(budget_pct=[50], arms=[ArmConfig("random")], model_seeds=[0, 1])
    first = stage_select(cfg, [], sample_examples, model_seed=0)
    second = stage_select(cfg, [], sample_examples, model_seed=1)
    assert first[50.0, "random", 0] != second[50.0, "random", 1]


def test_explicit_random_seed_overrides_model_seed(sample_examples):
    cfg = ExperimentConfig(
        budget_pct=[50],
        arms=[ArmConfig("random", {"seed": 9})],
        model_seeds=[0, 1],
    )
    first = stage_select(cfg, [], sample_examples, model_seed=0)
    second = stage_select(cfg, [], sample_examples, model_seed=1)
    label = "random:9"
    assert first[50.0, label, 0] == second[50.0, label, 1]


def test_summarize_computes_gap_against_random():
    summary, _per_type, gaps = summarize(make_results([0.6, 0.7], [0.4, 0.5]))
    assert gaps == {10.0: {ENTROPY: pytest.approx(0.2)}}
    assert summary.loc[(10.0, ENTROPY), ("entity_f1", "mean")] == pytest.approx(0.65)


def test_write_outputs_records_selection_per_model_seed(tmp_path, sample_examples):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6, 0.7], [0.4, 0.5])
    summary, per_type, gaps = summarize(results)
    cells = {
        (10.0, ENTROPY, 0): sample_examples[:2],
        (10.0, "random", 0): sample_examples[2:],
        (10.0, ENTROPY, 1): sample_examples[1:3],
        (10.0, "random", 1): sample_examples[:2],
    }
    write_outputs(run_dir, cells, results, summary, per_type, gaps, {"scorer": {}})

    selection = json.loads((run_dir / "selection.json").read_text())
    assert set(selection) == {"0", "1"}
    assert selection["0"]["10"][ENTROPY] == ["doc-1::3", "doc-2::3"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["model_seeds"] == [0, 1]
    assert metrics["by_budget"]["10"]["n_seed"] == 5
    assert metrics["by_budget"]["10"]["n_selected"] == 32


def test_single_seed_writes_null_standard_deviation(tmp_path, sample_examples):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6], [0.4])
    summary, per_type, gaps = summarize(results)
    cells = {
        (10.0, ENTROPY, 0): sample_examples[:1],
        (10.0, "random", 0): sample_examples[1:2],
    }
    write_outputs(run_dir, cells, results, summary, per_type, gaps, {})
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["by_budget"]["10"]["entity_f1"][ENTROPY]["std"] is None
