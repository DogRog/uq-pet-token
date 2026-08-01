import json

import pandas as pd
import pytest

from uq_pet.config import ExperimentConfig, load_config
from uq_pet.main import build_parser, make_run_dir, stage_select, summarize


def make_results(uncertainty_f1, random_f1) -> pd.DataFrame:
    rows = []
    for arm, scores in (("uncertainty", uncertainty_f1), ("random", random_f1)):
        for seed, f1 in enumerate(scores):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "n_train": 32,
                    "entity_f1": f1,
                    "entity_precision": f1,
                    "entity_recall": f1,
                    "token_accuracy": f1,
                    "per_type_f1": {"Actor": f1, "Activity": 0.5},
                }
            )
    return pd.DataFrame(rows)


# --- parser -------------------------------------------------------------------


def test_parser_requires_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_defaults():
    args = build_parser().parse_args(["--config", "configs/smoke.yaml"])
    assert args.skip_scoring is False
    assert args.dry_run is False
    assert args.limit is None
    assert args.run_name is None


def test_parser_flags():
    args = build_parser().parse_args(
        ["--config", "c.yaml", "--skip-scoring", "--limit", "5", "--dry-run", "-v"]
    )
    assert (args.skip_scoring, args.limit, args.dry_run, args.verbose) == (True, 5, True, True)


# --- run dir ------------------------------------------------------------------


def test_make_run_dir_snapshot_round_trips(tmp_path):
    cfg = ExperimentConfig(budget_pct=7, train_seeds=[4])
    run_dir = make_run_dir(cfg, tmp_path / "myrun.yaml", tmp_path / "results")

    assert run_dir.name.startswith("myrun_")
    assert (run_dir / "figures").is_dir()
    assert load_config(run_dir / "config.yaml") == cfg


def test_make_run_dir_honours_run_name(tmp_path):
    run_dir = make_run_dir(ExperimentConfig(), tmp_path / "c.yaml", tmp_path, run_name="custom")
    assert run_dir.name.startswith("custom_")


# --- selection ----------------------------------------------------------------


def test_stage_select_gives_both_arms_the_same_budget(sample_examples, sample_records):
    cfg = ExperimentConfig(budget_pct=50, train_seeds=[0], score="pure")
    records = [
        {"idx": i, "choices": [{"logprobs": [{"token": "1", "logprob": -float(i + 1)}]}]}
        for i in range(len(sample_examples))
    ]
    arms = stage_select(cfg, records, sample_examples)

    assert set(arms) == {"uncertainty", "random"}
    assert len(arms["uncertainty"]) == len(arms["random"]) == 2


def test_stage_select_raises_when_cache_is_short(sample_examples):
    cfg = ExperimentConfig(budget_pct=100, train_seeds=[0], score="pure")
    records = [{"idx": 0, "choices": [{"logprobs": [{"token": "1", "logprob": -1.0}]}]}]
    with pytest.raises(ValueError, match="incomplete"):
        stage_select(cfg, records, sample_examples)


def test_stage_select_single_arm_needs_no_scores(sample_examples):
    cfg = ExperimentConfig(budget_pct=50, arms=["random"], train_seeds=[0])
    arms = stage_select(cfg, [], sample_examples)
    assert list(arms) == ["random"]
    assert len(arms["random"]) == 2


# --- summarize ----------------------------------------------------------------


def test_summarize_computes_the_gap():
    results = make_results([0.6, 0.7], [0.4, 0.5])
    summary, per_type, gap = summarize(results)

    assert gap == pytest.approx(0.2)
    assert summary.loc["uncertainty", ("entity_f1", "mean")] == pytest.approx(0.65)
    assert summary.loc["random", ("entity_f1", "mean")] == pytest.approx(0.45)


def test_summarize_per_type_has_a_delta_column():
    _, per_type, _ = summarize(make_results([0.6, 0.6], [0.4, 0.4]))
    assert list(per_type.index) == ["Activity", "Actor"]
    assert per_type.loc["Actor", "delta"] == pytest.approx(0.2)
    assert per_type.loc["Activity", "delta"] == pytest.approx(0.0)


def test_summarize_negative_gap():
    _, _, gap = summarize(make_results([0.3], [0.5]))
    assert gap == pytest.approx(-0.2)


# --- outputs ------------------------------------------------------------------


def test_write_outputs_creates_every_artifact(tmp_path, sample_examples):
    from uq_pet.main import write_outputs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6, 0.7], [0.4, 0.5])
    summary, per_type, gap = summarize(results)
    arms = {"uncertainty": sample_examples[:2], "random": sample_examples[2:]}

    write_outputs(run_dir, arms, results, summary, per_type, gap, {"n_cached": 328})

    for name in ("selection.json", "results.csv", "summary.csv", "per_type_f1.csv", "metrics.json"):
        assert (run_dir / name).exists(), name

    selection = json.loads((run_dir / "selection.json").read_text())
    assert selection["uncertainty"] == ["doc-1::3", "doc-2::3"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["gap"] == pytest.approx(0.2)
    assert metrics["n_cached"] == 328
    assert metrics["entity_f1"]["uncertainty"]["mean"] == pytest.approx(0.65)


def test_metrics_json_is_strictly_valid_with_a_single_seed(tmp_path, sample_examples):
    """A one-seed run has no std; bare NaN would make the file unparseable."""
    from uq_pet.main import write_outputs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6], [0.4])
    summary, per_type, gap = summarize(results)
    write_outputs(run_dir, {"uncertainty": sample_examples[:1]}, results, summary, per_type, gap, {})

    raw = (run_dir / "metrics.json").read_text()
    assert "NaN" not in raw
    metrics = json.loads(raw)  # would raise on NaN with parse_constant left default
    assert metrics["entity_f1"]["uncertainty"]["std"] is None

    # per_type_f1 is a dict column and must not land in the flat CSV
    assert "per_type_f1" not in pd.read_csv(run_dir / "results.csv").columns
