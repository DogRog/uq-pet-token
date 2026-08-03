import json

import pandas as pd
import pytest

from uq_pet.config import ArmConfig, ExperimentConfig, load_config
from uq_pet.main import build_parser, make_run_dir, stage_select, summarize

ANLP = "avg_neg_logprob_filtered"
PURE = "avg_neg_logprob_pure"


def make_results(uncertainty_f1, random_f1, budget: float = 10.0) -> pd.DataFrame:
    """One budget, two arms, one row per seed."""
    rows = []
    for arm, scores in ((ANLP, uncertainty_f1), ("random", random_f1)):
        for seed, f1 in enumerate(scores):
            rows.append(
                {
                    "budget_pct": budget,
                    "arm": arm,
                    "seed": seed,
                    "n_train": int(328 * budget / 100),
                    "entity_f1": f1,
                    "entity_precision": f1,
                    "entity_recall": f1,
                    "token_accuracy": f1,
                    "per_type_f1": {"Actor": f1, "Activity": 0.5},
                }
            )
    return pd.DataFrame(rows)


def uncertainty_arm(strategy: str = ANLP) -> ArmConfig:
    return ArmConfig(strategy=strategy)


def scored_records(n: int) -> list[dict]:
    """n records whose scores strictly decrease with index."""
    return [
        {"idx": i, "choices": [{"logprobs": [{"token": "1", "logprob": -float(n - i)}]}]}
        for i in range(n)
    ]


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
    cfg = ExperimentConfig(budget_pct=[7, 14], train_seeds=[4], arms=[uncertainty_arm(PURE)])
    run_dir = make_run_dir(cfg, tmp_path / "myrun.yaml", tmp_path / "results")

    assert run_dir.name.startswith("myrun_")
    assert (run_dir / "figures").is_dir()
    assert load_config(run_dir / "config.yaml") == cfg


def test_make_run_dir_honours_run_name(tmp_path):
    run_dir = make_run_dir(ExperimentConfig(), tmp_path / "c.yaml", tmp_path, run_name="custom")
    assert run_dir.name.startswith("custom_")


# --- selection ----------------------------------------------------------------


def test_stage_select_gives_both_arms_the_same_budget(sample_examples):
    cfg = ExperimentConfig(
        budget_pct=[50], train_seeds=[0], arms=[ArmConfig("random"), uncertainty_arm(PURE)]
    )
    cells = stage_select(cfg, scored_records(len(sample_examples)), sample_examples)

    assert set(cells) == {(50.0, "random"), (50.0, PURE)}
    assert len({len(v) for v in cells.values()}) == 1


def test_stage_select_produces_a_cell_per_budget_and_arm(sample_examples):
    cfg = ExperimentConfig(
        budget_pct=[25, 50, 75],
        train_seeds=[0],
        arms=[ArmConfig("random"), uncertainty_arm(PURE)],
    )
    cells = stage_select(cfg, scored_records(len(sample_examples)), sample_examples)

    assert len(cells) == 3 * 2
    for budget, expected_n in ((25.0, 1), (50.0, 2), (75.0, 3)):
        sizes = {len(cells[budget, arm]) for arm in cfg.arm_labels()}
        assert sizes == {expected_n}, budget


def test_stage_select_uncertainty_is_nested_across_budgets(sample_examples):
    """Top-n of one fixed ranking, so a bigger budget is a superset."""
    cfg = ExperimentConfig(budget_pct=[25, 75], train_seeds=[0], arms=[uncertainty_arm(PURE)])
    cells = stage_select(cfg, scored_records(len(sample_examples)), sample_examples)

    small = cells[25.0, PURE]
    large = cells[75.0, PURE]
    assert all(ex in large for ex in small)


def test_stage_select_raises_when_cache_is_short(sample_examples):
    cfg = ExperimentConfig(budget_pct=[100], train_seeds=[0], arms=[uncertainty_arm(PURE)])
    records = [{"idx": 0, "choices": [{"logprobs": [{"token": "1", "logprob": -1.0}]}]}]
    with pytest.raises(ValueError, match="incomplete"):
        stage_select(cfg, records, sample_examples)


def test_stage_select_single_random_arm_needs_no_records(sample_examples):
    """arms: [random] must run with an empty LLM cache — the control scores the pool."""
    cfg = ExperimentConfig(budget_pct=[50], arms=[ArmConfig("random")], train_seeds=[0])
    cells = stage_select(cfg, [], sample_examples)
    assert list(cells) == [(50.0, "random")]
    assert len(cells[50.0, "random"]) == 2


def test_stage_select_random_draws_from_the_pool_not_from_the_cache(sample_examples):
    """A short cache limits the metric arms, never the control it is compared against."""
    cfg = ExperimentConfig(budget_pct=[100], arms=[ArmConfig("random")], train_seeds=[0])
    cells = stage_select(cfg, scored_records(1), sample_examples)
    assert sorted(cells[100.0, "random"], key=sample_examples.index) == sample_examples


# --- summarize ----------------------------------------------------------------


def test_summarize_computes_the_gap_against_random():
    results = make_results([0.6, 0.7], [0.4, 0.5])
    summary, _per_type, gaps = summarize(results)

    assert gaps == {10.0: {ANLP: pytest.approx(0.2)}}
    assert summary.loc[(10.0, ANLP), ("entity_f1", "mean")] == pytest.approx(0.65)
    assert summary.loc[(10.0, "random"), ("entity_f1", "mean")] == pytest.approx(0.45)


def test_summarize_negative_gap():
    _, _, gaps = summarize(make_results([0.3], [0.5]))
    assert gaps[10.0][ANLP] == pytest.approx(-0.2)


def test_summarize_reports_a_gap_per_budget():
    results = pd.concat(
        [make_results([0.6], [0.4], budget=5.0), make_results([0.5], [0.55], budget=25.0)]
    )
    _, _, gaps = summarize(results)
    assert set(gaps) == {5.0, 25.0}
    assert gaps[5.0][ANLP] == pytest.approx(0.2)
    assert gaps[25.0][ANLP] == pytest.approx(-0.05)


def test_summarize_without_a_random_arm_reports_no_gaps():
    results = make_results([0.6], [0.4])
    results = results[results["arm"] != "random"]
    _, _, gaps = summarize(results)
    assert gaps == {}


def test_summarize_handles_several_uncertainty_arms():
    results = make_results([0.6], [0.4])
    extra = make_results([0.5], [0.4])
    extra = extra[extra["arm"] == ANLP].assign(arm=PURE)
    results = pd.concat([results, extra], ignore_index=True)

    _, _, gaps = summarize(results)
    assert gaps[10.0] == {
        ANLP: pytest.approx(0.2),
        PURE: pytest.approx(0.1),
    }


# --- outputs ------------------------------------------------------------------


def test_write_outputs_creates_every_artifact(tmp_path, sample_examples):
    from uq_pet.main import write_outputs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6, 0.7], [0.4, 0.5])
    summary, per_type, gaps = summarize(results)
    cells = {(10.0, ANLP): sample_examples[:2], (10.0, "random"): sample_examples[2:]}

    write_outputs(run_dir, cells, results, summary, per_type, gaps, {"n_cached": 328})

    for name in ("selection.json", "results.csv", "summary.csv", "per_type_f1.csv", "metrics.json"):
        assert (run_dir / name).exists(), name

    selection = json.loads((run_dir / "selection.json").read_text())
    assert selection["10"][ANLP] == ["doc-1::3", "doc-2::3"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_cached"] == 328
    assert metrics["budgets"] == [10.0]
    assert metrics["by_budget"]["10"]["gap_vs_random"][ANLP] == pytest.approx(0.2)
    assert metrics["by_budget"]["10"]["entity_f1"][ANLP]["mean"] == pytest.approx(0.65)

    # per_type_f1 is a dict column and must not land in the flat CSV
    columns = pd.read_csv(run_dir / "results.csv").columns
    assert "per_type_f1" not in columns
    assert "budget_pct" in columns


def test_write_outputs_nests_selection_by_budget(tmp_path, sample_examples):
    from uq_pet.main import write_outputs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = pd.concat(
        [make_results([0.6], [0.4], budget=5.0), make_results([0.5], [0.55], budget=25.0)]
    )
    summary, per_type, gaps = summarize(results)
    cells = {
        (5.0, ANLP): sample_examples[:1],
        (5.0, "random"): sample_examples[1:2],
        (25.0, ANLP): sample_examples[:3],
        (25.0, "random"): sample_examples[1:],
    }
    write_outputs(run_dir, cells, results, summary, per_type, gaps, {})

    selection = json.loads((run_dir / "selection.json").read_text())
    assert set(selection) == {"5", "25"}
    assert set(selection["5"]) == {ANLP, "random"}

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["budgets"] == [5.0, 25.0]
    assert metrics["by_budget"]["25"]["gap_vs_random"][ANLP] == pytest.approx(-0.05)


def test_metrics_json_is_strictly_valid_with_a_single_seed(tmp_path, sample_examples):
    """A one-seed run has no std; bare NaN would make the file unparseable."""
    from uq_pet.main import write_outputs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = make_results([0.6], [0.4])
    summary, per_type, gaps = summarize(results)
    write_outputs(run_dir, {(10.0, ANLP): sample_examples[:1]}, results, summary, per_type, gaps, {})

    raw = (run_dir / "metrics.json").read_text()
    assert "NaN" not in raw
    metrics = json.loads(raw)  # would raise on NaN with parse_constant left default
    assert metrics["by_budget"]["10"]["entity_f1"][ANLP]["std"] is None


