"""Check the follow-up pilot's gap check on fake runs, without training."""

import json
from pathlib import Path

import pytest

import pilot_gap

CONFIG = Path(__file__).resolve().parents[1] / "configs/pet/pilot/bert_base_constant_avg.json"
POOL = [{"tokens": ["a"], "ner_tags": [0]}]
TEST = [{"tokens": ["secret"], "ner_tags": [0]}]


def rows(gap: float) -> list[dict]:
    return [
        {"seed": 0, "arm": arm, "round": idx, "percent_acquired": 100.0 * idx, "entity_f1": f1}
        for idx in (0, 1)
        for arm, f1 in (("uncertainty", 0.5 + gap), ("random", 0.5))
    ]


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    root = tmp_path / "sweep"
    root.mkdir()
    (root / "best_config.json").write_text(json.dumps({"learning_rate": 3e-5, "k": 64}))
    config = pilot_gap._sweep(CONFIG)[0]
    monkeypatch.setattr(pilot_gap, "_sweep", lambda path: (config, root))
    monkeypatch.setattr(
        pilot_gap, "load_splits", lambda dataset, percent: ((POOL, POOL, {}, TEST), {})
    )
    monkeypatch.setattr(
        pilot_gap, "split_tuning_pool", lambda *args, **kwargs: (POOL, {}, POOL, {})
    )
    calls = []

    def run(seed, pool, gold, evaluation, **settings):
        calls.append((evaluation, settings["weight_averaging"], settings["learning_rate"]))
        metric = settings["uq_metrics"][0]
        results = rows(0.04 if settings["weight_averaging"] else 0.02)
        return {metric: (results, [], [[["O"]]] * len(results))}

    monkeypatch.setattr(pilot_gap, "run_metric_comparisons", run)
    return root, calls


def test_gap_check_runs_both_variants_on_the_holdout_at_the_winner(sweep):
    root, calls = sweep
    gaps = pilot_gap.run_gap_check(CONFIG)
    assert gaps == {"constant": pytest.approx(0.02), "averaged": pytest.approx(0.04)}
    assert [averaging for _, averaging, _ in calls] == [False, True]
    assert all(evaluation is POOL and rate == 3e-5 for evaluation, _, rate in calls)
    saved = json.loads((root / "gap_check" / "averaged.json").read_text())
    assert (root / saved["run_dir"] / "results.csv").is_file()


def test_gap_check_skips_finished_variants_and_needs_a_winner(sweep):
    root, calls = sweep
    pilot_gap.run_gap_check(CONFIG)
    calls.clear()
    (root / "gap_check" / "averaged.json").unlink()
    assert pilot_gap.run_gap_check(CONFIG)["constant"] == pytest.approx(0.02)
    assert [averaging for _, averaging, _ in calls] == [True]
    (root / "best_config.json").unlink()
    with pytest.raises(ValueError, match="no frozen winner"):
        pilot_gap.run_gap_check(CONFIG)
