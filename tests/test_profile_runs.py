"""Check that the scaling profile records running out of GPU memory instead of crashing."""

import pytest

import profile_runs


def test_scaling_stops_at_the_first_process_count_out_of_memory(monkeypatch):
    def run(*splits, model_seeds, **settings):
        if len(model_seeds) > 2:
            raise RuntimeError("Seed 2 failed:\ntorch.OutOfMemoryError: CUDA out of memory.")
        return {"entropy": ([{"round": 0}, {"round": 1}, {"round": 1}], [], [])}

    monkeypatch.setattr(profile_runs.active_learning, "run_metric_comparisons", run)
    frame = profile_runs.scaling_profile((), {}, [1, 2, 4, 6], threads=None, mps=True)
    assert frame["processes"].to_list() == [1, 2, 4]
    assert frame["outcome"].to_list() == ["finished", "finished", "out of memory"]
    assert frame["arm_rounds_per_minute"].null_count() == 1
    assert frame["mps"].to_list() == [True, True, True]


def test_scaling_still_raises_other_errors(monkeypatch):
    def run(*splits, **settings):
        raise RuntimeError("Seed 0 failed:\nValueError: bad config")

    monkeypatch.setattr(profile_runs.active_learning, "run_metric_comparisons", run)
    with pytest.raises(RuntimeError, match="bad config"):
        profile_runs.scaling_profile((), {}, [1], threads=None, mps=None)
