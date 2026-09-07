import importlib
from pathlib import Path


def test_benchmark_runs_same_seeds_and_validation_for_each_worker_count(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("benchmark_seed_workers")
    monkeypatch.setattr(module, "download_pet_ner", lambda: "unused")
    monkeypatch.setattr(module, "load_pet_splits", lambda path: ("seed", "pool", "gold", "test"))
    monkeypatch.setattr(
        module, "make_tuning_split", lambda *args, **kwargs: ("pool", "gold", "validation")
    )
    monkeypatch.setattr(module, "get_device", lambda: "cpu")
    calls = []
    monkeypatch.setattr(
        module, "run_active_learning", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    ticks = iter(range(12))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    result = module.benchmark(module.BenchmarkConfig(trials=100, model_seeds=[0, 1, 2, 3, 4]))
    assert [kwargs["seed_workers"] for _, kwargs in calls] == [1, 2, 5, 5, 2, 1]
    assert all(args == ("seed", "pool", "gold", "validation") for args, _ in calls)
    assert all(kwargs["model_seeds"] == [0, 1, 2, 3, 4] for _, kwargs in calls)
    assert all(kwargs["max_pool_percent"] == 5 for _, kwargs in calls)
    assert all(row["seconds"] == [1, 1] for row in result["timings"])
