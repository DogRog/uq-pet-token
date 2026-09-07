"""Compare worker counts on a fixed, short search workload without creating a study."""

import argparse
import json
import statistics
import time
from pathlib import Path

from pydantic import Field, field_validator

from bert_token_uq_search import SearchConfig, _quiet_search_output, make_tuning_split
from uq_pet.active_learning import run_active_learning
from uq_pet.pet_data import download_pet_ner, load_pet_splits
from uq_pet.token_model import get_device


class BenchmarkConfig(SearchConfig):
    worker_counts: list[int] = [1, 2, 5]
    repeats: int = Field(default=2, ge=1)
    benchmark_pool_percent: float = Field(default=5, gt=0, le=100)

    @field_validator("worker_counts")
    @classmethod
    def validate_workers(cls, values):
        if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("worker_counts must contain distinct positive integers")
        return values


@_quiet_search_output()
def benchmark(config):
    experiment = config.experiment_config()
    if max(config.worker_counts) > len(experiment.model_seeds):
        raise ValueError("worker_counts cannot exceed the number of model seeds")
    experiment = experiment.model_copy(
        update={"max_pool_percent": min(experiment.max_pool_percent, config.benchmark_pool_percent)}
    )
    seed, pool, gold, _test = load_pet_splits(download_pet_ner())
    pool, gold, validation = make_tuning_split(
        pool, gold, validation_fraction=config.validation_fraction, seed=config.validation_seed
    )
    device = get_device()
    durations = {workers: [] for workers in config.worker_counts}
    # Reverse alternate runs to reduce the advantage from filesystem/model warmup.
    for repeat in range(config.repeats):
        order = config.worker_counts if repeat % 2 == 0 else config.worker_counts[::-1]
        for workers in order:
            started = time.perf_counter()
            run_active_learning(
                seed,
                pool,
                gold,
                validation,
                **{**experiment.active_learning_kwargs(), "seed_workers": workers},
                device=device,
            )
            durations[workers].append(time.perf_counter() - started)
    return {
        "device": str(device),
        "experiment": experiment.model_dump(),
        "validation_fraction": config.validation_fraction,
        "validation_seed": config.validation_seed,
        "timings": [
            {
                "seed_workers": workers,
                "seconds": values,
                "median_seconds": statistics.median(values),
            }
            for workers, values in durations.items()
        ],
        "fastest_seed_workers": min(
            durations, key=lambda workers: statistics.median(durations[workers])
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = BenchmarkConfig.model_validate_json(args.config.read_text())
    print(json.dumps(benchmark(config), indent=2))
