"""Measure where a seed's time goes and how throughput grows with parallel seeds.

Part A runs one seed of a short paired comparison (top-K entropy against random) and
times each phase, with the device synchronized around every timed call, plus how long a
fresh seed process takes to import and start the device. Part B runs the same short job
with 1, 2, 4, 6, and 8 seed processes side by side, sampling GPU use (nvidia-smi) and
CPU use (Linux), to show where throughput stops growing and which resource limits it.
It stops at the first process count that runs out of GPU memory and records it, and it
records whether the CUDA MPS daemon was running, which decides whether processes overlap.

The job is a run config (default: PET's frozen ModernBERT best-UQ settings) cut to
--pool-percent of the scoreable pool. Nothing is written under results/; tables print,
and a CSV of each part goes to logs/profile/. Results stay comparable only on one
machine with nothing else running.
"""

import argparse
import multiprocessing
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl
import torch

from run_queue import mps_running
from uq_pet import active_learning, token_model
from uq_pet.config import ExperimentConfig
from uq_pet.data_prep import load_splits
from uq_pet.paths import PROJECT_ROOT
from uq_pet.token_model import get_device, resolve_precision

LOGS = PROJECT_ROOT / "logs" / "profile"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class PhaseTimer:
    """Accumulate synchronized wall time per phase by wrapping module functions."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self._originals = []

    def wrap(self, function, label):
        def timed(*args, **kwargs):
            synchronize(self.device)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                synchronize(self.device)
                name = label(*args, **kwargs) if callable(label) else label
                self.seconds[name] += time.perf_counter() - start
                self.calls[name] += 1

        return timed

    def patch(self, owner, name: str, label) -> None:
        self._originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, self.wrap(getattr(owner, name), label))

    def restore(self) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()


def _startup_child(started: float, device: str, queue) -> None:
    torch.zeros(1, device=device)
    synchronize(torch.device(device))
    queue.put(time.time() - started)


def startup_seconds(device: torch.device) -> float:
    """Seconds from spawning a seed process until it has imported everything and the device works."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_startup_child, args=(time.time(), str(device), queue))
    process.start()
    seconds = queue.get()
    process.join()
    return seconds


def phase_profile(splits: tuple, settings: dict) -> pl.DataFrame:
    """Time one seed of the job by phase."""
    timer = PhaseTimer(settings["device"])
    timer.patch(
        active_learning,
        "train_items",
        lambda model, optimizer, tokenizer, items, **kwargs: (
            "bootstrap training" if "key" not in items[0] else "round training"
        ),
    )
    timer.patch(active_learning, "score_token_uncertainty", "pool scoring")
    timer.patch(active_learning, "evaluate_model", "evaluation")
    timer.patch(active_learning, "load_token_classifier", "model loading")
    timer.patch(token_model, "encode_targets", "  training: batch encoding")
    timer.patch(token_model, "evaluate_predictions", "  evaluation: entity scoring")
    timer.patch(torch.optim.AdamW, "step", "  training: optimizer step")
    try:
        start = time.perf_counter()
        active_learning.run_metric_comparisons(
            *splits, **{**settings, "model_seeds": [0], "seed_workers": 1}, uq_metrics=["entropy"]
        )
        wall = time.perf_counter() - start
    finally:
        timer.restore()
    startup = startup_seconds(settings["device"])
    rows = [
        {"phase": name, "seconds": seconds, "share": seconds / wall, "calls": timer.calls[name]}
        for name, seconds in sorted(timer.seconds.items(), key=lambda item: -item[1])
    ]
    rows.append({"phase": "whole seed", "seconds": wall, "share": 1.0, "calls": 1})
    rows.append({"phase": "seed process start", "seconds": startup, "share": None, "calls": 1})
    return pl.DataFrame(rows)


class LoadSampler:
    """Sample GPU use and memory (nvidia-smi) and CPU use (/proc/stat) about once a second."""

    def __init__(self):
        self.gpu, self.memory, self.cpu = [], [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @staticmethod
    def _cpu_times():
        stat = Path("/proc/stat")
        if not stat.is_file():
            return None
        values = [int(value) for value in stat.read_text().split("\n")[0].split()[1:]]
        return values[3] + values[4], sum(values)

    def _run(self):
        previous = self._cpu_times()
        while not self._stop.wait(1.0):
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.split("\n")[0]
                use, memory = (float(value) for value in output.split(","))
                self.gpu.append(use)
                self.memory.append(memory)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            current = self._cpu_times()
            if previous and current and current[1] > previous[1]:
                idle, total = current[0] - previous[0], current[1] - previous[1]
                self.cpu.append(100 * (1 - idle / total))
            previous = current


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def out_of_memory(error: Exception) -> bool:
    """Whether a run failed for GPU memory, in this process or in a seed process."""
    return isinstance(error, torch.OutOfMemoryError) or "OutOfMemoryError" in str(error)


def scaling_profile(
    splits: tuple, settings: dict, counts: list[int], threads: int | None, mps: bool | None
):
    """Run the job with each number of parallel seed processes and record throughput and load.

    The first count that runs out of GPU memory is recorded and ends the profile.
    """
    rows = []
    for count in counts:
        if threads is not None:
            # Seed workers get the parent's thread count divided by their number.
            torch.set_num_threads(threads * count)
        row = {"processes": count, "threads_per_process": max(1, torch.get_num_threads() // count)}
        with LoadSampler() as load:
            start = time.perf_counter()
            try:
                comparisons = active_learning.run_metric_comparisons(
                    *splits,
                    **{**settings, "model_seeds": list(range(count)), "seed_workers": count},
                    uq_metrics=["entropy"],
                )
            except RuntimeError as error:  # torch.OutOfMemoryError is one too
                if not out_of_memory(error):
                    raise
                comparisons = None
            wall = time.perf_counter() - start
        if comparisons is None:
            rounds = None
        else:
            rounds = sum(result["round"] > 0 for result in comparisons["entropy"][0])
        row |= {
            "outcome": "out of memory" if comparisons is None else "finished",
            "mps": mps,
            "wall_seconds": wall,
            "arm_rounds_per_minute": None if rounds is None else 60 * rounds / wall,
            "per_process": None if rounds is None else 60 * rounds / wall / count,
            "gpu_use_percent": _mean(load.gpu),
            "gpu_memory_peak_mb": max(load.memory) if load.memory else None,
            "cpu_use_percent": _mean(load.cpu),
        }
        rows.append(row)
        print(pl.DataFrame(rows[-1:]))
        if comparisons is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{count} processes ran out of GPU memory; larger counts are skipped.")
            break
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        speedup=pl.col("arm_rounds_per_minute") / pl.col("arm_rounds_per_minute").first()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/pet/best_uq/modernbert_base.json",
        help="Run config whose model and settings the job uses.",
    )
    parser.add_argument("--pool-percent", type=float, default=10.0)
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument(
        "--threads-per-process",
        type=int,
        help="CPU threads per seed process in part B (default: cores divided by processes).",
    )
    parser.add_argument("--skip-phases", action="store_true", help="Skip part A.")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip part B.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.model_validate_json(args.config.read_text())
    device = get_device()
    settings = {
        **config.active_learning_kwargs(),
        "max_pool_percent": args.pool_percent,
        "precision": resolve_precision(config.precision, device),
        "device": device,
    }
    settings.pop("uq_metric")
    splits, _ = load_splits(config.dataset, config.dataset_percent)
    active_learning.CONSOLE.quiet = True
    LOGS.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M%S}-{config.checkpoint.rsplit('/', 1)[-1]}-{config.dataset}"
    mps = mps_running() if device.type == "cuda" else None
    mps_note = {True: "MPS on", False: "MPS off", None: "no MPS"}[mps]
    print(
        f"{config.checkpoint} on {config.dataset} · {args.pool_percent:g}% of the pool · "
        f"{device} · {settings['precision']} · {torch.get_num_threads()} CPU threads · {mps_note}"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        if not args.skip_phases:
            phases = phase_profile(splits, settings)
            phases.write_csv(LOGS / f"{stem}-phases.csv")
            print("\nPart A: one seed by phase\n", phases)
        if not args.skip_scaling:
            scaling = scaling_profile(
                splits, settings, args.processes, args.threads_per_process, mps
            )
            scaling.write_csv(LOGS / f"{stem}-scaling.csv")
            print("\nPart B: parallel seed processes\n", scaling)
    print(f"\nCSV files: {LOGS}/{stem}-*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
