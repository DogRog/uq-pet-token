"""Run experiment stages as a queue of jobs in a fixed number of parallel slots.

Each job runs the same command the per-stage launchers run for one dataset and model,
in its own process with its own log. A model's best-UQ comparisons (top-K, then Gumbel
noise, at the frozen validation winner) wait for its random-baseline tuning; every other
job is independent. Ready jobs start longest first, so the longest datasets and models
are not left running alone at the end. A failed job skips only the jobs that need it.
Every job resumes on a rerun, as the stage CLIs do.

Arguments the queue does not know (for example --dry-run or --seed-workers 2) pass
through to every job.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("tune_random", "best_uq", "supervised", "random_search", "pilot")
DATASETS = ("pet", "quishpi", "medical")
MODELS = ("distilbert", "bert_base", "roberta_base", "deberta_v3_base", "modernbert_base")
PILOT_VARIANTS = ("split", "split_avg")
# Rough relative running times from the 2026-10 rerun: scoreable pool words per dataset,
# tuning hours per model on PET relative to BERT, and hours per stage relative to tuning.
DATASET_COST = {"pet": 5901, "quishpi": 2984, "medical": 3389}
MODEL_COST = {
    "distilbert": 0.8,
    "bert_base": 1.0,
    "roberta_base": 1.1,
    "deberta_v3_base": 2.1,
    "modernbert_base": 1.6,
}
STAGE_COST = {
    "tune_random": 1.0,
    "best_uq": 0.1,
    "supervised": 0.3,
    "random_search": 0.8,
    "pilot": 0.3,
}


@dataclass
class Job:
    name: str
    stage: str
    dataset: str
    model: str
    commands: list[list[str]]
    needs: list[str] = field(default_factory=list)
    # Freeze the tuning winner first; skip the job when there is none yet.
    freeze: Path | None = None

    @property
    def cost(self) -> float:
        return STAGE_COST[self.stage] * DATASET_COST[self.dataset] * MODEL_COST[self.model]


def _uv(script: str, *args: str) -> list[str]:
    return ["uv", "run", f"scripts/{script}", *args]


def build_jobs(stages, datasets, models, extra: list[str]) -> list[Job]:
    """One job per stage, dataset, model, and (for the pilot) variant, with dependencies."""
    jobs = []
    for dataset in datasets:
        for model in models:
            configs = Path("configs") / dataset
            tune = f"tune_random/{dataset}/{model}"
            for stage in stages:
                if stage == "tune_random":
                    config = str(configs / "tune_random" / f"{model}.json")
                    jobs.append(
                        Job(
                            tune,
                            stage,
                            dataset,
                            model,
                            [_uv("run_search.py", "--config", config, *extra)],
                        )
                    )
                elif stage == "best_uq":
                    frozen = str(configs / "best_uq" / f"{model}.json")
                    jobs.append(
                        Job(
                            f"best_uq/{dataset}/{model}",
                            stage,
                            dataset,
                            model,
                            [
                                _uv("run_uq_metrics.py", "--config", frozen, *extra),
                                _uv(
                                    "run_uq_metrics.py",
                                    "--config",
                                    frozen,
                                    "--gumbel-noise",
                                    *extra,
                                ),
                            ],
                            needs=[tune] if "tune_random" in stages else [],
                            freeze=configs / "tune_random" / f"{model}.json",
                        )
                    )
                elif stage == "supervised":
                    config = str(configs / "supervised" / f"{model}.json")
                    jobs.append(
                        Job(
                            f"supervised/{dataset}/{model}",
                            stage,
                            dataset,
                            model,
                            [_uv("run_supervised.py", "--config", config, *extra)],
                        )
                    )
                elif stage == "random_search":
                    config = str(configs / "random_search" / f"{model}.json")
                    jobs.append(
                        Job(
                            f"random_search/{dataset}/{model}",
                            stage,
                            dataset,
                            model,
                            [_uv("run_search.py", "--config", config, *extra)],
                        )
                    )
                elif stage == "pilot":
                    for variant in PILOT_VARIANTS:
                        config = configs / "pilot" / f"{model}_{variant}.json"
                        if config.is_file():
                            jobs.append(
                                Job(
                                    f"pilot/{dataset}/{model}_{variant}",
                                    stage,
                                    dataset,
                                    model,
                                    [_uv("run_search.py", "--config", str(config), *extra)],
                                )
                            )
    return jobs


def priority(job: Job, jobs: list[Job]) -> float:
    """A job's own cost plus that of every job waiting on it, so tuning leads its comparisons."""
    return job.cost + sum(other.cost for other in jobs if job.name in other.needs)


def run_queue(jobs: list[Job], slots: int, execute: Callable[[Job], str]) -> dict[str, str]:
    """Run jobs whose needs succeeded, longest first, at most `slots` at a time.

    `execute` returns "done", "skipped", or "failed"; a job whose need did not finish
    is "blocked".
    """
    known = {job.name for job in jobs}
    pending = sorted(jobs, key=lambda job: (-priority(job, jobs), job.name))
    status: dict[str, str] = {}
    running: dict[Future, Job] = {}
    with ThreadPoolExecutor(max_workers=slots) as executor:
        while pending or running:
            for job in list(pending):
                needs = [need for need in job.needs if need in known]
                if any(status.get(need) in {"failed", "blocked"} for need in needs):
                    status[job.name] = "blocked"
                    pending.remove(job)
                elif len(running) < slots and all(
                    status.get(need) in {"done", "skipped"} for need in needs
                ):
                    running[executor.submit(execute, job)] = job
                    pending.remove(job)
            if not running:
                # Nothing can start: every remaining job waits on a job that will not run.
                status.update({job.name: "blocked" for job in pending})
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                try:
                    status[job.name] = future.result()
                except Exception:
                    status[job.name] = "failed"
    return status


def make_executor(logs: Path, environment: dict[str, str]) -> Callable[[Job], str]:
    """Run a job's commands in order, logging to logs/<stage>-<dataset>-<model>.log."""

    def execute(job: Job) -> str:
        log_path = logs / f"{job.name.replace('/', '-')}.log"
        start = time.monotonic()
        print(f"Started {job.name}: {log_path}", flush=True)
        with log_path.open("a") as log:
            if job.freeze is not None:
                frozen = subprocess.run(
                    _uv("freeze_best_uq.py", str(job.freeze)),
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    env=environment,
                    cwd=PROJECT_ROOT,
                )
                if frozen.returncode != 0:
                    print(f"FAILED {job.name}: could not freeze its winner", flush=True)
                    return "failed"
                if not frozen.stdout.strip():
                    print(f"Skipped {job.name}: no tuning winner yet", flush=True)
                    return "skipped"
            for command in job.commands:
                log.write(f"\n$ {' '.join(command)}\n")
                log.flush()
                if subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=PROJECT_ROOT
                ).returncode:
                    print(f"FAILED {job.name}; last lines of {log_path}:", flush=True)
                    print("".join(log_path.read_text().splitlines(keepends=True)[-20:]), flush=True)
                    return "failed"
        print(f"Finished {job.name} in {(time.monotonic() - start) / 3600:.1f} h", flush=True)
        return "done"

    return execute


def available_cores() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def mps_running() -> bool | None:
    """Whether the CUDA MPS control daemon answers, or None off an NVIDIA machine.

    Without MPS, seed processes on one GPU take turns instead of running at once: on the
    40 GB A100, six processes ran no faster than one, and 1.9 times faster with MPS.
    """
    control = shutil.which("nvidia-cuda-mps-control")
    if shutil.which("nvidia-smi") is None or control is None:
        return None
    try:
        answer = subprocess.run(
            [control], input="get_server_list\n", capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    reply = (answer.stdout + answer.stderr).lower()
    return answer.returncode == 0 and "cannot" not in reply and "error" not in reply


MPS_WARNING = (
    "WARNING: CUDA MPS is not running, so parallel seed processes take turns on the GPU "
    "(six ran 1.9 times slower than with MPS). Start it with: nvidia-cuda-mps-control -d"
)


def job_environment(slots: int) -> dict[str, str]:
    """Share the CPU between parallel jobs unless the caller already chose a thread count."""
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", str(max(1, available_cores() // slots)))
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--slots", type=int, default=3, help="Jobs running at once.")
    parser.add_argument("--logs", type=Path, default=Path("logs/run_all"))
    args, extra = parser.parse_known_args(argv)
    if args.slots < 1:
        parser.error("--slots must be positive")
    jobs = build_jobs(args.stages, args.datasets, args.models, extra)
    if not jobs:
        parser.error("no jobs to run")
    logs = args.logs if args.logs.is_absolute() else PROJECT_ROOT / args.logs
    logs.mkdir(parents=True, exist_ok=True)
    environment = job_environment(args.slots)
    print(
        f"{len(jobs)} jobs in {args.slots} slots · OMP_NUM_THREADS={environment['OMP_NUM_THREADS']}"
        f" · logs in {logs}",
        flush=True,
    )
    if "--dry-run" not in extra and mps_running() is False:
        print(MPS_WARNING, flush=True)
    status = run_queue(jobs, args.slots, make_executor(logs, environment))
    failed = [name for name, state in status.items() if state in {"failed", "blocked"}]
    if failed:
        print(f"\nNot finished: {', '.join(f'{name} ({status[name]})' for name in failed)}.")
        print("Rerun to resume them.")
        return 1
    print("\nAll jobs finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
