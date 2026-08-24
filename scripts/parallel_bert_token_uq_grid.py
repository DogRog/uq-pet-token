#!/usr/bin/env python3
"""Run the PET token-UQ grid with two concurrent workers on one CUDA GPU."""

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from bert_token_uq_grid import (
    MODEL_SEEDS,
    PROJECT_ROOT,
    parse_csv_ints,
    require_wandb_credentials,
)
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

GRID_SCRIPT = PROJECT_ROOT / "scripts" / "bert_token_uq_grid.py"
LAUNCHES_DIR = PROJECT_ROOT / "results" / "parallel_launches"
WORKER_GROUPS = (
    (
        "worker-a",
        (
            "distilbert-base-cased",
            "bert-base-cased",
            "roberta-base",
        ),
    ),
    (
        "worker-b",
        (
            "microsoft/deberta-v3-base",
            "answerdotai/ModernBERT-base",
        ),
    ),
)
CONSOLE = Console()


def printable_command(command: list[str]) -> str:
    printable = ["uv", "run", str(GRID_SCRIPT.relative_to(PROJECT_ROOT)), *command[2:]]
    return " ".join(shlex.quote(part) for part in printable)


def build_worker_specs(args, launch_dir: Path) -> list[dict]:
    specs = []
    sweeps_dir = launch_dir / "sweeps"
    for worker_name, checkpoints in WORKER_GROUPS:
        command = [
            sys.executable,
            str(GRID_SCRIPT),
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
            "--checkpoints",
            ",".join(checkpoints),
            "--seeds",
            ",".join(str(seed) for seed in args.seeds),
            "--wandb-project",
            args.wandb_project,
            "--max-pool-percent",
            str(args.max_pool_percent),
            "--sweeps-dir",
            str(sweeps_dir),
            "--launch",
        ]
        if args.wandb:
            command.append("--wandb")
        specs.append(
            {
                "name": worker_name,
                "checkpoints": list(checkpoints),
                "command": command,
                "printable_command": printable_command(command),
                "log_path": str(launch_dir / "logs" / f"{worker_name}.log"),
                "status": "planned",
                "returncode": None,
            }
        )
    return specs


def print_plan(args, launch_dir: Path, specs: list[dict]) -> None:
    seed_count = len(args.seeds)
    total_runs = args.count * sum(len(spec["checkpoints"]) for spec in specs) * seed_count
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()
    summary.add_row("Workers", str(len(specs)))
    summary.add_row("CUDA device", args.cuda_device)
    summary.add_row("Sampled configs", str(args.count))
    summary.add_row("Model seeds", str(seed_count))
    summary.add_row("Total runs", f"[bold yellow]{total_runs}[/]")
    summary.add_row("Pool cap", f"{args.max_pool_percent:g}%")
    summary.add_row("Sampler seed", str(args.seed))
    summary.add_row("W&B", "[bold green]enabled[/]" if args.wandb else "[dim]disabled[/]")
    summary.add_row("Launch directory", str(launch_dir))
    CONSOLE.print(
        Panel.fit(
            summary,
            title="[bold]Parallel PET token-UQ grid[/]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    workers = Table(title="Worker Plan", title_style="bold italic", border_style="bright_blue")
    workers.add_column("Worker", style="bold cyan", no_wrap=True)
    workers.add_column("Checkpoints")
    workers.add_column("Runs", style="yellow", justify="right")
    workers.add_column("Command", style="green", overflow="fold")
    for spec in specs:
        worker_runs = args.count * len(spec["checkpoints"]) * seed_count
        workers.add_row(
            spec["name"],
            "\n".join(spec["checkpoints"]),
            str(worker_runs),
            spec["printable_command"],
        )
    CONSOLE.print()
    CONSOLE.print(workers)


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def combine_worker_results(launch_dir: Path) -> Path | None:
    combined = []
    for csv_path in sorted((launch_dir / "sweeps").glob("**/combined_results.csv")):
        with csv_path.open(newline="") as file:
            combined.extend(csv.DictReader(file))
    if not combined:
        return None

    output_path = launch_dir / "combined_results.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return output_path


def launch_workers(args, launch_dir: Path, specs: list[dict], manifest_path: Path) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    environment["PYTHONUNBUFFERED"] = "1"

    processes = []
    for spec in specs:
        log_path = Path(spec["log_path"])
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                spec["command"],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        spec["status"] = "running"
        spec["pid"] = process.pid
        processes.append((spec, process))

    write_manifest(manifest_path, {"settings": vars(args), "workers": specs})
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=CONSOLE,
        ) as progress:
            tasks = {
                spec["name"]: progress.add_task(
                    f"[cyan]{spec['name']}[/] running · log: {spec['log_path']}", total=None
                )
                for spec, _process in processes
            }
            pending = set(spec["name"] for spec, _process in processes)
            while pending:
                for spec, process in processes:
                    if spec["name"] not in pending:
                        continue
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    spec["returncode"] = returncode
                    spec["status"] = "completed" if returncode == 0 else "failed"
                    style = "green" if returncode == 0 else "red"
                    progress.update(
                        tasks[spec["name"]],
                        description=f"[{style}]{spec['name']} {spec['status']}[/]",
                    )
                    progress.stop_task(tasks[spec["name"]])
                    pending.remove(spec["name"])
                    write_manifest(manifest_path, {"settings": vars(args), "workers": specs})
                if pending:
                    time.sleep(2)
    except KeyboardInterrupt:
        CONSOLE.print("\n[bold yellow]Stopping both workers…[/]")
        for spec, process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                spec["status"] = "interrupted"
        for _spec, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write_manifest(manifest_path, {"settings": vars(args), "workers": specs})
        return 130

    return 1 if any(spec["returncode"] != 0 for spec in specs) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4, help="Sampled configurations per worker.")
    parser.add_argument("--seed", type=int, default=0, help="Shared configuration sampler seed.")
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=MODEL_SEEDS,
        help="Comma-separated model seeds.",
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device exposed to both workers (default: 0).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for every run.")
    parser.add_argument("--wandb-project", default="uq-pet-token-uq")
    parser.add_argument(
        "--max-pool-percent",
        type=float,
        default=100.0,
        help="Maximum scoreable-pool percentage acquired per run (default: 100).",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start both workers. Without this flag, only show the plan.",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")
    if not 0 < args.max_pool_percent <= 100:
        parser.error("--max-pool-percent must be in (0, 100]")
    if args.launch and args.wandb:
        load_dotenv(PROJECT_ROOT / ".env")
        try:
            require_wandb_credentials(os.environ)
        except ValueError as error:
            CONSOLE.print(
                Panel.fit(
                    str(error),
                    title="[bold]Missing W&B credentials[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            raise SystemExit(2) from None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    launch_dir = LAUNCHES_DIR / f"bert_token_uq_{stamp}_gridseed{args.seed}"
    specs = build_worker_specs(args, launch_dir)
    print_plan(args, launch_dir, specs)
    if not args.launch:
        CONSOLE.print()
        CONSOLE.print(
            "[bold yellow]Dry run complete.[/] Pass [bold green]--launch[/] to start both workers."
        )
        return

    (launch_dir / "logs").mkdir(parents=True)
    (launch_dir / "sweeps").mkdir()
    manifest_path = launch_dir / "parallel_manifest.json"
    write_manifest(manifest_path, {"settings": vars(args), "workers": specs})
    returncode = launch_workers(args, launch_dir, specs, manifest_path)
    combined_path = combine_worker_results(launch_dir)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()
    summary.add_row("Launch directory", str(launch_dir))
    summary.add_row("Manifest", str(manifest_path))
    summary.add_row("Worker A log", specs[0]["log_path"])
    summary.add_row("Worker B log", specs[1]["log_path"])
    if combined_path is not None:
        summary.add_row("Combined results", str(combined_path))
    summary.add_row(
        "Status",
        "[bold green]completed[/]" if returncode == 0 else "[bold red]failed[/]",
    )
    CONSOLE.print(
        Panel.fit(
            summary,
            title="[bold]Parallel grid finished[/]",
            border_style="green" if returncode == 0 else "red",
        )
    )
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
