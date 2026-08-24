#!/usr/bin/env python3
"""Dry-run-first local launcher for balanced PET token-UQ sweeps."""

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uq_pet.experiment import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MODEL_SEEDS,
    ExperimentConfig,
    require_wandb_credentials,
)
from uq_pet.token_model import UQ_METRICS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "bert_token_uq.py"
SWEEPS_DIR = PROJECT_ROOT / "results" / "sweeps"

SEARCH_SPACE = {
    "uq_metric": UQ_METRICS,
    "k": (8, 16, 32),
    "bootstrap_epochs": (10, 20, 30),
    "update_passes": (1, 2, 4),
    "learning_rate": (2e-5, 3e-5, 5e-5),
    "batch_size": (4, 8, 16),
    "replay_ratio": (0, 0.5, 1.0, 2.0),
    "weight_decay": (0.0, 0.01),
}
CONSOLE = Console()


def sample_search_configs(count: int, rng: random.Random) -> list[dict]:
    """Sample unique acquisition and training configurations with a local seeded RNG."""
    if count < 1:
        raise ValueError("count must be positive")

    max_unique = 1
    for values in SEARCH_SPACE.values():
        max_unique *= len(values)
    target = min(count, max_unique)

    configs: list[dict] = []
    seen: set[tuple] = set()
    keys = tuple(SEARCH_SPACE)
    while len(configs) < target:
        config = {key: rng.choice(SEARCH_SPACE[key]) for key in keys}
        signature = tuple(config[key] for key in keys)
        if signature not in seen:
            seen.add(signature)
            configs.append(config)
    return configs


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def build_runs(
    search_configs: list[dict],
    checkpoints: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    wandb_enabled: bool,
    wandb_project: str,
    max_pool_percent: float,
) -> list[dict]:
    """Cross sampled configurations with every requested checkpoint and seed."""
    runs = []
    for config_idx, search_config in enumerate(search_configs, start=1):
        for checkpoint in checkpoints:
            for seed in seeds:
                config = ExperimentConfig(
                    checkpoint=checkpoint,
                    model_seeds=[seed],
                    max_pool_percent=max_pool_percent,
                    **search_config,
                    wandb_enabled=wandb_enabled,
                    wandb_project=wandb_project,
                )
                params = config.model_dump()
                digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
                run_id = f"cfg{config_idx:02d}-{_slug(checkpoint)}-seed{seed}-{digest}"
                params["wandb_run_name"] = run_id
                runs.append({"run_id": run_id, "params": params})
    return runs


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def params_to_cli_args(params: dict) -> list[str]:
    return ["--config-json", json.dumps(params, separators=(",", ":"), sort_keys=True)]


def printable_command(params: dict) -> str:
    command = ["uv", "run", str(NOTEBOOK_PATH.relative_to(PROJECT_ROOT))]
    command.extend(params_to_cli_args(params))
    return " ".join(shlex.quote(part) for part in command)


def make_sweep_summary(
    search_config_count: int,
    checkpoint_count: int,
    seed_count: int,
    total_runs: int,
    *,
    sampler_seed: int,
    wandb_enabled: bool,
    max_pool_percent: float,
) -> Panel:
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()
    summary.add_row("Sampled configs", str(search_config_count))
    summary.add_row("Checkpoints", str(checkpoint_count))
    summary.add_row("Model seeds", str(seed_count))
    summary.add_row("Total runs", f"[bold yellow]{total_runs}[/]")
    summary.add_row("Pool cap", f"{max_pool_percent:g}%")
    summary.add_row("Sampler seed", str(sampler_seed))
    summary.add_row(
        "W&B",
        "[bold green]enabled[/]" if wandb_enabled else "[dim]disabled[/]",
    )
    return Panel.fit(
        summary,
        title="[bold]PET token-UQ sweep[/]",
        border_style="bright_blue",
        padding=(1, 2),
    )


def make_run_table(runs: list[dict]) -> Table:
    table = Table(
        title="Planned Runs",
        title_style="bold italic",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold white",
        row_styles=("", "dim"),
    )
    table.add_column("#", style="yellow", justify="right", no_wrap=True)
    table.add_column("Run", style="bold cyan", no_wrap=True)
    table.add_column("Command", style="green", overflow="fold")
    for index, run in enumerate(runs, start=1):
        table.add_row(str(index), run["run_id"], printable_command(run["params"]))
    return table


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def find_run_output(run_root: Path) -> Path | None:
    outputs = sorted(path for path in run_root.glob("bert_token_uq_*") if path.is_dir())
    return outputs[-1] if outputs else None


def write_combined_results(sweep_dir: Path, completed_runs: list[dict]) -> Path | None:
    combined = []
    for run in completed_runs:
        output_dir = Path(run["output_dir"])
        with (output_dir / "results.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                combined.append(
                    {
                        "run_id": run["run_id"],
                        **{
                            f"config_{key}": format_value(value)
                            for key, value in run["params"].items()
                        },
                        **row,
                    }
                )
    if not combined:
        return None

    output_path = sweep_dir / "combined_results.csv"
    fieldnames = list(combined[0])
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    return output_path


def parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one value")
    return parsed


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not parsed or any(seed < 0 for seed in parsed):
        raise argparse.ArgumentTypeError("provide at least one non-negative seed")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of unique parameter configurations to sample (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Local RNG seed for training-configuration sampling (default: 0).",
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_csv_strings,
        default=DEFAULT_CHECKPOINTS,
        help="Comma-separated checkpoint IDs (default: all five configured models).",
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=DEFAULT_MODEL_SEEDS,
        help="Comma-separated model seeds (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B for every launched run; disabled by default.",
    )
    parser.add_argument(
        "--wandb-project",
        default="uq-pet-token-uq",
        help="W&B project used with --wandb.",
    )
    parser.add_argument(
        "--max-pool-percent",
        type=float,
        default=ExperimentConfig().max_pool_percent,
        help="Maximum scoreable-pool percentage acquired per run (default: 100).",
    )
    parser.add_argument(
        "--sweeps-dir",
        type=Path,
        default=SWEEPS_DIR,
        help="Directory that receives launched sweep folders.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run jobs sequentially. Without this flag, only commands are printed.",
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

    search_configs = sample_search_configs(args.count, random.Random(args.seed))
    runs = build_runs(
        search_configs,
        args.checkpoints,
        args.seeds,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        max_pool_percent=args.max_pool_percent,
    )
    CONSOLE.print(
        make_sweep_summary(
            len(search_configs),
            len(args.checkpoints),
            len(args.seeds),
            len(runs),
            sampler_seed=args.seed,
            wandb_enabled=args.wandb,
            max_pool_percent=args.max_pool_percent,
        )
    )
    CONSOLE.print()

    if not args.launch:
        CONSOLE.print(make_run_table(runs))
        CONSOLE.print()
        CONSOLE.print(
            "[bold yellow]Dry run complete.[/] "
            "Pass [bold green]--launch[/] to execute these jobs sequentially."
        )
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sweep_dir = args.sweeps_dir / f"bert_token_uq_{stamp}_gridseed{args.seed}"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "runs").mkdir()
    manifest = {
        "sampler_seed": args.seed,
        "search_configs": search_configs,
        "checkpoints": list(args.checkpoints),
        "model_seeds": list(args.seeds),
        "wandb_enabled": args.wandb,
        "wandb_project": args.wandb_project,
        "max_pool_percent": args.max_pool_percent,
        "runs": [],
    }
    manifest_path = sweep_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    completed_runs = []
    failures = 0
    for index, run in enumerate(runs, start=1):
        CONSOLE.rule(
            f"[bold cyan]Run {index}/{len(runs)}[/] [bold]{run['run_id']}[/]",
            style="bright_blue",
        )
        CONSOLE.print(f"[dim]$[/] [green]{printable_command(run['params'])}[/]")
        run_root = sweep_dir / "runs" / run["run_id"]
        run_root.mkdir()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["UQ_PET_RESULTS_DIR"] = str(run_root)
        command = [sys.executable, str(NOTEBOOK_PATH), *params_to_cli_args(run["params"])]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
        output_dir = find_run_output(run_root)
        success = completed.returncode == 0 and output_dir is not None
        record = {
            **run,
            "status": "completed" if success else "failed",
            "returncode": completed.returncode,
            "output_dir": str(output_dir) if output_dir is not None else None,
        }
        manifest["runs"].append(record)
        write_manifest(manifest_path, manifest)
        if success:
            completed_runs.append(record)
        else:
            failures += 1
        status = "[bold green]completed[/]" if success else "[bold red]failed[/]"
        CONSOLE.print(f"Status: {status}")
        CONSOLE.print()

    combined_path = write_combined_results(sweep_dir, completed_runs)
    final_summary = Table.grid(padding=(0, 2))
    final_summary.add_column(style="bold cyan", justify="right")
    final_summary.add_column()
    final_summary.add_row("Sweep output", str(sweep_dir))
    if combined_path is not None:
        final_summary.add_row("Combined evaluations", str(combined_path))
    final_summary.add_row("Completed", f"[bold green]{len(completed_runs)}[/]")
    final_summary.add_row("Failed", f"[bold red]{failures}[/]" if failures else "0")
    CONSOLE.print(
        Panel.fit(
            final_summary,
            title="[bold]Sweep complete[/]",
            border_style="green" if not failures else "red",
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
