"""Run fixed experiment settings once per UQ metric through the notebook batch entry."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from uq_pet.experiment import ExperimentConfig
from uq_pet.token_model import UQ_METRICS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON file with fixed experiment settings.")
    source.add_argument("--config-json", help="Inline JSON with fixed experiment settings.")
    parser.add_argument(
        "--uq-metrics",
        nargs="+",
        choices=UQ_METRICS,
        default=list(UQ_METRICS),
        help="Metrics to run in order (default: all three); overrides config uq_metric.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print settings without training."
    )
    args = parser.parse_args(argv)
    try:
        payload = args.config.read_text() if args.config is not None else args.config_json
        settings = json.loads(payload)
        if not isinstance(settings, dict):
            raise ValueError("configuration must be a JSON object")
        if len(set(args.uq_metrics)) != len(args.uq_metrics):
            raise ValueError("uq-metrics must be unique")
        configs = [
            ExperimentConfig.model_validate({**settings, "uq_metric": metric})
            for metric in args.uq_metrics
        ]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(json.dumps([config.model_dump() for config in configs], indent=2))
        return 0

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] Running {config.uq_metric} vs random", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "notebooks" / "bert_token_uq.py"),
                "--config-json",
                config.model_dump_json(),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode:
            print(f"Stopped: {config.uq_metric} run failed.", file=sys.stderr)
            return completed.returncode if completed.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
