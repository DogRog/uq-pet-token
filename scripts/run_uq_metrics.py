"""Compare one fixed configuration using shared training and organized, resumable results."""

import argparse
import json
from pathlib import Path

import bert_token_uq_search as search
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
        help="Metrics to compare (default: all three); overrides config uq_metric.",
    )
    parser.add_argument("--seed-workers", type=int, help="Concurrent seeds (default: config or 2).")
    parser.add_argument("--sweep-name", help="Result group name (default: checkpoint-best-uq).")
    parser.add_argument("--sweeps-dir", type=Path, default=PROJECT_ROOT / "results" / "best_uq")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training.")
    args = parser.parse_args(argv)
    try:
        payload = args.config.read_text() if args.config is not None else args.config_json
        settings = json.loads(payload)
        if not isinstance(settings, dict):
            raise ValueError("configuration must be a JSON object")
        settings["seed_workers"] = (
            args.seed_workers if args.seed_workers is not None else settings.get("seed_workers", 2)
        )
        config = ExperimentConfig.model_validate(settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    name = config.checkpoint.rsplit("/", maxsplit=1)[-1] + "-best-uq"
    search_args = argparse.Namespace(
        config=None,
        config_json=json.dumps(
            {
                **config.model_dump(),
                "mode": "compare-fixed",
                "num_configs": 1,
                "uq_metrics": args.uq_metrics,
                "sweep_name": args.sweep_name or name,
                "sweeps_dir": str(args.sweeps_dir),
            }
        ),
        dry_run=args.dry_run,
        publish_wandb_only=False,
    )
    search.run(search_args, parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
