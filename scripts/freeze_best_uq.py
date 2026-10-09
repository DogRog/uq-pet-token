"""Freeze a completed random-baseline tuning winner into configs/best_uq/.

Prints the frozen config path, or nothing when the tuning sweep has no winner yet. The
tuning output is the source of truth, so a changed winner overwrites the saved config;
an unchanged one leaves the file untouched, also on dry runs.
"""

import argparse
import json
import sys
from pathlib import Path

from uq_pet.config import ExperimentConfig, RandomBaselineSearchConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def freeze(tune_config: Path, out_dir: Path) -> Path | None:
    """Write the validation winner with W&B and concurrency settings for best-UQ runs."""
    tuning = RandomBaselineSearchConfig.model_validate_json(tune_config.read_text())
    if tuning.tuning_split != "validation":
        raise ValueError("only validation winners become best-UQ configs (ADR 0007)")
    sweep = (
        tuning.sweeps_dir if tuning.sweeps_dir.is_absolute() else PROJECT_ROOT / tuning.sweeps_dir
    )
    winner_path = sweep / tuning.sweep_name / "best_config.json"
    if not winner_path.is_file():
        return None
    winner = json.loads(winner_path.read_text())
    frozen = {
        **winner,
        "wandb_enabled": tuning.wandb_enabled,
        "wandb_project": tuning.wandb_project.replace("-random-baseline", "-best-uq"),
        "seed_workers": 5,
    }
    ExperimentConfig.model_validate(frozen)
    path = out_dir / (tune_config.stem.removesuffix(f"_tune_random_{tuning.num_configs}") + ".json")
    if not path.is_file() or json.loads(path.read_text()) != frozen:
        path.write_text(json.dumps(frozen, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tune_config", type=Path, help="configs/tune_random/*.json file.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "configs" / "best_uq")
    args = parser.parse_args(argv)
    try:
        path = freeze(args.tune_config, args.out_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if path is None:
        print(f"No frozen winner yet for {args.tune_config}; finish its tuning.", file=sys.stderr)
    else:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
