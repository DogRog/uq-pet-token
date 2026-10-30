"""Compare the training-recipe pilot with the constant-rate tuning runs it pairs with.

Each pilot setting (a dataset and model with configs under configs/<dataset>/pilot/)
tunes the first 30 configurations of its random-baseline tuning sweep again, once with
the bootstrap learning rate split off and clipping (`split`), once also with averaged
models (`split_avg`). A follow-up tunes them with averaged models alone (`constant_avg`)
and checks the uncertainty-minus-random gap at its winner (scripts/pilot_gap.py). This
script checks that every pilot sweep pairs config for config with the constant-rate sweep,
prints the criteria fixed in configs/README.md for every recipe, and gives each verdict
once its runs are complete (ADR 0012). Nothing is trained.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from uq_pet.config import RandomBaselineSearchConfig
from uq_pet.paths import PROJECT_ROOT
from uq_pet.search import random_tuning_score, tuning_plan

RECIPES = ("constant", "split", "split_avg", "constant_avg")
FIRST_PILOT = ("constant", "split", "split_avg")
FOLLOW_UP = ("constant", "constant_avg")
# A run drops when its entity F1 falls this far below its best so far after round 0.
DROP = 0.10
# Round-0 entity F1 below this means the bootstrap failed.
FAILED_BOOTSTRAP = 0.05
# A recipe may lose at most this much winner validation F1 AUC to the one it extends.
AUC_TOLERANCE = 0.01
# Averaging may shrink the winner's validation uncertainty-minus-random F1 AUC by at most this.
GAP_TOLERANCE = 0.01
SETTINGS_NEEDED = 3
NEW_SETTINGS = {"bootstrap_learning_rate", "weight_averaging", "max_grad_norm"}


def run_stability(f1: list[float]) -> dict:
    """Round-0 F1, the largest fall below the best so far, and the mean round-to-round jitter.

    Jitter is each round's distance from the midpoint of its two neighbours, from round 1 on.
    """
    rounds = np.asarray(f1[1:], dtype=float)
    jitter = np.abs(rounds[1:-1] - (rounds[:-2] + rounds[2:]) / 2)
    return {
        "bootstrap_f1": float(f1[0]),
        "max_drop": float((np.maximum.accumulate(rounds) - rounds).max()) if len(rounds) else 0.0,
        "jitter": float(jitter.mean()) if len(jitter) else math.nan,
    }


def recipe_summary(trials: dict[str, list[dict]], objective: str) -> dict:
    """Stability over every seed's run of every configuration, and the tuning winner."""
    runs = [
        run_stability([row["entity_f1"] for row in sorted(seed_rows, key=lambda r: r["round"])])
        for rows in trials.values()
        for seed in sorted({row["seed"] for row in rows})
        for seed_rows in [[row for row in rows if row["seed"] == seed]]
    ]
    scores = {config_id: random_tuning_score(rows, objective) for config_id, rows in trials.items()}
    winner = min(scores, key=lambda config_id: (-scores[config_id], config_id))
    finals = [
        max((row for row in trials[winner] if row["seed"] == seed), key=lambda r: r["round"])[
            "entity_f1"
        ]
        for seed in sorted({row["seed"] for row in trials[winner]})
    ]
    return {
        "configs": len(trials),
        "drop_share": float(np.mean([run["max_drop"] >= DROP for run in runs])),
        "median_jitter": float(np.median([run["jitter"] for run in runs])),
        "failed_bootstrap_share": float(
            np.mean([run["bootstrap_f1"] < FAILED_BOOTSTRAP for run in runs])
        ),
        "winner": winner,
        "winner_auc": scores[winner],
        "winner_final_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
    }


def improves(candidate: dict, reference: dict) -> bool:
    """Fewer runs drop, and the winner keeps its validation F1 AUC within the tolerance."""
    return (
        candidate["drop_share"] < reference["drop_share"]
        and candidate["winner_auc"] >= reference["winner_auc"] - AUC_TOLERANCE
    )


def verdict(summaries: dict[str, dict[str, dict]]) -> dict:
    """Adopt the split if it improves on the constant rate in enough settings, then averaging."""
    split = [setting for setting, s in summaries.items() if improves(s["split"], s["constant"])]
    averaged = [setting for setting, s in summaries.items() if improves(s["split_avg"], s["split"])]
    adopt_split = len(split) >= SETTINGS_NEEDED
    return {
        "split_improves": split,
        "averaging_improves": averaged,
        "adopt_split": adopt_split,
        "adopt_averaging": adopt_split and len(averaged) >= SETTINGS_NEEDED,
    }


def _sweep(config_path: Path) -> tuple[RandomBaselineSearchConfig, Path]:
    config = RandomBaselineSearchConfig.model_validate_json(config_path.read_text())
    sweeps = (
        config.sweeps_dir if config.sweeps_dir.is_absolute() else PROJECT_ROOT / config.sweeps_dir
    )
    return config, sweeps / config.sweep_name


def _completed_trials(root: Path, config_ids: list[str]) -> dict[str, list[dict]]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return {}
    return {
        trial["config_id"]: pl.read_csv(root / trial["run_dir"] / "results.csv").to_dicts()
        for trial in json.loads(summary_path.read_text())["trials"]
        if trial["status"] == "complete" and trial["config_id"] in config_ids
    }


def _check_pairing(base: tuple, pilot: tuple) -> None:
    """Refuse a pilot sweep that does not retune the baseline's first configurations."""
    (base_config, base_root), (pilot_config, pilot_root) = base, pilot
    base_plan, pilot_plan = tuning_plan(base_config), tuning_plan(pilot_config)
    saved = base_root / "plan.json"
    if (
        saved.is_file()
        and json.loads(saved.read_text())["configurations"] != base_plan["configurations"]
    ):
        raise ValueError(f"{saved} no longer matches its config")
    unpaired = [
        pilot_entry["config_id"]
        for pilot_entry, base_entry in zip(
            pilot_plan["configurations"], base_plan["configurations"], strict=False
        )
        if {
            key: value
            for key, value in pilot_entry["parameters"].items()
            if key not in NEW_SETTINGS
        }
        != base_entry["parameters"]
    ]
    pilot_fixed = {
        key: value for key, value in pilot_plan["fixed_config"].items() if key not in NEW_SETTINGS
    }
    if unpaired or pilot_fixed != base_plan["fixed_config"]:
        raise ValueError(f"{pilot_config.sweep_name} does not pair with {base_config.sweep_name}")
    for name in ("validation_sentences", "validation_seed"):
        if pilot_plan[name] != base_plan[name]:
            raise ValueError(f"{pilot_config.sweep_name} uses another {name}")
    pilot_split, base_split = pilot_root / "split.json", base_root / "split.json"
    if pilot_split.is_file() and pilot_split.read_text() != base_split.read_text():
        raise ValueError(f"{pilot_config.sweep_name} holds out other validation sentences")


def follow_up_verdict(
    summaries: dict[str, dict[str, dict]], gaps: dict[str, dict[str, float]]
) -> dict:
    """Adopt averaged models at the constant rate if they improve and keep the gap."""
    improved = [
        setting for setting, s in summaries.items() if improves(s["constant_avg"], s["constant"])
    ]
    kept = [
        setting
        for setting, gap in gaps.items()
        if gap["averaged"] >= gap["constant"] - GAP_TOLERANCE
    ]
    return {
        "averaging_improves": improved,
        "gap_kept": kept,
        "adopt": len(improved) >= SETTINGS_NEEDED and len(kept) >= SETTINGS_NEEDED,
    }


def read_gaps(configs_dir: Path) -> dict[str, dict[str, float]]:
    """The finished gap checks, by setting, from each follow-up sweep."""
    gaps = {}
    for path in sorted(configs_dir.glob("*/pilot/*_constant_avg.json")):
        _, root = _sweep(path)
        files = {name: root / "gap_check" / f"{name}.json" for name in ("constant", "averaged")}
        if all(file.is_file() for file in files.values()):
            setting = f"{path.parent.parent.name}/{path.name.removesuffix('_constant_avg.json')}"
            gaps[setting] = {
                name: json.loads(file.read_text())["gap_auc"] for name, file in files.items()
            }
    return gaps


def collect(configs_dir: Path) -> tuple[pl.DataFrame, dict[str, dict[str, dict]], set[str]]:
    """Summarize every recipe of every pilot setting; also name the recipes not yet finished."""
    rows, summaries, unfinished = [], {}, set()
    for split_path in sorted(configs_dir.glob("*/pilot/*_split.json")):
        dataset, model = split_path.parent.parent.name, split_path.name.removesuffix("_split.json")
        sweeps = {
            "constant": _sweep(configs_dir / dataset / "tune_random" / f"{model}.json"),
            "split": _sweep(split_path),
            "split_avg": _sweep(split_path.with_name(f"{model}_split_avg.json")),
            "constant_avg": _sweep(split_path.with_name(f"{model}_constant_avg.json")),
        }
        for recipe in RECIPES[1:]:
            _check_pairing(sweeps["constant"], sweeps[recipe])
        planned = [
            entry["config_id"] for entry in tuning_plan(sweeps["split"][0])["configurations"]
        ]
        setting = f"{dataset}/{model}"
        summaries[setting] = {}
        for recipe in RECIPES:
            config, root = sweeps[recipe]
            trials = _completed_trials(root, planned)
            if len(trials) != len(planned):
                unfinished.add(recipe)
            row = {"setting": setting, "recipe": recipe, "configs": len(trials)}
            if trials:
                summary = recipe_summary(trials, config.objective)
                summaries[setting][recipe] = summary
                row |= summary
            rows.append(row)
    return pl.DataFrame(rows), summaries, unfinished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=PROJECT_ROOT / "configs")
    args = parser.parse_args(argv)
    try:
        table, summaries, unfinished = collect(args.configs_dir)
        gaps = read_gaps(args.configs_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(table)
    if len(summaries) != SETTINGS_NEEDED + 1:
        print(f"\nThe criteria cover 4 settings; found {len(summaries)}, so no verdict.")
        return 0
    if unfinished & set(FIRST_PILOT):
        print("\nNo first-pilot verdict until every configuration is complete.")
    else:
        result = verdict(summaries)
        print(
            "\nSplit improves on the constant rate in: "
            f"{', '.join(result['split_improves']) or 'none'}"
        )
        print(
            "Averaging improves on the split in: "
            f"{', '.join(result['averaging_improves']) or 'none'}"
        )
        if not result["adopt_split"]:
            print("First-pilot verdict: keep the constant learning rate.")
        elif result["adopt_averaging"]:
            print("First-pilot verdict: adopt the split bootstrap, clipping, and averaged models.")
        else:
            print("First-pilot verdict: adopt the split bootstrap and clipping; no averaging.")
    if unfinished & set(FOLLOW_UP) or len(gaps) != SETTINGS_NEEDED + 1:
        print("\nNo follow-up verdict until every sweep and gap check is complete.")
        return 0
    result = follow_up_verdict(summaries, gaps)
    print("\nWinner gap AUC, without and with averaging:")
    for setting, gap in gaps.items():
        print(f"  {setting}: {gap['constant']:+.4f} -> {gap['averaged']:+.4f}")
    print(
        "Averaging improves on the constant rate in: "
        f"{', '.join(result['averaging_improves']) or 'none'}"
    )
    print(f"Averaging keeps the gap in: {', '.join(result['gap_kept']) or 'none'}")
    if result["adopt"]:
        print("Follow-up verdict: rerun the paper's experiments with averaged models.")
    else:
        print("Follow-up verdict: keep the constant learning rate without averaging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
