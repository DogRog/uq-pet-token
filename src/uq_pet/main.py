"""Seed train -> BERT UQ -> select -> continue train -> evaluate."""

import argparse
import gc
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from uq_pet.bert_uq import score_pool
from uq_pet.config import RESULTS_DIR, ExperimentConfig, config_to_yaml, load_config
from uq_pet.dataset import download_pet_ner, sentence_key, split_dataset, to_examples
from uq_pet.model_training import (
    configure_hf_logging,
    copy_model_state,
    get_device,
    train_and_evaluate,
    train_token_classifier,
)
from uq_pet.plotting import plot_results
from uq_pet.uncertainty import RANDOM, n_from_percent, score_arm, select, validate_arm

logger = logging.getLogger("uq_pet")
SCORES = ["entity_f1", "entity_precision", "entity_recall", "token_accuracy"]
CellKey = tuple[float, str, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uq_pet.main",
        description="Compare BERT uncertainty vs. random PET training-data selection.",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML run config")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--run-name",
        help="name the run directory instead of using the config stem",
    )
    parser.add_argument("--force-download", action="store_true", help="re-fetch PET jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="train the seed model, score/select the pool, then stop before continuation",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO logging to console")
    return parser


def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(console)
    return logger


def make_run_dir(
    cfg: ExperimentConfig,
    config_path: Path,
    results_dir: Path = RESULTS_DIR,
    run_name: str | None = None,
) -> Path:
    """Create a unique run directory and snapshot the resolved configuration."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / f"{run_name or config_path.stem}_{stamp}"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_to_yaml(cfg))
    return run_dir


def stage_select(
    cfg: ExperimentConfig,
    records: list[dict],
    pool_examples: list[dict],
    model_seed: int,
) -> dict[CellKey, list[dict]]:
    """Select one nested top-n set per budget and arm for one fitted model seed."""
    pool_records = [
        {"idx": idx, "n_tokens": len(example["tokens"])}
        for idx, example in enumerate(pool_examples)
    ]
    scores = {}
    for arm in cfg.arms:
        params = dict(arm.params)
        if arm.strategy == RANDOM:
            # Unless explicitly overridden, random selection varies with the same seed
            # that varies the fitted UQ model and its continuation training.
            params.setdefault("seed", model_seed)
        scores[arm.resolved_label()] = score_arm(
            arm.strategy,
            pool_records if arm.strategy == RANDOM else records,
            **params,
        )

    cells: dict[CellKey, list[dict]] = {}
    for budget in cfg.budget_pct:
        n = n_from_percent(len(pool_examples), budget)
        for arm in cfg.arms:
            label = arm.resolved_label()
            cells[budget, label, model_seed] = select(
                scores[label],
                pool_examples,
                n,
                tie_seed=cfg.tie_seed + model_seed,
            )
        sizes = {len(cells[budget, label, model_seed]) for label in cfg.arm_labels()}
        if len(sizes) != 1:
            raise ValueError(f"arms must have the same budget at {budget}%, got {sizes}")
    return cells


def stage_train(
    cfg: ExperimentConfig,
    cells: dict[CellKey, list[dict]],
    seed_examples: list[dict],
    test_examples: list[dict],
    initial_state: dict[str, torch.Tensor],
    model_seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Continue independent clones of one seed-trained state and evaluate them."""
    rows = []
    for (budget, arm, seed), selected in cells.items():
        if seed != model_seed:
            continue
        train_examples = [*seed_examples, *selected]
        metrics = train_and_evaluate(
            train_examples,
            test_examples,
            seed,
            cfg.train,
            device=device,
            initial_state=initial_state,
        )
        rows.append(
            {
                "budget_pct": budget,
                "arm": arm,
                "seed": seed,
                "n_seed": len(seed_examples),
                "n_selected": len(selected),
                "n_train": len(train_examples),
                **metrics,
            }
        )
        logger.info(
            "%5.3g%%  %-24s seed=%d  selected=%d  entity_F1=%.4f",
            budget,
            arm,
            seed,
            len(selected),
            metrics["entity_f1"],
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate metrics over model seeds and calculate paired gaps vs. random."""
    summary = results.groupby(["budget_pct", "arm"])[SCORES].agg(["mean", "std"]).round(4)
    entity_types = sorted({tag for values in results["per_type_f1"] for tag in values})
    per_type = (
        results.join(results["per_type_f1"].apply(pd.Series))
        .groupby(["budget_pct", "arm"])[entity_types]
        .mean()
        .round(3)
        .T.rename_axis("entity type")
    )
    means = results.groupby(["budget_pct", "arm"])["entity_f1"].mean()
    gaps: dict[float, dict[str, float]] = {}
    for budget in results["budget_pct"].unique():
        at_budget = means.loc[budget]
        if RANDOM in at_budget.index:
            gaps[float(budget)] = {
                arm: round(float(value - at_budget[RANDOM]), 4)
                for arm, value in at_budget.items()
                if arm != RANDOM
            }
    return summary, per_type, gaps


def _json_float(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def write_outputs(
    run_dir: Path,
    cells: dict[CellKey, list[dict]],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    per_type: pd.DataFrame,
    gaps: dict,
    meta: dict,
) -> None:
    """Write selections, flat results, aggregates, and machine-readable metadata."""
    selection: dict[str, dict[str, dict[str, list[str]]]] = {}
    for (budget, arm, seed), examples in cells.items():
        selection.setdefault(str(seed), {}).setdefault(f"{budget:g}", {})[arm] = [
            sentence_key(example) for example in examples
        ]
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    results.drop(columns="per_type_f1").to_csv(run_dir / "results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")
    per_type.to_csv(run_dir / "per_type_f1.csv")

    stats = results.groupby(["budget_pct", "arm"])["entity_f1"].agg(["mean", "std"])
    by_budget = {}
    for budget in sorted(results["budget_pct"].unique()):
        first = results.loc[results["budget_pct"] == budget].iloc[0]
        by_budget[f"{budget:g}"] = {
            "n_seed": int(first["n_seed"]),
            "n_selected": int(first["n_selected"]),
            "n_train": int(first["n_train"]),
            "entity_f1": {
                arm: {"mean": _json_float(row["mean"]), "std": _json_float(row["std"])}
                for arm, row in stats.loc[budget].iterrows()
            },
            "gap_vs_random": gaps.get(float(budget), {}),
        }
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                **meta,
                "budgets": [float(value) for value in sorted(results["budget_pct"].unique())],
                "arms": list(dict.fromkeys(results["arm"])),
                "model_seeds": sorted(results["seed"].unique().tolist()),
                "by_budget": by_budget,
            },
            indent=2,
        )
    )


def release_device_cache() -> None:
    """Collect released models and return cached accelerator blocks."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    for arm in cfg.arms:
        try:
            validate_arm(arm.strategy, arm.params)
        except ValueError as error:
            raise SystemExit(f"{args.config}: {error}") from None

    run_dir = make_run_dir(cfg, args.config, args.results_dir, run_name=args.run_name)
    setup_logging(run_dir / "run.log", args.verbose)
    configure_hf_logging(quiet=not args.verbose)
    download_pet_ner(force=args.force_download)
    seed_set, pool, test = split_dataset(n_seed=cfg.n_seed, seed_split_seed=cfg.seed_split_seed)
    seed_examples, pool_examples, test_examples = map(to_examples, (seed_set, pool, test))
    device = get_device()
    logger.info(
        "Split: %d labelled seed, %d pool, %d test; checkpoint=%s device=%s",
        len(seed_examples),
        len(pool_examples),
        len(test_examples),
        cfg.train.checkpoint,
        device,
    )

    all_cells: dict[CellKey, list[dict]] = {}
    result_frames = []
    per_seed_meta = {}
    for model_seed in cfg.model_seeds:
        logger.info(
            "Seed %d: fitting the shared scorer for %d epoch(s)",
            model_seed,
            cfg.uq.bootstrap_epochs,
        )
        model, tokenizer, bootstrap_info = train_token_classifier(
            seed_examples,
            model_seed,
            cfg.train,
            device=device,
            epochs=cfg.uq.bootstrap_epochs,
            early_stopping=False,
        )
        initial_state = copy_model_state(model)
        records = score_pool(
            model,
            tokenizer,
            pool_examples,
            max_length=cfg.train.max_length,
            batch_size=cfg.uq.score_batch_size,
            device=device,
        )
        cells = stage_select(cfg, records, pool_examples, model_seed)
        all_cells.update(cells)
        per_seed_meta[str(model_seed)] = {
            "bootstrap": bootstrap_info,
            "n_scored": len(records),
            "n_truncated": sum(record["truncated"] for record in records),
        }
        # Delete the caller's references before collecting; passing them to a helper
        # would leave these references alive throughout continuation training.
        del model, tokenizer
        release_device_cache()

        if not args.dry_run:
            result_frames.append(
                stage_train(
                    cfg,
                    cells,
                    seed_examples,
                    test_examples,
                    initial_state,
                    model_seed,
                    device,
                )
            )
        del initial_state

    if args.dry_run:
        for (budget, arm, seed), examples in all_cells.items():
            keys = [sentence_key(example) for example in examples]
            print(f"seed={seed} {budget:g}% {arm} ({len(examples)}): {keys}")
        print(f"\nWrote bootstrap log and resolved config to {run_dir}")
        return 0

    results = pd.concat(result_frames, ignore_index=True)
    summary, per_type, gaps = summarize(results)
    if not args.no_plot:
        plot_results(results, run_dir / "figures" / "arm_f1.png")
    meta = {
        "scorer": {
            "checkpoint": cfg.train.checkpoint,
            "bootstrap_epochs": cfg.uq.bootstrap_epochs,
            "per_seed": per_seed_meta,
        }
    }
    write_outputs(run_dir, all_cells, results, summary, per_type, gaps, meta)
    print(summary.to_string())
    if gaps:
        print("\nentity F1 vs. random:")
        for budget in sorted(gaps):
            n_selected = results.loc[results["budget_pct"] == budget, "n_selected"].iloc[0]
            for arm, gap in gaps[budget].items():
                print(f"  {budget:5.3g}% (selected={n_selected:3d})  {arm:<24} {gap:+.4f}")
    print(f"\nWrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
