"""The whole pipeline: download -> split -> prompt -> score -> select -> train -> report.

    uv run python -m uq_pet.main --config configs/nhr_gemma.yaml

There is exactly one persistent artifact, the LLM score cache, and it is resumable by
index — so a second run with a complete cache does no API work automatically. Nothing
should import from this module; it is the entry point, not a library.
"""

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import pandas as pd

from uq_pet import llm
from uq_pet.config import (
    RESULTS_DIR,
    ExperimentConfig,
    config_to_yaml,
    load_config,
)
from uq_pet.dataset import download_pet_ner, sentence_key, split_dataset, to_examples
from uq_pet.model_training import (
    configure_hf_logging,
    get_device,
    train_and_evaluate,
)
from uq_pet.plotting import plot_results
from uq_pet.prompt import build_system_prompt, build_user_prompts, prompt_fingerprint
from uq_pet.uncertainty import RANDOM, n_from_percent, score_arm, select, validate_arm

logger = logging.getLogger("uq_pet")

MIN_CACHE_ALIGNMENT = 0.80
SCORES = ["entity_f1", "entity_precision", "entity_recall", "token_accuracy"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uq_pet.main",
        description="Compare uncertainty-based vs. random training-data selection on PET.",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML run config")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="never call the API; fail if the cache does not cover the budget",
    )
    parser.add_argument("--limit", type=int, help="score only pool indices 0..N-1")
    parser.add_argument("--force-download", action="store_true", help="re-fetch the raw PET jsonl")
    parser.add_argument(
        "--dry-run", action="store_true", help="stop after selection and print both arms"
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
) -> Path:
    """Create results/<config stem>/ and snapshot the config before any work.

    An existing directory for this config is cleared, not merged into, so a shorter
    re-run cannot leave a previous run's figures and CSVs behind.
    """
    run_dir = results_dir / config_path.stem
    if run_dir.exists():
        print(f"Overwriting {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_to_yaml(cfg))
    return run_dir


def stage_score(
    cfg: ExperimentConfig,
    few_shot,
    pool_examples: list[dict],
    *,
    skip: bool,
    limit: int | None,
) -> tuple[list[dict], str, dict]:
    """Ensure the pool is scored, from cache where possible. Returns (records, sha, meta)."""
    system_prompt = build_system_prompt(few_shot)
    prompt_sha = prompt_fingerprint(system_prompt)
    user_prompts = build_user_prompts(pool_examples)
    keys = [sentence_key(ex) for ex in pool_examples]
    cache_path = cfg.llm.cache_path("pool", tag=cfg.few_shot_tag())
    load_kwargs = {
        "model": cfg.llm.model,
        "params": cfg.llm.sampling_params(),
        "prompt_sha": prompt_sha,
        "keys": keys,
    }

    if skip:
        cached = llm.load_cache(cache_path, **load_kwargs)
        records = sorted(cached.values(), key=lambda r: r["idx"])
        if not records:
            raise SystemExit(
                f"--skip-scoring: no usable records in {cache_path}. Drop the flag to score."
            )
        logger.info("--skip-scoring: using %d cached records", len(records))
    else:
        records = llm.score_split(
            cfg.llm,
            system_prompt,
            user_prompts,
            keys,
            cache_path,
            prompt_sha,
            split="pool",
            limit=limit if limit is not None else cfg.llm.limit,
        )

    failed = llm.failed_indices(records)
    alignment = llm.verify_cache_alignment(records, pool_examples)
    logger.info(
        "Scored pool: %d records, %d failed, %d truncated choices, alignment %.3f",
        len(records),
        len(failed),
        llm.truncated_count(records),
        alignment,
    )
    if alignment < MIN_CACHE_ALIGNMENT:
        raise SystemExit(
            f"Cache alignment {alignment:.3f} is below {MIN_CACHE_ALIGNMENT}: the cached "
            f"responses do not match the sentences at their indices. The split has probably "
            f"changed — delete {cache_path} and re-score."
        )

    meta = {
        "n_cached": len(records),
        "n_failed": len(failed),
        "n_truncated": llm.truncated_count(records),
        "cache_alignment": round(alignment, 4),
        "prompt_sha": prompt_sha,
    }
    return records, prompt_sha, meta


def stage_select(
    cfg: ExperimentConfig,
    records: list[dict],
    pool_examples: list[dict],
) -> dict[tuple[float, str], list[dict]]:
    """Select one set of sentences per (budget, arm).

    Metric scores don't depend on the budget, so each metric is evaluated once and
    reused across the sweep.
    """
    # The control draws from the whole pool, not just the sentences the LLM managed to
    # score, so it never inherits the cache's coverage: `random` needs an index and
    # nothing else, so it gets one bare record per pool sentence.
    pool_records = [{"idx": idx} for idx in range(len(pool_examples))]
    scores = {
        arm.resolved_label(): score_arm(
            arm.strategy,
            pool_records if arm.strategy == RANDOM else records,
            **arm.params,
        )
        for arm in cfg.arms
    }

    cells: dict[tuple[float, str], list[dict]] = {}
    for budget in cfg.budget_pct:
        n = n_from_percent(len(pool_examples), budget)
        logger.info("Budget %.3g%%: %d of %d pool sentences", budget, n, len(pool_examples))
        for arm in cfg.arms:
            label = arm.resolved_label()
            cells[budget, label] = select(scores[label], pool_examples, n, tie_seed=cfg.tie_seed)

        sizes = {label: len(cells[budget, label]) for label in cfg.arm_labels()}
        if len(set(sizes.values())) != 1:
            # Unequal budgets make the arms incomparable; assert would vanish under -O.
            raise ValueError(f"arms must have the same budget at {budget}%, got {sizes}")
    return cells


def stage_train(
    cfg: ExperimentConfig,
    cells: dict[tuple[float, str], list[dict]],
    test_examples: list[dict],
) -> pd.DataFrame:
    """Train one model per (budget, arm, seed) and score it on the held-out test split."""
    device = get_device()
    logger.info(
        "Training %d cell(s) = %d budget(s) x %d arm(s) x %d seed(s) on %s, "
        "evaluating on %d sentences",
        len(cells) * len(cfg.train_seeds),
        len(cfg.budget_pct),
        len(cfg.arms),
        len(cfg.train_seeds),
        device,
        len(test_examples),
    )

    rows = []
    for (budget, arm), examples in cells.items():
        for seed in cfg.train_seeds:
            metrics = train_and_evaluate(examples, test_examples, seed, cfg.train, device=device)
            rows.append(
                {
                    "budget_pct": budget,
                    "arm": arm,
                    "seed": seed,
                    "n_train": len(examples),
                    **metrics,
                }
            )
            logger.info(
                "  %5.3g%%  %-24s seed=%d  entity_F1=%.4f",
                budget,
                arm,
                seed,
                metrics["entity_f1"],
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Mean +/- std over seeds, the per-entity-type table, and the gaps vs. random.

    `gaps` is {budget: {arm: mean entity_f1 - random's mean entity_f1}}, empty when no
    `random` control arm was configured.
    """
    summary = results.groupby(["budget_pct", "arm"])[SCORES].agg(["mean", "std"]).round(4)

    entity_types = sorted({t for d in results["per_type_f1"] for t in d})
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
        if RANDOM not in at_budget.index:
            continue
        baseline = at_budget[RANDOM]
        gaps[float(budget)] = {
            arm: round(float(value - baseline), 4)
            for arm, value in at_budget.items()
            if arm != RANDOM
        }
    return summary, per_type, gaps


def _json_float(value) -> float | None:
    """None for NaN — a single-seed run has no std, and bare NaN is not valid JSON."""
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def write_outputs(
    run_dir: Path,
    cells: dict[tuple[float, str], list[dict]],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    per_type: pd.DataFrame,
    gaps: dict,
    meta: dict,
) -> None:
    selection: dict[str, dict[str, list[str]]] = {}
    for (budget, arm), examples in cells.items():
        selection.setdefault(f"{budget:g}", {})[arm] = [sentence_key(ex) for ex in examples]
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2))

    results.drop(columns="per_type_f1").to_csv(run_dir / "results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")
    per_type.to_csv(run_dir / "per_type_f1.csv")

    stats = results.groupby(["budget_pct", "arm"])["entity_f1"].agg(["mean", "std"]).round(4)
    sizes = results.groupby("budget_pct")["n_train"].first()
    by_budget = {}
    for budget in sorted(results["budget_pct"].unique()):
        by_budget[f"{budget:g}"] = {
            "n_train": int(sizes.loc[budget]),
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
                "budgets": [float(b) for b in sorted(results["budget_pct"].unique())],
                "arms": list(dict.fromkeys(results["arm"])),
                "train_seeds": sorted(results["seed"].unique().tolist()),
                "by_budget": by_budget,
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    # Fail on a bad metric name or parameter now, not after the pool has been scored.
    for arm in cfg.arms:
        try:
            validate_arm(arm.strategy, arm.params)
        except ValueError as e:
            raise SystemExit(f"{args.config}: {e}") from None

    run_dir = make_run_dir(cfg, args.config, args.results_dir)
    setup_logging(run_dir / "run.log", args.verbose)
    configure_hf_logging(quiet=not args.verbose)
    logger.info("Run directory: %s", run_dir)

    download_pet_ner(force=args.force_download)
    few_shot, pool, test = split_dataset(n_few_shot=cfg.n_few_shot, few_shot_seed=cfg.few_shot_seed)
    pool_examples = to_examples(pool)
    test_examples = to_examples(test)
    logger.info(
        "Split: %d few-shot (seed %d), %d pool, %d test",
        len(few_shot),
        cfg.few_shot_seed,
        len(pool_examples),
        len(test_examples),
    )

    # `random` scores the pool directly, so a run whose arms are all controls reads no
    # cache and needs no API key — scoring it anyway would spend the gateway on numbers
    # no arm looks at, or refuse to start under --skip-scoring for want of a cache.
    if any(arm.strategy != RANDOM for arm in cfg.arms):
        records, _sha, meta = stage_score(
            cfg, few_shot, pool_examples, skip=args.skip_scoring, limit=args.limit
        )
    else:
        logger.info("Every arm is the %s control: skipping the scoring stage", RANDOM)
        records, meta = [], {"scored": False}
    cells = stage_select(cfg, records, pool_examples)

    if args.dry_run:
        for (budget, arm), examples in cells.items():
            keys = [sentence_key(ex) for ex in examples]
            print(f"{budget:g}% {arm} ({len(examples)}): {keys}")
        return 0

    results = stage_train(cfg, cells, test_examples)
    summary, per_type, gaps = summarize(results)

    if not args.no_plot:
        plot_results(results, run_dir / "figures" / "arm_f1.png")
    write_outputs(run_dir, cells, results, summary, per_type, gaps, meta)

    print(summary.to_string())
    if gaps:
        print("\nentity F1 vs. the random control:")
        for budget in sorted(gaps):
            n_train = results.loc[results["budget_pct"] == budget, "n_train"].iloc[0]
            for arm, gap in gaps[budget].items():
                print(f"  {budget:5.3g}% (n={n_train:3d})  {arm:<28} {gap:+.4f}")
        stds = results.groupby(["budget_pct", "arm"])["entity_f1"].std()
        print(
            f"\n(seed std ~{stds.mean():.4f} over {len(cfg.train_seeds)} seeds — "
            f"a gap smaller than that is noise)"
        )
    print(f"\nWrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
