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
from datetime import datetime
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
from uq_pet.prompt import build_system_prompt, build_user_prompts, prompt_fingerprint
from uq_pet.uncertainty import n_from_percent, score_records, select

logger = logging.getLogger("uq_pet")

MIN_CACHE_ALIGNMENT = 0.80
SCORES = ["entity_f1", "entity_precision", "entity_recall", "token_accuracy"]
ARM_COLOR = {"uncertainty": "#4269d0", "random": "#e08a2e"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uq_pet.main",
        description="Compare uncertainty-based vs. random training-data selection on PET.",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML run config")
    parser.add_argument("--run-name", help="run directory stem (default: the config's stem)")
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
    run_name: str | None = None,
) -> Path:
    """Create results/<name>_<timestamp>/ and snapshot the config before any work."""
    stem = run_name or config_path.stem
    run_dir = results_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
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
    cache_path = cfg.llm.cache_path("pool")
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
    cfg: ExperimentConfig, records: list[dict], pool_examples: list[dict]
) -> dict[str, list[dict]]:
    """Pick the same number of sentences for every arm."""
    n = n_from_percent(len(pool_examples), cfg.budget_pct)
    scores = score_records(records, digits_only=(cfg.score == "filtered"))
    logger.info("Budget: %d of %d pool sentences (%.3g%%)", n, len(pool_examples), cfg.budget_pct)

    arms = {
        name: select(
            name,
            pool_examples,
            n,
            scores=scores,
            seed=cfg.selection_seed,
            tie_seed=cfg.tie_seed,
        )
        for name in cfg.arms
    }

    sizes = {name: len(examples) for name, examples in arms.items()}
    if len(set(sizes.values())) != 1:
        # An unequal budget would make the arms incomparable; assert would vanish under -O.
        raise ValueError(f"arms must have the same budget, got {sizes}")
    return arms


def stage_train(
    cfg: ExperimentConfig, arms: dict[str, list[dict]], test_examples: list[dict]
) -> pd.DataFrame:
    """Train one model per (arm, seed) and score it on the held-out test split."""
    device = get_device()
    logger.info(
        "Training %d arm(s) x %d seed(s) on %s, evaluating on %d sentences",
        len(arms),
        len(cfg.train_seeds),
        device,
        len(test_examples),
    )

    rows = []
    for arm, examples in arms.items():
        for seed in cfg.train_seeds:
            metrics = train_and_evaluate(examples, test_examples, seed, cfg.train, device=device)
            rows.append({"arm": arm, "seed": seed, "n_train": len(examples), **metrics})
            logger.info("  %-12s seed=%d  entity_F1=%.4f", arm, seed, metrics["entity_f1"])
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Mean +/- std over seeds, the per-entity-type table, and the uncertainty gap."""
    summary = results.groupby("arm")[SCORES].agg(["mean", "std"]).round(4)

    entity_types = sorted({t for d in results["per_type_f1"] for t in d})
    per_type = (
        results.join(results["per_type_f1"].apply(pd.Series))
        .groupby("arm")[entity_types]
        .mean()
        .round(3)
        .T.rename_axis("entity type")
    )

    means = results.groupby("arm")["entity_f1"].mean()
    gap = float(means.get("uncertainty", float("nan")) - means.get("random", float("nan")))
    if {"uncertainty", "random"} <= set(per_type.columns):
        per_type["delta"] = (per_type["uncertainty"] - per_type["random"]).round(3)
    return summary, per_type, gap


def plot_arms(results: pd.DataFrame, sizes: dict[str, int], gap: float, out_path: Path) -> None:
    """Test-set entity F1 per arm: bar = mean over seeds, dots = individual seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(sizes)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for x, arm in enumerate(arms):
        seed_scores = results.loc[results["arm"] == arm, "entity_f1"]
        ax.bar(x, seed_scores.mean(), width=0.55, color=ARM_COLOR.get(arm, "#888"), zorder=2)
        ax.scatter(
            [x] * len(seed_scores),
            seed_scores,
            color="white",
            edgecolor="#2b2b2b",
            linewidth=1.2,
            s=34,
            zorder=3,
        )
        ax.text(
            x,
            seed_scores.mean() / 2,
            f"{seed_scores.mean():.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xticks(range(len(arms)), [f"{a}\n(n={sizes[a]})" for a in arms])
    ax.set_ylabel("entity-level micro F1 (test split)")
    ax.set_title(f"Selection strategy vs. test F1  ({gap:+.3f} for uncertainty)", pad=12)
    ax.set_ylim(0, max(results["entity_f1"]) * 1.25 or 1.0)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _json_float(value) -> float | None:
    """None for NaN — a single-seed run has no std, and bare NaN is not valid JSON."""
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def write_outputs(
    run_dir: Path,
    arms: dict[str, list[dict]],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    per_type: pd.DataFrame,
    gap: float,
    meta: dict,
) -> None:
    (run_dir / "selection.json").write_text(
        json.dumps({arm: [sentence_key(ex) for ex in ex_list] for arm, ex_list in arms.items()},
                   indent=2)
    )
    results.drop(columns="per_type_f1").to_csv(run_dir / "results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")
    per_type.to_csv(run_dir / "per_type_f1.csv")

    means = results.groupby("arm")["entity_f1"].agg(["mean", "std"]).round(4)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                **meta,
                "gap": _json_float(gap),
                "n_train": int(results["n_train"].iloc[0]),
                "train_seeds": sorted(results["seed"].unique().tolist()),
                "entity_f1": {
                    arm: {"mean": _json_float(row["mean"]), "std": _json_float(row["std"])}
                    for arm, row in means.iterrows()
                },
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    run_dir = make_run_dir(cfg, args.config, args.results_dir, args.run_name)
    setup_logging(run_dir / "run.log", args.verbose)
    configure_hf_logging(quiet=not args.verbose)
    logger.info("Run directory: %s", run_dir)

    download_pet_ner(force=args.force_download)
    few_shot, pool, test = split_dataset()
    pool_examples = to_examples(pool)
    test_examples = to_examples(test)
    logger.info(
        "Split: %d few-shot, %d pool, %d test", len(few_shot), len(pool_examples), len(test_examples)
    )

    records, _sha, meta = stage_score(
        cfg, few_shot, pool_examples, skip=args.skip_scoring, limit=args.limit
    )
    arms = stage_select(cfg, records, pool_examples)

    if args.dry_run:
        for arm, examples in arms.items():
            print(f"{arm} ({len(examples)}): {[sentence_key(ex) for ex in examples]}")
        return 0

    results = stage_train(cfg, arms, test_examples)
    summary, per_type, gap = summarize(results)

    sizes = {arm: len(examples) for arm, examples in arms.items()}
    if not args.no_plot:
        plot_arms(results, sizes, gap, run_dir / "figures" / "arm_f1.png")
    write_outputs(run_dir, arms, results, summary, per_type, gap, meta)

    print(summary.to_string())
    means = results.groupby("arm")["entity_f1"].mean()
    if {"uncertainty", "random"} <= set(means.index):
        stds = results.groupby("arm")["entity_f1"].std()
        print(
            f"\nentity F1: uncertainty {means['uncertainty']:.4f} vs random "
            f"{means['random']:.4f}  ->  {gap:+.4f}   "
            f"(seed std ~{stds.mean():.4f}, n={len(cfg.train_seeds)} seeds)"
        )
    print(f"\nWrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
