"""Grid orchestration: (budget x strategy x seed) -> train -> evaluate -> record.

Each run gets its own directory under results/<run_id>/ holding a config.yaml
snapshot, per-cell records.jsonl, a metrics.json summary, and run.log. Completed
cells are skipped when resuming a run, so the grid is resumable.
"""

import json
import logging
import multiprocessing
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .config import RESULTS_DIR, ExperimentConfig, TrainConfig, config_to_yaml, load_config
from .data import sentence_key, split_pool_test
from .llm_scoring import load_cache
from .train import evaluate_model_on, train_token_classifier
from .uncertainty import METRICS, compute_metric, select, strategy_metric

logger = logging.getLogger("uq_pet")


def cell_id(budget: int, strategy: str, seed: int) -> str:
    return f"budget={budget}|strategy={strategy}|seed={seed}"


def create_run_dir(cfg: ExperimentConfig, config_stem: str) -> Path:
    """Create results/<config_stem>_<timestamp>/ with a config.yaml snapshot."""
    run_id = f"{config_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_to_yaml(cfg))
    return run_dir


def resume_run_dir(run_id: str) -> tuple[ExperimentConfig, Path]:
    """Re-open an existing run; the config comes from its snapshot."""
    run_dir = RESULTS_DIR / run_id
    snapshot = run_dir / "config.yaml"
    if not snapshot.exists():
        raise FileNotFoundError(f"No config.yaml snapshot in {run_dir}")
    return load_config(snapshot), run_dir


def latest_run_id() -> str | None:
    """Newest run dir under results/ that has records, by mtime."""
    candidates = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "records.jsonl").exists()
    ] if RESULTS_DIR.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "records.jsonl").stat().st_mtime).name


def load_completed_runs(records_path: Path) -> dict[str, dict]:
    completed = {}
    if records_path.exists():
        with open(records_path) as f:
            for line in f:
                record = json.loads(line)
                completed[cell_id(record["budget_pct"], record["strategy"], record["seed"])] = record
    return completed


def write_metrics_summary(records: list[dict], out_path: Path) -> dict:
    """Aggregate per-(budget, strategy) mean/std of the headline metrics."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["budget_pct"], r["strategy"]), []).append(r["metrics"])

    def stats(values: list[float]) -> dict:
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def box_type(strategy: str) -> str | None:
        name = strategy_metric(strategy)
        return METRICS[name].box if name in METRICS else None

    summary = {
        "run_id": out_path.parent.name,
        "n_cells": len(records),
        "cells": [
            {
                "budget_pct": budget,
                "strategy": strategy,
                "box_type": box_type(strategy),
                "n_seeds": len(metrics),
                "entity_f1": stats([m["entity_f1"] for m in metrics]),
                "token_accuracy": stats([m["token_accuracy"] for m in metrics]),
            }
            for (budget, strategy), metrics in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def _train_cell(selected: list[dict], test_examples: list[dict], train_cfg: TrainConfig,
                seed: int) -> dict:
    """Train one grid cell and evaluate it on the test set.

    Runs in a worker process when cfg.workers > 1, so it must stay a
    top-level (picklable) function. Training re-seeds per cell, so results
    do not depend on which process a cell lands in.
    """
    model, tokenizer = train_token_classifier(selected, train_cfg, seed)
    metrics = evaluate_model_on(model, tokenizer, test_examples, train_cfg)
    del model
    return metrics


def run_grid(cfg: ExperimentConfig, run_dir: Path) -> None:
    records_path = run_dir / "records.jsonl"
    pool, test = split_pool_test(seed=cfg.llm.seed)
    pool_examples = list(pool)
    test_examples = list(test)
    by_key = {sentence_key(ex): ex for ex in pool_examples}
    all_keys = list(by_key)

    # Precompute uncertainty scores per metric from the LLM sample cache.
    metric_names = {
        name for s in cfg.strategies if (name := strategy_metric(s)) is not None
    }
    scores_by_metric: dict[str, dict[str, float]] = {}
    if metric_names:
        cache = load_cache(cfg.llm.cache_path())
        missing = [k for k in all_keys if k not in cache]
        if missing:
            raise RuntimeError(
                f"{len(missing)} pool sentences missing from LLM cache "
                f"{cfg.llm.cache_path()}; run `score-pool` first."
            )
        # White-box metrics need model internals in the cache; fail before
        # any training rather than mid-grid.
        whitebox = sorted(n for n in metric_names if METRICS[n].box == "white")
        if whitebox:
            no_entropy = [k for k in all_keys if not cache[k].get("token_entropies")]
            if no_entropy:
                raise RuntimeError(
                    f"Strategies {whitebox} are white-box but {len(no_entropy)} cached "
                    f"records in {cfg.llm.cache_path()} have no token_entropies (cache "
                    "was produced by a black-box API backend). Re-run score-pool with "
                    "llm.backend: mlx."
                )
        for name in metric_names:
            scores_by_metric[name] = {
                k: compute_metric(name, cache[k]) for k in all_keys
            }

    completed = load_completed_runs(records_path)

    # Full pool (100%) is strategy-independent: run it once per seed as "full".
    grid = []
    for budget in cfg.budgets:
        strategies = ["full"] if budget >= 100 else cfg.strategies
        for strategy in strategies:
            for seed in range(cfg.repeats):
                grid.append((budget, strategy, seed))

    # Per-cell detail goes to run.log (the console handler only shows
    # warnings); the console gets one progress bar over the whole grid.
    print(f"Grid: {len(grid)} cells ({len(completed)} already recorded)")
    logger.info(f"Grid: {len(grid)} cells ({len(completed)} already recorded)")
    pending = [cell for cell in grid if cell_id(*cell) not in completed]

    # Selection is cheap and deterministic, so it happens up front in the
    # parent; workers only train and evaluate.
    selected_by_cell: dict[tuple, list[str]] = {}
    for budget, strategy, seed in pending:
        n = round(len(all_keys) * budget / 100)
        if strategy == "full":
            selected_keys = all_keys
        else:
            selected_keys = select(
                strategy, all_keys,
                scores_by_metric.get(strategy_metric(strategy)), n, seed,
            )
        selected_by_cell[(budget, strategy, seed)] = selected_keys
        logger.info(f"{cell_id(budget, strategy, seed)} -> {len(selected_keys)} sentences")

    progress = tqdm(total=len(grid), initial=len(grid) - len(pending),
                    desc="Grid", unit="cell")
    with open(records_path, "a") as runs_file:

        def record_cell(cell: tuple, metrics: dict) -> None:
            budget, strategy, seed = cell
            cid = cell_id(*cell)
            selected_keys = selected_by_cell[cell]
            metric_name = strategy_metric(strategy)
            mean_len = sum(len(by_key[k]["tokens"]) for k in selected_keys) / len(selected_keys)
            record = {
                "budget_pct": budget,
                "n_selected": len(selected_keys),
                "strategy": strategy,
                "metric": metric_name,
                "box_type": METRICS[metric_name].box if metric_name else None,
                "seed": seed,
                "selected_keys": selected_keys,
                "selected_mean_tokens": mean_len,
                "metrics": metrics,
                "train_config": vars(cfg.train),
                "llm_config": {
                    "backend": cfg.llm.backend,
                    "model": cfg.llm.model,
                    "num_samples": cfg.llm.num_samples,
                    "temperature": cfg.llm.temperature,
                    "seed": cfg.llm.seed,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            runs_file.write(json.dumps(record) + "\n")
            runs_file.flush()
            completed[cid] = record
            write_metrics_summary(list(completed.values()), run_dir / "metrics.json")
            logger.info(f"{cid} entity_f1={metrics['entity_f1']:.4f} "
                        f"token_acc={metrics['token_accuracy']:.4f}")
            progress.set_postfix_str(f"{cid} f1={metrics['entity_f1']:.3f}")
            progress.update(1)

        if cfg.workers > 1:
            # One worker process per concurrent cell, sharing the GPU.
            # Records land in completion order; resume keys on cell_id, so
            # order in records.jsonl does not matter.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as pool:
                futures = {
                    pool.submit(_train_cell, [by_key[k] for k in selected_by_cell[cell]],
                                test_examples, cfg.train, cell[2]): cell
                    for cell in pending
                }
                for future in as_completed(futures):
                    record_cell(futures[future], future.result())
        else:
            for cell in pending:
                progress.set_postfix_str(cell_id(*cell))
                metrics = _train_cell([by_key[k] for k in selected_by_cell[cell]],
                                      test_examples, cfg.train, cell[2])
                record_cell(cell, metrics)

    progress.close()
    print(f"Done. Records in {records_path}")
    logger.info(f"Done. Records in {records_path}")
