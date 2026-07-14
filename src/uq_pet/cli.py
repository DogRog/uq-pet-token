"""Command-line entry points: download-data | score-pool | run | report."""

import argparse
import asyncio
import logging
from pathlib import Path

from .config import load_config


def _setup_logging(log_path: Path | None = None) -> None:
    """Full per-cell detail goes to run.log; the console only shows warnings
    (progress lives in a tqdm bar, which log lines would break up)."""
    logger = logging.getLogger("uq_pet")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    handlers: list[logging.Handler] = [stream_handler]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)


def cmd_download_data(args) -> None:
    from .data import download_pet_ner

    dest = download_pet_ner(force=args.force)
    print(f"Dataset at {dest}")


def cmd_score_pool(args) -> None:
    from .data import split_pool_test
    from .llm_scoring import score_pool

    cfg = load_config(args.config)
    pool, _ = split_pool_test(seed=cfg.llm.seed)
    cache = asyncio.run(score_pool(cfg.llm, pool, limit=args.limit))
    print(f"Cache now holds {len(cache)} sentences at {cfg.llm.cache_path()}")


def cmd_run(args) -> None:
    from .experiment import create_run_dir, resume_run_dir, run_grid

    if args.resume:
        cfg, run_dir = resume_run_dir(args.resume)
    else:
        config_path = Path(args.config)
        cfg = load_config(config_path)
        run_dir = create_run_dir(cfg, config_path.stem)
    _setup_logging(run_dir / "run.log")
    print(f"Run dir: {run_dir}")
    run_grid(cfg, run_dir)


def cmd_report(args) -> None:
    from .config import RESULTS_DIR
    from .experiment import latest_run_id
    from .reporting import build_report

    run_id = args.run_id or latest_run_id()
    if run_id is None:
        raise SystemExit("No runs found under results/; run an experiment first.")
    build_report(RESULTS_DIR / run_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uq-pet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download-data", help="download the PET dataset to data/raw/")
    p_download.add_argument("--force", action="store_true", help="re-download even if present")
    p_download.set_defaults(func=cmd_download_data)

    p_score = sub.add_parser("score-pool", help="LLM repeated-sampling pass over the pool")
    p_score.add_argument("--config", required=True, help="YAML experiment config")
    p_score.add_argument("--limit", type=int, default=None,
                         help="score only the first N pool sentences (smoke test)")
    p_score.set_defaults(func=cmd_score_pool)

    p_run = sub.add_parser("run", help="run the selection/training grid")
    group = p_run.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="YAML experiment config (starts a new run)")
    group.add_argument("--resume", metavar="RUN_ID",
                       help="resume an existing results/<run_id>/ (config from its snapshot)")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="figures + summary table for a run")
    p_report.add_argument("--run-id", default=None,
                          help="run to report on (default: latest)")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
