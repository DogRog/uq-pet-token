"""Runs the experiments defined by config files.

Annotates the gold documents with the configured LLM, scores the predictions, and updates
res/pred/runs.csv and res/pred/summary.csv.

    uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml
    uv run python src/run-configs.py configs/          # every *.yaml in the folder

Running a config again resumes its unfinished runs and rescores all of them. If a run of a config
fails, its remaining runs are skipped and the next config is run; the exit code is 1 then.
"""
import argparse
import pathlib
import sys
import traceback

import dotenv
import yaml

import autoannotate
from experiments import config as run_config
from experiments import report, runner

REPO_DIR = pathlib.Path(__file__).parent.parent
GOLD_FILE = REPO_DIR / "res" / "gold" / "pet.jsonl"
PRED_DIR = REPO_DIR / "res" / "pred"


def main():
    parser = argparse.ArgumentParser(description="Annotate and score with the given run configs.")
    parser.add_argument("configs", nargs="+", type=pathlib.Path,
                        help="config files, or folders to run every *.yaml / *.yml in")
    parser.add_argument("--workers", type=int, default=4,
                        help="documents annotated at the same time per run (default: %(default)s)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = []
    for path in args.configs:
        if not path.exists():
            parser.error(f"{path} does not exist")
        paths += sorted([*path.glob("*.yaml"), *path.glob("*.yml")]) if path.is_dir() else [path]
    # check every config and API key before the first LLM call
    try:
        configs = [run_config.load_config(p) for p in paths]
    except (ValueError, yaml.YAMLError) as e:
        parser.error(str(e))
    names = [c.name for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        parser.error(f"config names must be unique, as they name the output folders: {duplicates}")
    dotenv.load_dotenv(REPO_DIR / ".env")
    clients = {c.provider: autoannotate.get_provider(c.provider).client() for c in configs}

    failed = []
    for config in configs:
        for run in range(1, config.repeats + 1):
            directory = runner.run_dir(PRED_DIR, config, run)
            try:
                runner.annotate_run(config, run, directory, GOLD_FILE, clients[config.provider], args.workers)
            except Exception:
                traceback.print_exc()
                print(f"{config.name} run {run} failed, skipping the rest of {config.name}.")
                failed.append(config.name)
                break
            finally:
                if runner.predictions_file(directory, config.task).exists():
                    report.score_run(directory, GOLD_FILE, config.task)

    report.print_summary(report.write_summary(PRED_DIR))
    print(f"Tables: {PRED_DIR / 'summary.csv'} (per config), {PRED_DIR / 'runs.csv'} (per run).")
    if failed:
        print(f"Failed: {', '.join(failed)}. Run them again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
