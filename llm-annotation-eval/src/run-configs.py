"""Runs the experiments defined by config files.

Annotates the gold documents of the configured dataset with the configured LLM, scores the
predictions, writes each run's test-predictions.jsonl for uq-pet-token (see eval.test_split), and
updates res/pred/runs.csv and res/pred/summary.csv.

    uv run python src/run-configs.py configs/deepseek-v4.1-flash.yaml
    uv run python src/run-configs.py configs/          # every *.yaml in the folder
    uv run python src/run-configs.py configs/pet/md.yaml --only 'gemma-4-31b-*'

A config file holds one config or, under "models", one per model and reasoning effort (see
experiments.config.load_configs).

Running a config again resumes its unfinished runs and rescores all of them. If a run of a config
fails, its remaining runs are skipped and the next config is run; the exit code is 1 then.
"""
import argparse
import fnmatch
import pathlib
import sys
import traceback

import dotenv
import yaml

import autoannotate
from eval import datasets, test_split
from experiments import config as run_config
from experiments import report, runner

REPO_DIR = pathlib.Path(__file__).parent.parent
PRED_DIR = REPO_DIR / "res" / "pred"


def main():
    parser = argparse.ArgumentParser(description="Annotate and score with the given run configs.")
    parser.add_argument("configs", nargs="+", type=pathlib.Path,
                        help="config files, or folders to run every *.yaml / *.yml in")
    parser.add_argument("--workers", type=int, default=4,
                        help="documents annotated at the same time per run (default: %(default)s)")
    parser.add_argument("--only", nargs="+", metavar="PATTERN",
                        help="only run the configs whose names match one of these patterns, e.g. 'gemma-*'")
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
        configs = [c for p in paths for c in run_config.load_configs(p)]
    except (ValueError, yaml.YAMLError) as e:
        parser.error(str(e))
    if args.only:
        configs = [c for c in configs if any(fnmatch.fnmatchcase(c.name, o) for o in args.only)]
        if not configs:
            parser.error(f"no config name matches {args.only}")
    names = [c.name for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        parser.error(f"config names must be unique, as they name the output folders: {duplicates}")
    for c in configs:
        dataset = datasets.DATASETS[c.dataset]
        for path in (dataset.gold_file, dataset.test_file):
            if not path.exists():
                parser.error(
                    f"{path} is missing, needed by {c.name}; generate it with "
                    "uv run --project ../uq-pet-token python scripts/uq-pet-token-splits.py"
                )
    dotenv.load_dotenv(REPO_DIR / ".env")
    clients = {c.provider: autoannotate.get_provider(c.provider).client() for c in configs}

    failed = []
    for config in configs:
        dataset = datasets.DATASETS[config.dataset]
        test = test_split.load_test_split(dataset.test_file)
        for run in range(1, config.repeats + 1):
            directory = runner.run_dir(PRED_DIR, config, run)
            try:
                # a md-test-sentences run annotates the test sentences instead of the documents
                input_file = dataset.input_file(config.task)
                client = clients[config.provider]
                runner.annotate_run(config, run, directory, input_file, client, args.workers)
            except Exception:
                traceback.print_exc()
                print(f"{config.name} run {run} failed, skipping the rest of {config.name}.")
                failed.append(config.name)
                break
            finally:
                if runner.predictions_file(directory, config.task).exists():
                    report.score_run(directory, dataset.gold_file, config.task, test, dataset.types)
                    predictions_file = runner.predictions_file(directory, config.task)
                    test_split.export(predictions_file, test, dataset.types)

    report.print_summary(report.write_summary(PRED_DIR))
    print(f"Tables: {PRED_DIR / 'summary.csv'} (per config), {PRED_DIR / 'runs.csv'} (per run).")
    if failed:
        print(f"Failed: {', '.join(failed)}. Run them again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
