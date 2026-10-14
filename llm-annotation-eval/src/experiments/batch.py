"""Runs configs: checks them before the first LLM call, then annotates and scores each run.

notebooks/run_experiments.py runs configs with this. After each run that has predictions, the run is
scored (report.score_run) and its test-predictions.jsonl written for uq-pet-token; a run that
fails skips the remaining runs of its config. Afterwards report.write_summary rebuilds runs.csv
and summary.csv from every run folder.

Running a config again resumes its unfinished runs and rescores all of them.
"""
import dataclasses
import fnmatch
import pathlib
import traceback
import typing

import dotenv

import autoannotate
from eval import datasets, test_split
from experiments import config as run_config
from experiments import report, runner

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
PRED_DIR = REPO_DIR / "res" / "pred"


def select(
    configs: typing.List[run_config.RunConfig], patterns: typing.Sequence[str]
) -> typing.List[run_config.RunConfig]:
    """The configs whose names match one of the patterns, e.g. 'gemma-*'; all if there are none."""
    if not patterns:
        return list(configs)
    return [c for c in configs if any(fnmatch.fnmatchcase(c.name, p) for p in patterns)]


def problems(configs: typing.List[run_config.RunConfig]) -> typing.List[str]:
    """What keeps the configs from running together, e.g. missing gold files; empty if nothing."""
    found = []
    names = [c.name for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        found.append(f"Config names must be unique, as they name the output folders: {duplicates}")
    for c in configs:
        dataset = datasets.DATASETS[c.dataset]
        for path in (dataset.gold_file, dataset.test_file, dataset.bootstrap_file):
            if not path.exists():
                found.append(
                    f"{path.name} is missing, needed by {c.name}; generate it with uv run "
                    "--project ../uq-pet-token python scripts/uq-pet-token-splits.py"
                )
    return found


def clients(configs: typing.List[run_config.RunConfig]) -> typing.Dict[str, typing.Any]:
    """One client per provider of the configs, with the API keys from .env.

    Raises if a key is missing, before the first run starts.
    """
    dotenv.load_dotenv(REPO_DIR / ".env")
    return {c.provider: autoannotate.get_provider(c.provider).client() for c in configs}


@dataclasses.dataclass
class RunResult:
    run: int
    directory: pathlib.Path
    # the scores in metrics.json, None if the run has no predictions
    metrics: typing.Optional[typing.Dict[str, typing.Any]]
    # why the run stopped, None if it finished
    error: typing.Optional[str] = None


# called after each document: the run, its folder, the document id and the error if it was skipped
OnDocument = typing.Callable[[int, pathlib.Path, str, typing.Optional[BaseException]], None]


def run_config(
    config: run_config.RunConfig, client, workers: int = 4, pred_dir: pathlib.Path = PRED_DIR,
    on_document: typing.Optional[OnDocument] = None,
) -> typing.List[RunResult]:
    """Annotates and scores the runs of a config, up to the first that fails.

    Args:
        config: the config, its runs are <pred_dir>/<config name>/run-<i>/
        client: the client of the config's provider, see clients
        workers: documents annotated at the same time per run
        pred_dir: where the run folders are, res/pred/ unless testing
        on_document: called after each document of a run, see runner.annotate_run
    """
    dataset = datasets.DATASETS[config.dataset]
    test = test_split.load_test_split(dataset.test_file)
    results = []
    for run in range(1, config.repeats + 1):
        directory = runner.run_dir(pred_dir, config, run)
        result = RunResult(run, directory, None)
        results.append(result)

        def on_run_document(document_id, error, run=run, directory=directory):
            if on_document is not None:
                on_document(run, directory, document_id, error)

        try:
            # a md-test-sentences run annotates the test sentences instead of the documents
            runner.annotate_run(
                config, run, directory, dataset.input_file(config.task), client, workers,
                on_document=on_run_document,
            )
        except Exception as e:
            traceback.print_exc()
            result.error = f"{type(e).__name__}: {e}"
        finally:
            predictions_file = runner.predictions_file(directory, config.task)
            if predictions_file.exists():
                result.metrics = report.score_run(
                    directory, dataset.gold_file, config.task, test, dataset.types
                )
                test_split.export(predictions_file, test, dataset.types)
        if result.error is not None:
            break
    return results
