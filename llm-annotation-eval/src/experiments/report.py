"""Scores runs and collects them into tables for reporting.

- <run folder>/metrics.json: scores of one run, see eval.scoring.score
- res/pred/runs.csv: one row per run, with scores, token usage and parser issues
- res/pred/summary.csv: one row per config, mean and standard deviation over its complete runs
- <run folder>/<task>/test-predictions.jsonl: the run's predictions on the test sentences, for
  uq-pet-token, see export_test_predictions

The tables are rebuilt from every run folder in res/pred/, not only the configs run last.
``scores_so_far`` scores a run that is still going, for the notebook's live view.
"""
import csv
import json
import os
import pathlib
import statistics
import typing

from rich.console import Console
from rich.table import Table

import autoannotate
from eval import datasets, scoring, test_split
from experiments import runner

TASKS = ("md", "re")
SCORES = ("precision", "recall", "f1")
# per-run values that are averaged over runs in summary.csv
# scores on uq-pet-token's test sentences, see eval.test_split: metrics.json key -> column
TEST_SCORES = {
    "entity_precision": "test_precision",
    "entity_recall": "test_recall",
    "entity_f1": "test_f1",
    "entity_macro_f1": "test_macro_f1",
}
RUN_VALUES = (
    [f"{t}_{s}" for t in TASKS for s in SCORES]
    + [f"{t}_micro_{s}" for t in TASKS for s in SCORES]
    + list(TEST_SCORES.values())
    + ["prompt_tokens", "cached_tokens", "cache_write_tokens"]
    + ["completion_tokens", "reasoning_tokens", "llm_seconds"]
)


def score_run(
    directory: pathlib.Path, gold_file: pathlib.Path, task: str, test: typing.List[test_split.Row],
    types: typing.Sequence[str],
) -> typing.Dict[str, typing.Any]:
    """Scores a run on the gold documents and, if it covers them, on the test sentences.

    Args:
        directory: the run folder
        gold_file: the gold documents, see eval.datasets.read_documents
        task: "end-to-end" is scored on mentions and relations, "md" on mentions, and
            "md-test-sentences" only on the test sentences
        test: uq-pet-token's test sentences of the dataset, see eval.test_split
        types: the dataset's mention types, see eval.datasets
    """
    predicted = autoannotate.JsonImporter(runner.predictions_file(directory, task)).do_import()
    if task == "md-test-sentences":
        metrics = {"documents": len(predicted)}
    else:
        tasks = ("md", "re") if task == "end-to-end" else ("md",)
        metrics = scoring.score(datasets.read_documents(gold_file), predicted, tasks)
    test_predicted, dropped = test_split.predict_test_split(test, predicted, types)
    if test_predicted is not None:
        # mentions dropped as they overlap an earlier one, which BIO tags can't represent
        test_scores = test_split.evaluate(test, test_predicted)
        metrics["test"] = {**test_scores, "dropped_overlapping": dropped}
    runner.write_json(directory / "metrics.json", metrics)
    return metrics


def scores_so_far(
    directory: pathlib.Path, task: str, gold: typing.List[autoannotate.Document],
    test: typing.List[test_split.Row], types: typing.Sequence[str],
) -> typing.Dict[str, typing.Any]:
    """The scores of the documents a run has predicted so far, also while it runs.

    Like score_run, but on the test sentences of the predicted documents only, and without
    writing metrics.json; with the tokens and dropped answer lines of every call so far.

    Args:
        directory: the run folder
        task: the config's task
        gold: the gold documents, see eval.datasets.read_documents
        test: uq-pet-token's test sentences of the dataset
        types: the dataset's mention types
    """
    predicted = runner.read_predictions(directory, task)
    calls = [c for c in runner.read_calls(directory) if "error" not in c]
    scores: typing.Dict[str, typing.Any] = {
        "documents": len(predicted),
        "calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "cached_tokens": runner.reported_sum(calls, "cached_tokens"),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "reasoning_tokens": runner.reported_sum(calls, "reasoning_tokens"),
        "truncated_answers": sum(c["finish_reason"] == "length" for c in calls),
        "dropped_lines": sum(i["kind"].startswith("dropped") for c in calls for i in c["issues"]),
    }
    if not predicted:
        return scores
    if task != "md-test-sentences":
        tasks = ("md", "re") if task == "end-to-end" else ("md",)
        metrics = scoring.score(gold, predicted, tasks)
        for t in tasks:
            scores[f"{t}_f1"] = metrics[t]["macro"]["f1"]["mean"]
            scores[f"{t}_micro_f1"] = metrics[t]["micro"]["f1"]
    ids = {d.id for d in predicted}
    covered = [
        r for r in test if test_split.sentence_id(r) in ids or r["document name"] in ids
    ]
    scores["test_sentences"] = len(covered)
    if covered:
        test_predicted, _ = test_split.predict_test_split(covered, predicted, types)
        test_scores = test_split.evaluate(covered, test_predicted)
        scores["test_f1"] = test_scores["entity_f1"]
        scores["test_partial_f1"] = test_scores["entity_partial_f1"]
    return scores


def export_test_predictions(
    pred_dir: pathlib.Path,
) -> typing.Tuple[typing.List[pathlib.Path], typing.List[pathlib.Path]]:
    """Writes test-predictions.jsonl next to every pet.jsonl in pred_dir, for uq-pet-token.

    Each file holds the test split of uq-pet-token for the run's dataset (e.g. the 84 sentences
    of res/gold/pet-v1.1-test.jsonl for PET) in the same format and order, with the predicted BIO
    tags; see eval.test_split.export. Runs export theirs when they are scored (batch.run_config).

    Returns:
        The written files, and the predictions skipped as they don't cover the test split.
    """
    tests: typing.Dict[str, typing.List[test_split.Row]] = {}
    # runs of a config: <folder>/run-<i>/<task>/pet.jsonl; earlier predictions: <folder>/<task>/
    files = sorted(pred_dir.glob("*/run-*/*/pet.jsonl")) + sorted(pred_dir.glob("*/*/pet.jsonl"))
    written, skipped = [], []
    for predictions_file in files:
        # predictions from before the configs have no run.json, they are PET's
        manifest_file = predictions_file.parent.parent / "run.json"
        name = "pet"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf8"))
            name = runner.manifest_settings(manifest)["dataset"]
        dataset = datasets.DATASETS[name]
        if name not in tests:
            tests[name] = test_split.load_test_split(dataset.test_file)
        out_file = test_split.export(predictions_file, tests[name], dataset.types)
        (written if out_file is not None else skipped).append(out_file or predictions_file)
    return written, skipped


def _run_row(directory: pathlib.Path) -> typing.Dict[str, typing.Any]:
    manifest = json.loads((directory / "run.json").read_text(encoding="utf8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf8"))
    totals = manifest["totals"]
    row: typing.Dict[str, typing.Any] = {
        "config": manifest["config"],
        "run": manifest["run"],
        **runner.manifest_settings(manifest),
        "documents": metrics["documents"],
        "complete": metrics["documents"] == manifest["documents_planned"],
    }
    for task in TASKS:
        # empty for tasks the run did not do, e.g. relations in mention detection runs
        task_metrics = metrics.get(task)
        for name in SCORES:
            row[f"{task}_{name}"] = task_metrics["macro"][name]["mean"] if task_metrics else None
            row[f"{task}_{name}_sd_docs"] = task_metrics["macro"][name]["std"] if task_metrics else None
        for name in SCORES:
            row[f"{task}_micro_{name}"] = task_metrics["micro"][name] if task_metrics else None
    # empty if the run did not annotate all test sentences
    test_metrics = metrics.get("test")
    for key, column in TEST_SCORES.items():
        row[column] = test_metrics[key] if test_metrics else None
    row.update({
        "prompt_tokens": totals["all"]["prompt_tokens"],
        # run.json totals from before cached tokens were logged don't have them
        "cached_tokens": totals["all"].get("cached_tokens"),
        "cache_write_tokens": totals["all"].get("cache_write_tokens"),
        "completion_tokens": totals["all"]["completion_tokens"],
        "reasoning_tokens": totals["all"]["reasoning_tokens"],
        "llm_seconds": totals["all"]["duration_s"],
        "truncated_answers": totals["all"]["finish_reasons"].get("length", 0),
        "dropped_mention_lines": sum(n for k, n in totals["mentions"]["issues"].items() if k.startswith("dropped")),
        "dropped_relation_lines": sum(n for k, n in totals["relations"]["issues"].items() if k.startswith("dropped")),
    })
    return row


def _summary_row(config: str, runs: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
    complete = [r for r in runs if r["complete"]]
    documents = sorted({r["documents"] for r in complete})
    row: typing.Dict[str, typing.Any] = {
        "config": config,
        "dataset": runs[0]["dataset"],
        "task": runs[0]["task"],
        "provider": runs[0]["provider"],
        "model": runs[0]["model"],
        "reasoning_effort": runs[0]["reasoning_effort"],
        "runs": len(complete),
        "documents": "/".join(map(str, documents)),
    }
    for key in RUN_VALUES:
        values = [r[key] for r in complete if r[key] is not None]
        row[key] = statistics.mean(values) if values else None
        # sample standard deviation over runs, undefined for a single run
        row[f"{key}_sd_runs"] = statistics.stdev(values) if len(values) > 1 else None
    return row


def _write_csv(path: pathlib.Path, rows: typing.List[typing.Dict[str, typing.Any]]) -> None:
    # write to a temporary file and rename, so parallel invocations never leave a half-written table
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def run_dirs(pred_dir: pathlib.Path) -> typing.List[pathlib.Path]:
    """Run folders in pred_dir that have been scored."""
    return sorted(d.parent for d in pred_dir.glob("*/run-*/metrics.json") if (d.parent / "run.json").exists())


def collect_runs(pred_dir: pathlib.Path) -> typing.List[typing.Dict[str, typing.Any]]:
    """:return: one row per scored run in pred_dir, the rows of runs.csv"""
    runs = [_run_row(d) for d in run_dirs(pred_dir)]
    runs.sort(key=lambda r: (r["config"], r["run"]))
    return runs


def write_summary(pred_dir: pathlib.Path) -> typing.List[typing.Dict[str, typing.Any]]:
    """Writes runs.csv and summary.csv to pred_dir and returns the summary rows."""
    runs = collect_runs(pred_dir)
    if not runs:
        return []
    _write_csv(pred_dir / "runs.csv", runs)

    by_config: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]] = {}
    for run in runs:
        by_config.setdefault(run["config"], []).append(run)
    summary = [_summary_row(config, config_runs) for config, config_runs in by_config.items()]
    _write_csv(pred_dir / "summary.csv", summary)
    return summary


def print_summary(summary: typing.List[typing.Dict[str, typing.Any]]) -> None:
    def fmt(row: typing.Dict[str, typing.Any], key: str, percent: bool = True) -> str:
        if row[key] is None:
            return "-"
        value = f"{row[key]:.1%}" if percent else f"{row[key]:,.0f}"
        if row[f"{key}_sd_runs"] is not None:
            value += f" ± {row[f'{key}_sd_runs']:.1%}" if percent else f" ± {row[f'{key}_sd_runs']:,.0f}"
        return value

    table = Table(
        title="Mean over complete runs (± SD over runs); MD and RE F1 are means over documents, "
        "test F1 is on uq-pet-token's test sentences of the dataset"
    )
    columns = ("config", "dataset", "task", "reasoning", "runs", "docs", "MD F1", "RE F1", "test F1")
    for column in (*columns, "tokens in", "tokens out"):
        table.add_column(column)
    for row in summary:
        table.add_row(
            row["config"], row["dataset"], row["task"], str(row["reasoning_effort"] or "-"), str(row["runs"]), row["documents"],
            fmt(row, "md_f1"), fmt(row, "re_f1"), fmt(row, "test_f1"),
            fmt(row, "prompt_tokens", percent=False), fmt(row, "completion_tokens", percent=False),
        )
    Console().print(table)
