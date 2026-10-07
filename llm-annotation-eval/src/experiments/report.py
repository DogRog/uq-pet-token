"""
Scores runs and collects them into tables for reporting.

- <run folder>/metrics.json: scores of one run, see eval.scoring.score
- res/pred/runs.csv: one row per run, with scores, token usage and parser issues
- res/pred/summary.csv: one row per config, mean and standard deviation over its complete runs

The tables are rebuilt from every run folder in res/pred/, not only the configs run last.
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
from eval import scoring
from experiments import runner

TASKS = ("md", "re")
SCORES = ("precision", "recall", "f1")
# per-run values that are averaged over runs in summary.csv
RUN_VALUES = (
    [f"{t}_{s}" for t in TASKS for s in SCORES]
    + [f"{t}_micro_{s}" for t in TASKS for s in SCORES]
    + ["prompt_tokens", "completion_tokens", "reasoning_tokens", "llm_seconds"]
)


def score_run(directory: pathlib.Path, gold_file: pathlib.Path, task: str) -> typing.Dict[str, typing.Any]:
    """:param task: "end-to-end" scores mentions and relations, "md" only mentions"""
    predicted = autoannotate.JsonImporter(runner.predictions_file(directory, task)).do_import()
    tasks = ("md", "re") if task == "end-to-end" else ("md",)
    metrics = scoring.score(autoannotate.JsonImporter(gold_file).do_import(), predicted, tasks)
    runner.write_json(directory / "metrics.json", metrics)
    return metrics


def _run_row(directory: pathlib.Path) -> typing.Dict[str, typing.Any]:
    manifest = json.loads((directory / "run.json").read_text(encoding="utf8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf8"))
    totals = manifest["totals"]
    row: typing.Dict[str, typing.Any] = {
        "config": manifest["config"],
        "run": manifest["run"],
        **manifest["settings"],
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
    row.update({
        "prompt_tokens": totals["all"]["prompt_tokens"],
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
    """Writes runs.csv and summary.csv to pred_dir, :return: the summary rows"""
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

    table = Table(title="Mean over complete runs (± SD over runs); F1 is the mean over documents")
    for column in ("config", "task", "reasoning", "runs", "docs", "MD F1", "RE F1", "tokens in", "tokens out"):
        table.add_column(column)
    for row in summary:
        table.add_row(
            row["config"], row["task"], str(row["reasoning_effort"] or "-"), str(row["runs"]), row["documents"],
            fmt(row, "md_f1"), fmt(row, "re_f1"),
            fmt(row, "prompt_tokens", percent=False), fmt(row, "completion_tokens", percent=False),
        )
    Console().print(table)
