import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys

    import marimo as mo
    import pandas as pd
    return json, mo, pd, sys


@app.cell
def _(mo, sys):
    REPO_DIR = mo.notebook_dir().parent
    PRED_DIR = REPO_DIR / "res" / "pred"
    # read the runs with the same code that writes res/pred/runs.csv
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from experiments import report
    return PRED_DIR, report


@app.cell
def _():
    # names in the tables; models not listed here show the last part of their id
    MODEL_NAMES = {
        "deepseek-ai/DeepSeek-V4.1-Flash": "DeepSeek-V4.1-Flash",
        "Qwen/Qwen3.8-27B-FP8": "Qwen3.8-27B",
        "gpt-oss-120b": "gpt-oss-120b",
        "RedHatAI/gemma-4-31B-it-FP8-block": "Gemma-4-31B",
        "google/gemma-4-E4B-it": "Gemma-4-E4B",
        "GaleneAI/Magistral-Small-2509-FP8-Dynamic": "Magistral-Small-2509",
    }
    # reasoning_effort as sent; None means the parameter was left out (the model's default)
    REASONING_NAMES = {None: "default", "none": "off", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}
    return MODEL_NAMES, REASONING_NAMES


@app.cell
def _(mo):
    mo.md("""
    # Evaluation results

    One row per run config, over its runs in `res/pred/`. Scores are percentages: precision, recall
    and F1 per document, averaged over the documents of a run (macro, as `src/llm-metrics.py`
    prints them), then averaged over the runs. A predicted mention counts as correct if its type
    matches a gold mention and their tokens overlap; each gold mention can be matched once.
    """)
    return


@app.cell
def _(PRED_DIR, json, pd, report):
    runs = pd.DataFrame(report.collect_runs(PRED_DIR))

    # micro scores per mention / relation type, from each run's metrics.json
    _rows = []
    for _directory in report.run_dirs(PRED_DIR):
        _manifest = json.loads((_directory / "run.json").read_text(encoding="utf8"))
        _metrics = json.loads((_directory / "metrics.json").read_text(encoding="utf8"))
        for _scored in ("md", "re"):
            for _type, _scores in _metrics.get(_scored, {}).get("per_type", {}).items():
                _rows.append({"config": _manifest["config"], "run": _manifest["run"], "scored": _scored,
                              "type": _type, **_scores})
    per_type = pd.DataFrame(_rows)
    return per_type, runs


@app.cell
def _(mo, runs):
    mo.stop(runs.empty, mo.callout(mo.md("No scored runs in `res/pred/` yet, run `src/run-configs.py` first."), kind="warn"))

    _task_names = {"md": "Mention detection", "end-to-end": "End to end (mentions and relations)"}
    _tasks = [t for t in _task_names if t in set(runs["task"])]
    task = mo.ui.dropdown(options={_task_names[t]: t for t in _tasks}, value=_task_names[_tasks[0]], label="Task")
    spread = mo.ui.radio(
        options={"over runs": "runs", "over documents": "documents"}, value="over runs", inline=True,
        label="± is the standard deviation",
    )
    only_complete = mo.ui.checkbox(value=True, label="Only runs with all documents")
    mo.hstack([task, spread, only_complete], justify="start", gap=2)
    return only_complete, spread, task


@app.cell
def _(MODEL_NAMES, REASONING_NAMES, only_complete, pd, runs, spread, task):
    selected = runs[runs["task"] == task.value]
    if only_complete.value:
        selected = selected[selected["complete"]]
    scored = ["md"] if task.value == "md" else ["md", "re"]
    _reasoning_order = list(REASONING_NAMES)

    def model_name(model):
        return MODEL_NAMES.get(model, model.split("/")[-1])

    def reasoning_name(effort):
        return REASONING_NAMES.get(effort, effort)

    # one row per config with numeric means and standard deviations, as fractions
    _rows = []
    for _config, _group in selected.groupby("config"):
        _first = _group.iloc[0]
        _row = {
            "config": _config,
            "model": model_name(_first["model"]),
            "reasoning": reasoning_name(_first["reasoning_effort"]),
            "runs": len(_group),
            "documents": "/".join(str(d) for d in sorted(_group["documents"].unique())),
            "_order": (model_name(_first["model"]).lower(), _reasoning_order.index(_first["reasoning_effort"])
                       if _first["reasoning_effort"] in _reasoning_order else len(_reasoning_order)),
        }
        for _s in scored:
            for _name in ("precision", "recall", "f1"):
                _values = _group[f"{_s}_{_name}"]
                _row[f"{_s}_{_name}"] = _values.mean()
                if spread.value == "runs":
                    # sample SD over runs, undefined for a single run
                    _row[f"{_s}_{_name}_sd"] = _values.std(ddof=1) if len(_values) > 1 else float("nan")
                else:
                    _row[f"{_s}_{_name}_sd"] = _group[f"{_s}_{_name}_sd_docs"].mean()
        # per document; the summed call durations, so parallel requests don't shorten them
        _row["output_tokens_per_doc"] = (_group["completion_tokens"] / _group["documents"]).mean()
        _row["seconds_per_doc"] = (_group["llm_seconds"] / _group["documents"]).mean()
        _rows.append(_row)
    results = pd.DataFrame(_rows)
    if not results.empty:
        results = results.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return results, scored


@app.cell
def _(mo, pd, results, scored, spread):
    mo.stop(results.empty, mo.callout(mo.md("No runs match the selection."), kind="warn"))

    def with_sd(mean, sd):
        if pd.isna(sd):
            return f"{100 * mean:.1f}"
        return f"{100 * mean:.1f} ± {100 * sd:.1f}"

    _labels = {"md": "MD", "re": "RE"}
    _columns = {"Model": results["model"], "Reasoning": results["reasoning"], "Runs": results["runs"]}
    for _s in scored:
        for _name, _short in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
            _columns[f"{_labels[_s]} {_short}"] = [
                with_sd(m, sd) for m, sd in zip(results[f"{_s}_{_name}"], results[f"{_s}_{_name}_sd"])
            ]
    _columns["Output tokens / doc"] = results["output_tokens_per_doc"].round().astype(int)
    _columns["Seconds / doc"] = results["seconds_per_doc"].round(1)
    table = pd.DataFrame(_columns)

    _spread_note = (
        "how much a config's score varies between repeated runs"
        if spread.value == "runs" else
        "how much the score varies between documents, averaged over the runs"
    )
    mo.vstack([
        mo.ui.table(table, selection=None, pagination=False),
        mo.md(f"± shows {_spread_note}. Seconds / doc sums the duration of the document's LLM calls; "
              "they depend on the load on the gateway at the time of the run."),
    ])
    return table, with_sd


@app.cell
def _(mo, per_type, results, scored):
    mo.stop(results.empty or per_type.empty)

    # micro F1 per type, averaged over the runs; one table per scored task
    _tables = {}
    for _s in scored:
        _selected = per_type[(per_type["scored"] == _s) & per_type["config"].isin(results["config"])]
        _f1 = _selected.groupby(["config", "type"])["f1"].mean().unstack("type")
        _f1 = (100 * _f1.reindex(results["config"])).round(1)
        _f1.index = [f"{m} ({r})" for m, r in zip(results["model"], results["reasoning"])]
        _gold = _selected.groupby("type")["num_gold"].first()
        _f1.columns = [f"{t} ({_gold[t]})" for t in _f1.columns]
        _tables[_s] = _f1.reset_index(names="Config")
    _titles = {"md": "Mention F1 per type", "re": "Relation F1 per type"}
    mo.accordion({
        f"{_titles[_s]} (micro, mean over runs; gold count per run in brackets)":
            mo.ui.table(_t, selection=None, pagination=False)
        for _s, _t in _tables.items()
    })
    return


@app.cell
def _(mo, results, scored, spread, task, with_sd):
    mo.stop(results.empty)

    def escape(text):
        for _char in ("\\", "&", "%", "_", "#", "$", "{", "}"):
            text = text.replace(_char, "\\" + _char)
        return text

    _labels = {"md": "MD", "re": "RE"}
    _header = ["Model", "Reasoning"] + [f"{_labels[s]} {n}" for s in scored for n in ("P", "R", "F1")] + ["Tokens/doc"]
    # the best mean F1 of every task in bold
    _best = {s: results[f"{s}_f1"].max() for s in scored}
    _lines = []
    for _, _row in results.iterrows():
        _cells = [escape(_row["model"]), escape(_row["reasoning"])]
        for _s in scored:
            for _name in ("precision", "recall", "f1"):
                _cell = with_sd(_row[f"{_s}_{_name}"], _row[f"{_s}_{_name}_sd"]).replace("±", "$\\pm$")
                if _name == "f1" and _row[f"{_s}_f1"] == _best[_s]:
                    _cell = f"\\textbf{{{_cell}}}"
                _cells.append(_cell)
        _cells.append(f"{_row['output_tokens_per_doc']:,.0f}".replace(",", "{,}"))
        _lines.append(" & ".join(_cells) + " \\\\")
    _task_text = "Mention detection" if task.value == "md" else "End-to-end mention and relation extraction"
    _spread_text = "over runs" if spread.value == "runs" else "over documents, averaged over runs"
    latex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{ll" + "r" * (len(_header) - 2) + "}",
        "\\toprule",
        " & ".join(_header) + " \\\\",
        "\\midrule",
        *_lines,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{_task_text} on PET, macro-averaged over documents, mean $\\pm$ standard deviation "
        f"{_spread_text}, in percent. Tokens/doc: output tokens per document, including reasoning.}}",
        "\\end{table}",
    ])
    _name = "mention-detection" if task.value == "md" else "end-to-end"
    mo.vstack([
        mo.md("## LaTeX\n\nNeeds `\\usepackage{booktabs}`."),
        mo.hstack([
            mo.download(data=latex.encode(), filename=f"{_name}.tex", mimetype="text/x-tex", label="Download .tex"),
            mo.download(data=results.to_csv(index=False).encode(), filename=f"{_name}.csv", mimetype="text/csv",
                        label="Download .csv (numbers as fractions)"),
        ], justify="start"),
        mo.md(f"```latex\n{latex}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
