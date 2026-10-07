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
    GOLD_FILE = REPO_DIR / "res" / "gold" / "pet.jsonl"
    PRED_DIR = REPO_DIR / "res" / "pred"
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from autoannotate import JsonImporter
    from eval import scoring

    return GOLD_FILE, JsonImporter, PRED_DIR, scoring


@app.cell
def _(GOLD_FILE, JsonImporter, PRED_DIR, json, pd, scoring):
    # what the folder holding pet.jsonl says was predicted
    _scored = {"end-to-end": ("md", "re"), "md": ("md",), "re": ("re",), "isolated-re": ("re",)}

    _gold = JsonImporter(GOLD_FILE).do_import()
    _rows = []
    # runs of a config: <folder>/run-<i>/<task>/pet.jsonl;
    # predictions from before the configs: <folder>/<task>/pet.jsonl
    _files = sorted(PRED_DIR.glob("*/run-*/*/pet.jsonl")) + sorted(PRED_DIR.glob("*/*/pet.jsonl"))
    for _predictions in _files:
        _task = _predictions.parent.name
        if _task not in _scored:
            continue
        _run_dir = _predictions.parent.parent
        _is_run = _run_dir.name.startswith("run-")
        _metrics = scoring.score(_gold, JsonImporter(_predictions).do_import(), _scored[_task])
        # run.json records how many documents a run was meant to annotate and the tokens it used
        _manifest_file = _run_dir / "run.json"
        _manifest = None
        if _manifest_file.exists():
            _manifest = json.loads(_manifest_file.read_text(encoding="utf8"))
        _planned = _manifest["documents_planned"] if _manifest else len(_gold)
        _row = {
            "folder": _run_dir.parent.name if _is_run else _run_dir.name,
            "task": _task,
            "documents": _metrics["documents"],
            "complete": _metrics["documents"] == _planned,
            "output_tokens": _manifest["totals"]["all"]["completion_tokens"] if _manifest else None,
        }
        for _s in _scored[_task]:
            for _name in ("precision", "recall", "f1"):
                _row[f"{_s}_{_name}"] = _metrics[_s]["macro"][_name]["mean"]
                _row[f"{_s}_{_name}_sd_docs"] = _metrics[_s]["macro"][_name]["std"]
        _rows.append(_row)
    runs = pd.DataFrame(_rows)
    return (runs,)


@app.cell
def _(mo):
    spread = mo.ui.dropdown(
        options={"over runs": "runs", "over documents": "documents"},
        value="over runs",
        label="± standard deviation",
    )
    return (spread,)


@app.cell
def _(mo, pd, runs, spread):
    mo.stop(runs.empty, mo.md("No predictions in `res/pred/` yet."))
    _complete = runs[runs["complete"]]

    def _mean_sd(group, column):
        mean = group[column].mean()
        if spread.value == "documents":
            # SD over the documents of each run, averaged over the runs
            sd = group[f"{column}_sd_docs"].mean()
        elif len(group) > 1:
            sd = group[column].std()
        else:
            return f"{100 * mean:.1f}"
        return f"{100 * mean:.1f} ± {100 * sd:.1f}"

    # relation columns only if some run extracted relations
    _scored = [s for s in ("md", "re") if _complete.get(f"{s}_f1", pd.Series()).notna().any()]
    _rows = []
    for (_folder, _task), _group in _complete.groupby(["folder", "task"]):
        _row = {"folder": _folder, "task": _task, "runs": len(_group)}
        for _s in _scored:
            for _name in ("precision", "recall", "f1"):
                _values_known = _group[f"{_s}_{_name}"].notna().all()
                _row[f"{_s} {_name}"] = _mean_sd(_group, f"{_s}_{_name}") if _values_known else ""
        _tokens = _group["output_tokens"] / _group["documents"]
        _row["output tokens / doc"] = round(_tokens.mean()) if _tokens.notna().all() else None
        _rows.append(_row)
    table = pd.DataFrame(_rows)
    # predictions from before the configs have no run.json, so no token counts
    table["output tokens / doc"] = table["output tokens / doc"].astype("Int64")
    return (table,)


@app.cell
def _(mo, spread, table):
    mo.vstack([
        mo.md("""
        # Evaluation results

        Scores every `pet.jsonl` in `res/pred/` against the gold annotations, one row per folder
        and task, over its runs with all documents. Scores in percent, macro-averaged over
        documents, as mean ± standard deviation over runs or over documents. A predicted mention
        is correct if its type matches a gold mention and their tokens overlap, a relation if its
        type matches and both its head and tail overlap; each gold annotation matches once.
        """),
        spread,
        mo.ui.table(table, selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
