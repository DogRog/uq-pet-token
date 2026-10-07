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
    # uq-pet-token's PET test split, see src/eval/test_split.py
    TEST_FILE = REPO_DIR / "res" / "gold" / "pet-v1.1-test.jsonl"
    PRED_DIR = REPO_DIR / "res" / "pred"
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from autoannotate import JsonImporter
    from eval import scoring, test_split

    return GOLD_FILE, JsonImporter, PRED_DIR, TEST_FILE, scoring, test_split


@app.cell
def _(GOLD_FILE, JsonImporter, PRED_DIR, TEST_FILE, json, pd, scoring, test_split):
    # what the folder holding pet.jsonl says was predicted, and what to score on the gold
    # documents; md-test-sentences runs annotated single test sentences, scored only on the test
    _scored = {
        "end-to-end": ("md", "re"),
        "md": ("md",),
        "re": ("re",),
        "isolated-re": ("re",),
        "md-test-sentences": (),
    }

    _gold = JsonImporter(GOLD_FILE).do_import()
    _test = test_split.load_test_split(TEST_FILE)
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
        _documents = JsonImporter(_predictions).do_import()
        _metrics = {"documents": len(_documents)}
        if _scored[_task]:
            _metrics = scoring.score(_gold, _documents, _scored[_task])
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
        # mentions on the test sentences, scored like uq-pet-token's models
        _test_predicted, _ = test_split.predict_test_split(_test, _documents)
        if _task != "re" and _task != "isolated-re" and _test_predicted is not None:
            _test_scores = test_split.evaluate(_test, _test_predicted)
            _row["test_precision"] = _test_scores["entity_precision"]
            _row["test_recall"] = _test_scores["entity_recall"]
            _row["test_f1"] = _test_scores["entity_f1"]
            _row["test_macro_f1"] = _test_scores["entity_macro_f1"]
        _rows.append(_row)
    runs = pd.DataFrame(_rows)
    return (runs,)


@app.cell
def _(mo):
    evaluation = mo.ui.dropdown(
        options={
            "all 45 documents (overlapping mentions match)": "all",
            "test split, 84 sentences (exact mentions match, as uq-pet-token)": "test",
        },
        value="all 45 documents (overlapping mentions match)",
        label="Evaluate on",
    )
    return (evaluation,)


@app.cell
def _(evaluation, mo):
    # test scores are pooled over all test sentences, so there is no spread over documents
    _options = {"over runs": "runs"}
    if evaluation.value == "all":
        _options["over documents"] = "documents"
    spread = mo.ui.dropdown(options=_options, value="over runs", label="± standard deviation")
    return (spread,)


@app.cell
def _(evaluation, mo, pd, runs, spread):
    mo.stop(runs.empty, mo.md("No predictions in `res/pred/` yet."))

    # table column -> runs column
    if evaluation.value == "test":
        _complete = runs[runs["complete"] & runs.get("test_f1", pd.Series(dtype=float)).notna()]
        _columns = {
            "md precision": "test_precision",
            "md recall": "test_recall",
            "md f1": "test_f1",
            "md macro f1": "test_macro_f1",
        }
    else:
        _scored_on_documents = runs.get("md_f1", pd.Series(dtype=float)).notna()
        _scored_on_documents |= runs.get("re_f1", pd.Series(dtype=float)).notna()
        _complete = runs[runs["complete"] & _scored_on_documents]
        # relation columns only if some run extracted relations
        _scored = [s for s in ("md", "re") if _complete.get(f"{s}_f1", pd.Series()).notna().any()]
        _columns = {
            f"{s} {name}": f"{s}_{name}" for s in _scored for name in ("precision", "recall", "f1")
        }

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

    _rows = []
    for (_folder, _task), _group in _complete.groupby(["folder", "task"]):
        _row = {"folder": _folder, "task": _task, "runs": len(_group)}
        for _label, _column in _columns.items():
            _known = _group[_column].notna().all()
            _row[_label] = _mean_sd(_group, _column) if _known else ""
        _tokens = _group["output_tokens"] / _group["documents"]
        _row["output tokens / doc"] = round(_tokens.mean()) if _tokens.notna().all() else None
        _rows.append(_row)
    table = pd.DataFrame(_rows)
    # predictions from before the configs have no run.json, so no token counts
    table["output tokens / doc"] = table["output tokens / doc"].astype("Int64")
    return (table,)


@app.cell
def _(evaluation, mo, spread, table):
    _notes = {
        "all": """
        Scored against `res/gold/pet.jsonl` (the PET release), macro-averaged over documents. A
        predicted mention is correct if its type matches a gold mention and their tokens overlap,
        a relation if its type matches and both its head and tail overlap; each gold annotation
        matches once.
        """,
        "test": """
        Scored only on uq-pet-token's 84 test sentences against their PET v1.1 tags
        (`res/gold/pet-v1.1-test.jsonl`), with seqeval as uq-pet-token does: a mention is correct
        only if its type and both span boundaries match. Precision, recall and F1 are micro over
        all mentions; macro F1 averages the F1 of the mention types. Overlapping predicted
        mentions keep the one listed first. Task `md-test-sentences`: the LLM got each test
        sentence on its own, like uq-pet-token's models; other tasks annotated whole documents.
        """,
    }
    mo.vstack([
        mo.md(f"""
        # Evaluation results

        One row per folder in `res/pred/` and task, over its runs with all documents, in percent,
        as mean ± standard deviation over runs or over documents.
        {_notes[evaluation.value]}
        """),
        mo.hstack([evaluation, spread], justify="start", gap=2),
        mo.ui.table(table, selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
