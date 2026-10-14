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
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from autoannotate import JsonImporter
    from eval import datasets, scoring, test_split
    from experiments import runner

    return JsonImporter, PRED_DIR, datasets, runner, scoring, test_split


@app.cell
def _(JsonImporter, PRED_DIR, datasets, json, pd, runner, scoring, test_split):
    # gold documents, test split and every sentence of each dataset whose files are there (the
    # medical ones are generated, see scripts/uq-pet-token-splits.py); see src/eval/datasets.py
    gold = {}
    for _name, _dataset in datasets.DATASETS.items():
        if _dataset.gold_file.exists() and _dataset.test_file.exists():
            gold[_name] = {
                "documents": datasets.read_documents(_dataset.gold_file),
                "test": test_split.load_test_split(_dataset.test_file),
                "sentences": test_split.load_test_split(_dataset.sentences_file),
            }

    # what the folder holding pet.jsonl says was predicted, and what to score on the gold
    # documents; md-test-sentences runs annotated single test sentences, scored only on the test
    _scored = {
        "end-to-end": ("md", "re"),
        "md": ("md",),
        "re": ("re",),
        "isolated-re": ("re",),
        "md-test-sentences": (),
    }

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
        # run.json records the dataset, how many documents a run was meant to annotate and the
        # tokens it used; predictions from before the configs annotated PET
        _manifest_file = _run_dir / "run.json"
        _manifest = None
        if _manifest_file.exists():
            _manifest = json.loads(_manifest_file.read_text(encoding="utf8"))
        _dataset = runner.manifest_settings(_manifest)["dataset"] if _manifest else "pet"
        if _dataset not in gold:
            continue
        _gold = gold[_dataset]
        _types = datasets.DATASETS[_dataset].types
        _documents = JsonImporter(_predictions).do_import()
        # scored both ways up front, so switching the matching doesn't rescore every run
        _metrics = {"documents": len(_documents)}
        _exact_metrics = None
        if _scored[_task]:
            _metrics = scoring.score(_gold["documents"], _documents, _scored[_task])
            _exact_metrics = scoring.score(
                _gold["documents"], _documents, _scored[_task], exact=True
            )
        _planned = _manifest["documents_planned"] if _manifest else len(_gold["documents"])
        _row = {
            "dataset": _dataset,
            "folder": _run_dir.parent.name if _is_run else _run_dir.name,
            "task": _task,
            "documents": _metrics["documents"],
            # md-test-sentences documents are single sentences, so token costs are per sentence
            "sentences": sum(len(_d.sentences) for _d in _documents),
            "complete": _metrics["documents"] == _planned,
            # a run still in its first session has no totals yet
            "output_tokens": (
                _manifest["totals"]["all"]["completion_tokens"]
                if _manifest and "totals" in _manifest
                else None
            ),
            "reasoning_effort": _manifest["settings"]["reasoning_effort"] if _manifest else None,
            # the model is the folder name before "-md" without the dataset, e.g. "gemma-4-31b"
            # for gemma-4-31b-md-reasoning-low and quishpi-gemma-4-31b-md-reasoning-low
            "model": (_run_dir.parent.name if _is_run else _run_dir.name)
            .removeprefix(f"{_dataset}-")
            .split("-md")[0],
        }
        _effort = _row["reasoning_effort"]
        _row["reasoning"] = {"none": "off"}.get(_effort, _effort or "default")
        # md_f1 etc. with overlapping mentions matching, exact_md_f1 etc. with exact ones
        for _prefix, _m in (("", _metrics), ("exact_", _exact_metrics)):
            for _s in _scored[_task]:
                for _name in ("precision", "recall", "f1"):
                    _row[f"{_prefix}{_s}_{_name}"] = _m[_s]["macro"][_name]["mean"]
                    _row[f"{_prefix}{_s}_{_name}_sd_docs"] = _m[_s]["macro"][_name]["std"]
        # mentions on the test sentences, scored like uq-pet-token's models
        _test_predicted, _ = test_split.predict_test_split(_gold["test"], _documents, _types)
        if _task != "re" and _task != "isolated-re" and _test_predicted is not None:
            _test_scores = test_split.evaluate(_gold["test"], _test_predicted)
            _row["test_precision"] = _test_scores["entity_precision"]
            _row["test_recall"] = _test_scores["entity_recall"]
            _row["test_f1"] = _test_scores["entity_f1"]
            _row["test_macro_f1"] = _test_scores["entity_macro_f1"]
            for _name in ("precision", "recall", "f1", "macro_f1"):
                _row[f"test_partial_{_name}"] = _test_scores[f"entity_partial_{_name}"]
        # the same scoring on all the dataset's sentences, for runs that annotated whole documents
        _all_predicted, _ = test_split.predict_test_split(_gold["sentences"], _documents, _types)
        if _task != "re" and _task != "isolated-re" and _all_predicted is not None:
            _all_scores = test_split.evaluate(_gold["sentences"], _all_predicted)
            _row["all_sentences_f1"] = _all_scores["entity_f1"]
        _rows.append(_row)
    runs = pd.DataFrame(_rows)
    return gold, runs


@app.cell
def _(gold, mo):
    dataset = mo.ui.dropdown(options=list(gold), value="pet", label="Dataset")
    return (dataset,)


@app.cell
def _(dataset, gold, mo):
    _documents = len(gold[dataset.value]["documents"])
    _sentences = len(gold[dataset.value]["test"])
    _all = f"all {_documents} documents"
    evaluation = mo.ui.dropdown(
        options={
            _all: "all",
            f"test split, {_sentences} sentences (exact mentions match, as uq-pet-token)": "test",
        },
        value=_all,
        label="Evaluate on",
    )
    return (evaluation,)


@app.cell
def _(mo):
    # only for scores on all documents: the test split is scored with seqeval, always exact
    matching = mo.ui.radio(
        options={"partial (tokens overlap)": "partial", "exact (same tokens)": "exact"},
        value="partial (tokens overlap)",
        label="Mentions match",
        inline=True,
    )
    return (matching,)


@app.cell
def _(evaluation, mo):
    # test scores are pooled over all test sentences, so there is no spread over documents
    _options = {"over runs": "runs"}
    if evaluation.value == "all":
        _options["over documents"] = "documents"
    spread = mo.ui.dropdown(options=_options, value="over runs", label="± standard deviation")
    return (spread,)


@app.cell
def _(dataset, evaluation, matching, mo, pd, runs, spread):
    mo.stop(runs.empty, mo.md("No predictions in `res/pred/` yet."))
    runs_of_dataset = runs[runs["dataset"] == dataset.value]

    # table column -> runs column
    if evaluation.value == "test":
        _test_f1 = runs_of_dataset.get("test_f1", pd.Series(dtype=float))
        _complete = runs_of_dataset[runs_of_dataset["complete"] & _test_f1.notna()]
        _columns = {
            "md precision": "test_precision",
            "md recall": "test_recall",
            "md f1": "test_f1",
            "md macro f1": "test_macro_f1",
        }
    else:
        _scored_on_documents = runs_of_dataset.get("md_f1", pd.Series(dtype=float)).notna()
        _scored_on_documents |= runs_of_dataset.get("re_f1", pd.Series(dtype=float)).notna()
        _complete = runs_of_dataset[runs_of_dataset["complete"] & _scored_on_documents]
        # relation columns only if some run extracted relations
        _scored = [s for s in ("md", "re") if _complete.get(f"{s}_f1", pd.Series()).notna().any()]
        _prefix = "exact_" if matching.value == "exact" else ""
        _columns = {
            f"{s} {name}": f"{_prefix}{s}_{name}"
            for s in _scored
            for name in ("precision", "recall", "f1")
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
        _tokens = _group["output_tokens"] / _group["sentences"]
        _row["output tokens / sentence"] = round(_tokens.mean()) if _tokens.notna().all() else None
        _rows.append(_row)
    table = pd.DataFrame(
        _rows, columns=["folder", "task", "runs", *_columns, "output tokens / sentence"]
    )
    # predictions from before the configs have no run.json, so no token counts
    table["output tokens / sentence"] = table["output tokens / sentence"].astype("Int64")
    return runs_of_dataset, table


@app.cell
def _(dataset, datasets, evaluation, gold, matching, mo, spread, table):
    _dataset = datasets.DATASETS[dataset.value]
    _gold_file = _dataset.gold_file.relative_to(_dataset.gold_file.parents[2])
    _test_file = _dataset.test_file.relative_to(_dataset.test_file.parents[2])
    _sources = {
        "pet": "`res/gold/pet.jsonl` (the PET release)",
        "quishpi": (
            f"`{_gold_file}` (Quishpi's sentences as cut by uq-pet-token, grouped into documents)"
        ),
        "medical": f"`{_gold_file}` (MedicalProcessInstruks' sentences, grouped into documents)",
    }
    _match = {"partial": "their tokens overlap", "exact": "they cover the same tokens"}
    _notes = {
        "all": f"""
        Scored against {_sources[dataset.value]}, macro-averaged over documents. A
        predicted mention is correct if its type matches a gold mention and
        {_match[matching.value]}, a relation if its type matches and both its head and tail
        match that way; each gold annotation matches once.
        """,
        "test": f"""
        Scored only on uq-pet-token's {len(gold[dataset.value]["test"])} test sentences against
        their tags (`{_test_file}`), with seqeval as uq-pet-token does: a mention is correct
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
        mo.hstack(
            [dataset, evaluation, spread] + ([matching] if evaluation.value == "all" else []),
            justify="start",
            gap=2,
        ),
        mo.ui.table(table, selection=None, pagination=False),
    ])
    return



@app.cell
def _(pd, runs_of_dataset):
    # mention detection configs, one row per model and reasoning effort
    _reasoning_order = ["off", "minimal", "low", "medium", "high"]
    _md = runs_of_dataset[
        runs_of_dataset["complete"] & runs_of_dataset["task"].isin(["md", "md-test-sentences"])
    ]

    def _mean_sd(values):
        if len(values) < 2:
            return f"{100 * values.mean():.1f}"
        return f"{100 * values.mean():.1f} ± {100 * values.std():.1f}"

    def _change(new, old):
        return f"{100 * (new.mean() - old.mean()):+.1f}" if len(new) and len(old) else ""

    # models ordered by their best test F1, the reasoning efforts from off to high
    _best = list(_md.groupby("model")["test_f1"].max().sort_values(ascending=False).index)
    _rows = []
    for (_model, _reasoning), _group in _md.groupby(["model", "reasoning"]):
        _whole = _group[_group["task"] == "md"]
        _all = _whole["all_sentences_f1"].dropna()
        _test = _whole["test_f1"].dropna()
        _sentences = _group[_group["task"] == "md-test-sentences"]["test_f1"].dropna()
        _rows.append({
            "model": _model,
            "reasoning": _reasoning,
            "all sentences": _mean_sd(_all) if len(_all) else "",
            "test: whole documents": _mean_sd(_test) if len(_test) else "",
            "test: single sentences": _mean_sd(_sentences) if len(_sentences) else "",
            "change": _change(_sentences, _test),
        })
    _columns = [
        "model", "reasoning", "all sentences", "test: whole documents", "test: single sentences",
        "change",
    ]
    test_table = pd.DataFrame(sorted(_rows, key=lambda r: (
        _best.index(r["model"]),
        _reasoning_order.index(r["reasoning"]) if r["reasoning"] in _reasoning_order else 99,
    )), columns=_columns)
    return (test_table,)


@app.cell
def _(dataset, gold, mo, test_table):
    _sentences = len(gold[dataset.value]["sentences"])
    _test = len(gold[dataset.value]["test"])
    mo.vstack([
        mo.md(f"""
        ## Test split: whole documents vs single sentences

        Mention F1 in percent on {dataset.value} (BIO tags as uq-pet-token's, exact spans, micro),
        mean ± standard deviation over runs. All sentences: the LLM annotated whole documents,
        scored on all {_sentences} sentences of the dataset. Test: scored only on uq-pet-token's
        {_test} test sentences, either from the whole-document runs or from runs where the LLM got
        each test sentence on its own, like uq-pet-token's models. Change: single sentences minus
        whole documents on the test split, in points.
        """),
        mo.ui.table(test_table, selection=None, pagination=False),
    ])
    return


@app.cell
def _(json, mo, pd):
    # uq-pet-token's fully supervised baseline (its README, "Supervised upper bound"): each model is
    # tuned on validation, trained on all training sentences and scored once on the test split
    # per model seed. results/supervised/<sweep>/test/pool_100pct/<run>/results.csv has a row per
    # seed, with the metric names test_split.evaluate copies
    UQ_PET_TOKEN_DIR = mo.notebook_dir().parent.parent / "uq-pet-token"
    _results = []
    for _config_file in sorted(UQ_PET_TOKEN_DIR.glob("results/supervised/*/run_config.json")):
        _sweep = _config_file.parent
        _config = json.loads(_config_file.read_text(encoding="utf8"))
        _summary = json.loads((_sweep / "summary.json").read_text(encoding="utf8"))
        for _test_run in _summary.get("test_runs", []):
            if _test_run["slot"] != "test/pool_100pct" or _test_run["status"] != "complete":
                continue
            _seeds = pd.read_csv(_sweep / _test_run["run_dir"] / "results.csv")
            _seeds = _seeds[(_seeds["arm"] == "supervised") & (_seeds["sentence_percent"] == 100)]
            _results.append(_seeds.assign(
                dataset=_config["dataset"], model=_config["checkpoint"].split("/")[-1]
            ))
    supervised = pd.concat(_results, ignore_index=True) if _results else pd.DataFrame()
    return UQ_PET_TOKEN_DIR, supervised


@app.cell
def _(mo):
    # uq-pet-token tunes on exact match and reports partial match beside it (its ADR 0010)
    comparison_matching = mo.ui.radio(
        options={"exact (same span)": "exact", "partial (tokens overlap)": "partial"},
        value="exact (same span)",
        label="Mentions match",
        inline=True,
    )
    return (comparison_matching,)


@app.cell
def _(comparison_matching, dataset, pd, runs_of_dataset, supervised):
    # uq-pet-token's supervised models and the LLM runs on the same test sentences, scored the same
    _metrics = {"precision": "P", "recall": "R", "f1": "F1", "macro_f1": "macro F1"}
    _partial = "partial_" if comparison_matching.value == "partial" else ""

    def _mean_sd(values):
        if values.isna().any():
            # runs saved before a metric was added
            return ""
        if len(values) < 2:
            return f"{100 * values.mean():.1f}"
        return f"{100 * values.mean():.1f} ± {100 * values.std():.1f}"

    def _scores(group, column):
        return {
            label: _mean_sd(group[column.format(name)])
            if column.format(name) in group
            else ""
            for name, label in _metrics.items()
        }

    _rows = []
    _supervised = supervised[supervised["dataset"] == dataset.value] if len(supervised) else supervised
    for _model, _group in (_supervised.groupby("model") if len(_supervised) else []):
        _rows.append({
            "approach": "fine-tuned (uq-pet-token)",
            "model": _model,
            "reasoning": "",
            "input": "single sentences",
            "training sentences": int(_group["n_sentences"].iloc[0]),
            "runs": len(_group),
            **_scores(_group, f"entity_{_partial}{{}}"),
            "_f1": _group[f"entity_{_partial}f1"].mean(),
        })
    _test_f1 = runs_of_dataset.get(f"test_{_partial}f1", pd.Series(dtype=float))
    _llm = runs_of_dataset[runs_of_dataset["complete"] & _test_f1.notna()]
    for (_folder, _task), _group in _llm.groupby(["folder", "task"]):
        _rows.append({
            "approach": "LLM (prompted)",
            "model": _group["model"].iloc[0],
            "reasoning": _group["reasoning"].iloc[0],
            "input": "single sentences" if _task == "md-test-sentences" else "whole documents",
            "training sentences": 0,
            "runs": len(_group),
            **_scores(_group, f"test_{_partial}{{}}"),
            "_f1": _group[f"test_{_partial}f1"].mean(),
        })
    # fine-tuned models first, each approach by F1
    comparison_table = (
        pd.DataFrame(
            _rows,
            columns=[
                "approach", "model", "reasoning", "input", "training sentences", "runs",
                *_metrics.values(), "_f1",
            ],
        )
        .sort_values(
            ["approach", "_f1"],
            ascending=False,
            key=lambda c: c.str.startswith("fine-tuned") if c.name == "approach" else c,
        )
        .drop(columns="_f1")
        .reset_index(drop=True)
    )
    return (comparison_table,)


@app.cell
def _(
    UQ_PET_TOKEN_DIR, comparison_matching, comparison_table, dataset, gold, mo, supervised
):
    _test = len(gold[dataset.value]["test"])
    _missing = ""
    if not len(supervised) or not (supervised["dataset"] == dataset.value).any():
        _missing = (
            f"**No supervised results of uq-pet-token for {dataset.value} in "
            f"`{UQ_PET_TOKEN_DIR}/results/supervised/`.**"
        )
    _match = {
        "exact": "only if its type and both span boundaries match (seqeval)",
        "partial": (
            "if its type matches and it shares a token with a gold mention, each gold mention "
            "matched once (uq-pet-token's `partial_entity_scores`)"
        ),
    }
    mo.vstack([
        mo.md(f"""
        ## Comparison with uq-pet-token

        Mention detection on uq-pet-token's {_test} test sentences of {dataset.value}, both
        projects scored the same way on the same BIO tags: a predicted mention is correct
        {_match[comparison_matching.value]}. P, R and F1 are micro over all mentions, macro F1
        averages the F1 of the mention types; in percent, mean ± standard deviation over model
        seeds (fine-tuned) or runs (LLMs). Fine-tuned: uq-pet-token's fully supervised baseline,
        tuned on its validation holdout and trained on all its training sentences. LLMs: no
        training beyond the prompt's format example; given whole documents (only their test
        sentences scored) or each test sentence on its own like the fine-tuned models.
        {_missing}
        """),
        comparison_matching,
        mo.ui.table(comparison_table, selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
