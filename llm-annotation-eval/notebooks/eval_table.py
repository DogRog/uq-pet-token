import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # LLM annotation results

    Every run in `res/pred/`, rescored from its predictions, next to uq-pet-token's fine-tuned
    models on the same test sentences. Each chapter answers one question and is listed in the
    outline. Scores are in percent, mean ± standard deviation.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - The floating panel picks the **dataset** every chapter shows and the **entity matching**
      every score uses; **Reload results** rescores the runs on disk.
    - **Exact** match counts a predicted mention only if its type and both span boundaries
      match a gold mention; it is uq-pet-token's tuning objective and headline (its ADR 0010).
      **Partial** match needs the same type and one shared token, each gold mention matched
      once; uq-pet-token reports it only.
    - **Test split** scores are uq-pet-token's: its test sentences as BIO tags, scored by a
      copy of its `evaluate_predictions` (seqeval for exact match). P, R and F1 are micro over
      all mentions; macro F1 averages the F1 of the mention types.
    - **Tasks**: `md-test-sentences` sends each test sentence on its own, like uq-pet-token's
      models see them; `md` sends whole documents.
    - Only runs that annotated every planned document count; chapter 1 lists the others.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import json
    import sys

    import marimo as mo
    import pandas as pd
    from wigglystuff import FloatingPanel

    REPO_DIR = mo.notebook_dir().parent
    PRED_DIR = REPO_DIR / "res" / "pred"
    # this project is the llm-annotation-eval/ folder of uq-pet-token
    UQ_PET_TOKEN_DIR = REPO_DIR.parent
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from autoannotate import JsonImporter
    from eval import datasets, scoring, test_split
    from experiments import runner

    DATASET_NAMES = {"pet": "PET", "quishpi": "Quishpi", "medical": "MedicalProcessInstruks"}

    def mean_sd(values, sd=None):
        """Mean ± SD in percent, the SD over the values unless given; empty if one is missing."""
        if values.empty or values.isna().any():
            return ""
        if sd is None:
            if len(values) < 2:
                return f"{100 * values.mean():.1f}"
            sd = values.std()
        return f"{100 * values.mean():.1f} ± {100 * sd:.1f}"

    return (
        DATASET_NAMES,
        FloatingPanel,
        JsonImporter,
        PRED_DIR,
        UQ_PET_TOKEN_DIR,
        datasets,
        json,
        mean_sd,
        mo,
        pd,
        runner,
        scoring,
        test_split,
    )


@app.cell
def _(DATASET_NAMES, FloatingPanel, datasets, mo):
    # the medical files are generated, see scripts/uq-pet-token-splits.py
    _available = [
        name
        for name, d in datasets.DATASETS.items()
        if d.gold_file.exists() and d.test_file.exists()
    ]
    dataset = mo.ui.dropdown(
        {DATASET_NAMES.get(name, name): name for name in _available},
        value=DATASET_NAMES["pet"],
        label="Dataset",
    )
    matching = mo.ui.dropdown(
        {"exact": "exact", "partial": "partial"}, value="exact", label="Entity matching"
    )
    refresh = mo.ui.run_button(label="Reload results")
    FloatingPanel(mo.vstack([dataset, matching, refresh]))
    return dataset, matching, refresh


@app.cell
def _(
    JsonImporter,
    PRED_DIR,
    datasets,
    json,
    pd,
    refresh,
    runner,
    scoring,
    test_split,
):
    refresh.value
    # gold documents, test split and every sentence of each dataset whose files are there
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
    _scored = {"md": ("md",), "md-test-sentences": ()}

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
            _metrics = scoring.score(_gold["documents"], _documents)
            _exact_metrics = scoring.score(_gold["documents"], _documents, exact=True)
        _planned = _manifest["documents_planned"] if _manifest else len(_gold["documents"])
        _row = {
            "dataset": _dataset,
            "folder": _run_dir.parent.name if _is_run else _run_dir.name,
            "run": _run_dir.name if _is_run else "",
            "task": _task,
            "documents": _metrics["documents"],
            "planned": _planned,
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
        if _test_predicted is not None:
            _test_scores = test_split.evaluate(_gold["test"], _test_predicted)
            for _name in ("precision", "recall", "f1", "macro_f1"):
                _row[f"test_{_name}"] = _test_scores[f"entity_{_name}"]
                _row[f"test_partial_{_name}"] = _test_scores[f"entity_partial_{_name}"]
        # the same scoring on all the dataset's sentences, for runs that annotated whole documents
        _all_predicted, _ = test_split.predict_test_split(_gold["sentences"], _documents, _types)
        if _all_predicted is not None:
            _all_scores = test_split.evaluate(_gold["sentences"], _all_predicted)
            _row["all_sentences_f1"] = _all_scores["entity_f1"]
            _row["all_sentences_partial_f1"] = _all_scores["entity_partial_f1"]
        _rows.append(_row)
    runs = pd.DataFrame(_rows)
    return gold, runs


@app.cell
def _(UQ_PET_TOKEN_DIR, json, pd, refresh):
    refresh.value
    # uq-pet-token's fully supervised baseline (its README, "Supervised upper bound"): each model is
    # tuned on validation, trained on all training sentences and scored once on the test split
    # per model seed. results/supervised/<sweep>/test/pool_100pct/<run>/results.csv has a row per
    # seed, with the metric names test_split.evaluate copies
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
            _results.append(
                _seeds.assign(
                    dataset=_config["dataset"], model=_config["checkpoint"].split("/")[-1]
                )
            )
    supervised = pd.concat(_results, ignore_index=True) if _results else pd.DataFrame()
    return (supervised,)


@app.cell
def _(dataset, gold, matching, runs):
    complete_runs = (
        runs[(runs["dataset"] == dataset.value) & runs["complete"]] if len(runs) else runs
    )
    # test_f1 -> test_partial_f1, all_sentences_f1 -> all_sentences_partial_f1
    partial = "partial_" if matching.value == "partial" else ""
    dataset_name = dataset.selected_key
    test_sentences = len(gold[dataset.value]["test"])
    return complete_runs, dataset_name, partial, test_sentences


@app.cell(hide_code=True)
def _(DATASET_NAMES, mo, runs):
    mo.stop(
        runs.empty,
        mo.md("## 1. Where the runs stand\nNo predictions in `res/pred/` yet."),
    )
    # complete runs per task and dataset, with the unfinished ones after them
    _progress = (
        runs.assign(
            finished=runs["complete"].astype(int),
            unfinished=(~runs["complete"]).astype(int),
            dataset=runs["dataset"].map(lambda d: DATASET_NAMES.get(d, d)),
        )
        .groupby(["task", "dataset"])[["finished", "unfinished"]]
        .sum()
        .reset_index()
    )
    _progress["runs"] = _progress["finished"].astype(str) + _progress["unfinished"].map(
        lambda n: f" (+{n} unfinished)" if n else ""
    )
    _progress = _progress.pivot(index="task", columns="dataset", values="runs").fillna("")
    _order = [n for n in DATASET_NAMES.values() if n in _progress.columns]
    _progress = _progress[_order].reset_index()
    _open = runs[~runs["complete"]][["dataset", "folder", "run", "task", "documents", "planned"]]
    mo.vstack(
        [
            mo.md(
                "## 1. Where the runs stand\n"
                "Complete runs per task and dataset, across every folder in `res/pred/`. The "
                "other chapters use complete runs only; **Reload results** in the floating panel "
                "rescores new ones."
            ),
            mo.hstack(
                [
                    mo.stat(
                        f"{runs['complete'].sum()}",
                        label="Complete runs",
                        caption=f"{runs[runs['complete']]['folder'].nunique()} configurations",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{(~runs['complete']).sum()}",
                        label="Unfinished runs",
                        caption="resume them from run_experiments.py",
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1,
            ),
            mo.ui.table(_progress, selection=None, label="Complete runs"),
            mo.accordion({"Unfinished runs": mo.ui.table(_open, selection=None)})
            if len(_open)
            else mo.md(""),
        ]
    )
    return


@app.cell
def _(complete_runs, dataset, mean_sd, partial, pd, supervised):
    # uq-pet-token's supervised models and the LLM runs on the same test sentences, scored the same
    _metrics = {"precision": "P", "recall": "R", "f1": "F1", "macro_f1": "macro F1"}

    def _scores(group, column):
        return {
            label: mean_sd(group[column.format(name)]) if column.format(name) in group else ""
            for name, label in _metrics.items()
        }

    _rows = []
    _supervised = (
        supervised[supervised["dataset"] == dataset.value] if len(supervised) else supervised
    )
    for _model, _group in _supervised.groupby("model") if len(_supervised) else []:
        _rows.append(
            {
                "approach": "fine-tuned (uq-pet-token)",
                "model": _model,
                "reasoning": "",
                "input": "single sentences",
                "training sentences": int(_group["n_sentences"].iloc[0]),
                "runs": len(_group),
                **_scores(_group, f"entity_{partial}{{}}"),
                "_f1": _group[f"entity_{partial}f1"].mean(),
            }
        )
    _test_f1 = complete_runs.get(
        f"test_{partial}f1", pd.Series(index=complete_runs.index, dtype=float)
    )
    _llm = complete_runs[_test_f1.notna()] if len(complete_runs) else complete_runs
    for (_folder, _task), _group in _llm.groupby(["folder", "task"]) if len(_llm) else []:
        _rows.append(
            {
                "approach": "LLM (prompted)",
                "model": _group["model"].iloc[0],
                "reasoning": _group["reasoning"].iloc[0],
                "input": "single sentences" if _task == "md-test-sentences" else "whole documents",
                "training sentences": 0,
                "runs": len(_group),
                **_scores(_group, f"test_{partial}{{}}"),
                "_f1": _group[f"test_{partial}f1"].mean(),
            }
        )
    # fine-tuned models first, each approach by F1
    _table = pd.DataFrame(
        _rows,
        columns=[
            "approach",
            "model",
            "reasoning",
            "input",
            "training sentences",
            "runs",
            *_metrics.values(),
            "_f1",
        ],
    ).sort_values(
        ["approach", "_f1"],
        ascending=False,
        key=lambda c: c.str.startswith("fine-tuned") if c.name == "approach" else c,
    )
    _fine_tuned = _table[_table["approach"].str.startswith("fine-tuned")]
    _prompted = _table[_table["approach"].str.startswith("LLM")]
    # the best of each approach, as a row of the table, or None
    comparison_best = {
        "fine-tuned": _fine_tuned.iloc[0] if len(_fine_tuned) else None,
        "LLM": _prompted.iloc[0] if len(_prompted) else None,
    }
    comparison_table = _table.drop(columns="_f1").reset_index(drop=True)
    return comparison_best, comparison_table


@app.cell(hide_code=True)
def _(
    UQ_PET_TOKEN_DIR,
    comparison_best,
    comparison_table,
    dataset,
    dataset_name,
    matching,
    mo,
    supervised,
    test_sentences,
):
    _tuned, _llm = comparison_best["fine-tuned"], comparison_best["LLM"]
    _stats = [
        mo.stat(
            f"{100 * row['_f1']:.1f}",
            label=f"Best {approach} F1",
            caption=f"{row['model']}"
            + (f", reasoning {row['reasoning']}" if row["reasoning"] else ""),
            bordered=True,
        )
        for approach, row in (("fine-tuned", _tuned), ("LLM", _llm))
        if row is not None
    ]
    if _tuned is not None and _llm is not None:
        _stats.append(
            mo.stat(
                f"{100 * (_llm['_f1'] - _tuned['_f1']):+.1f} pp",
                label="Best LLM − best fine-tuned",
                caption="F1 on the test split",
                bordered=True,
            )
        )
    _no_supervised = not len(supervised) or not (supervised["dataset"] == dataset.value).any()
    mo.vstack(
        [
            mo.md(
                "## 2. Against uq-pet-token's fine-tuned models\n"
                f"Mention detection on uq-pet-token's {test_sentences} test sentences of "
                f"{dataset_name}, {matching.value} match, both projects scored the same way on "
                "the same BIO tags: fine-tuned models trained on every training sentence, and "
                "LLMs prompted with five labelled sentences."
            ),
            mo.hstack(_stats, justify="start", gap=1, wrap=True),
            mo.callout(
                mo.md(
                    f"No supervised results of uq-pet-token for {dataset_name} in "
                    f"`{UQ_PET_TOKEN_DIR}/results/supervised/`."
                ),
                kind="warn",
            )
            if _no_supervised
            else mo.md(""),
            mo.ui.table(comparison_table, selection=None, pagination=False),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Fine-tuned**: uq-pet-token's fully supervised baseline, tuned on its validation
      holdout, trained on all its training sentences and scored once per model seed.
    - **LLM**: no training beyond the prompt's format example, uq-pet-token's 5 bootstrap
      sentences; given whole documents (only their test sentences scored) or each test
      sentence on its own like the fine-tuned models.
    - In percent, mean ± SD over model seeds (fine-tuned) or runs (LLMs). An empty cell is
      a metric uq-pet-token had not saved yet.
    - The best LLM is picked on the test split itself, which favours the LLM.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(dataset, gold, mo):
    _documents = len(gold[dataset.value]["documents"])
    _sentences = len(gold[dataset.value]["test"])
    # the test split first: every run covers it, while single-sentence runs have no documents
    _test = f"test split, {_sentences} sentences"
    evaluation = mo.ui.dropdown(
        options={_test: "test", f"all {_documents} documents": "all"},
        value=_test,
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
def _(complete_runs, evaluation, matching, mean_sd, partial, pd, spread):
    # a score no run has, aligned with the runs so it can filter them
    _missing = pd.Series(index=complete_runs.index, dtype=float)
    # table column -> runs column
    if evaluation.value == "test":
        _test_f1 = complete_runs.get(f"test_{partial}f1", _missing)
        _complete = complete_runs[_test_f1.notna()] if len(complete_runs) else complete_runs
        _columns = {
            "md precision": f"test_{partial}precision",
            "md recall": f"test_{partial}recall",
            "md f1": f"test_{partial}f1",
            "md macro f1": f"test_{partial}macro_f1",
        }
    else:
        _scored_on_documents = complete_runs.get("md_f1", _missing).notna()
        _complete = complete_runs[_scored_on_documents] if len(complete_runs) else complete_runs
        # the lenient score of the gold documents is overlap, not uq-pet-token's partial match
        _prefix = "exact_" if matching.value == "exact" else ""
        _columns = {f"md {name}": f"{_prefix}md_{name}" for name in ("precision", "recall", "f1")}

    def _mean_sd(group, column):
        if spread.value == "documents":
            # SD over the documents of each run, averaged over the runs
            return mean_sd(group[column], group[f"{column}_sd_docs"].mean())
        return mean_sd(group[column])

    _rows = []
    for (_folder, _task), _group in _complete.groupby(["folder", "task"]) if len(_complete) else []:
        _row = {"folder": _folder, "task": _task, "runs": len(_group)}
        for _label, _column in _columns.items():
            _row[_label] = _mean_sd(_group, _column)
        _tokens = _group["output_tokens"] / _group["sentences"]
        _row["output tokens / sentence"] = round(_tokens.mean()) if _tokens.notna().all() else None
        _rows.append(_row)
    configuration_table = pd.DataFrame(
        _rows, columns=["folder", "task", "runs", *_columns, "output tokens / sentence"]
    )
    # predictions from before the configs have no run.json, so no token counts
    configuration_table["output tokens / sentence"] = configuration_table[
        "output tokens / sentence"
    ].astype("Int64")
    return (configuration_table,)


@app.cell(hide_code=True)
def _(
    configuration_table,
    dataset,
    dataset_name,
    datasets,
    evaluation,
    matching,
    mo,
    spread,
    test_sentences,
):
    _dataset = datasets.DATASETS[dataset.value]
    _gold_file = _dataset.gold_file.relative_to(_dataset.gold_file.parents[2])
    _test_file = _dataset.test_file.relative_to(_dataset.test_file.parents[2])
    _sources = {
        "pet": f"`{_gold_file}` (PET v1.1's sentences, grouped into documents)",
        "quishpi": (
            f"`{_gold_file}` (Quishpi's sentences as cut by uq-pet-token, grouped into documents)"
        ),
        "medical": f"`{_gold_file}` (MedicalProcessInstruks' sentences, grouped into documents)",
    }
    _match = {"partial": "their tokens overlap", "exact": "they cover the same tokens"}
    _notes = {
        "all": f"""
    - Scored against {_sources[dataset.value]}, macro-averaged over documents.
    - A predicted mention is correct if its type matches a gold mention and
      {_match[matching.value]}. Each gold mention matches once.
    - Partial match here is this project's overlap match, not uq-pet-token's partial match;
      the test split uses uq-pet-token's.
    - Runs that only annotated test sentences (`md-test-sentences`) have no score here.
    """,
        "test": f"""
    - Scored only on uq-pet-token's {test_sentences} test sentences against their tags
      (`{_test_file}`), as uq-pet-token scores its models.
    - Precision, recall and F1 are micro over all mentions; macro F1 averages the F1 of the
      mention types. Overlapping predicted mentions keep the one listed first.
    - Test scores pool all test sentences, so there is no spread over documents.
    """,
    }
    mo.vstack(
        [
            mo.md(
                "## 3. Every configuration\n"
                f"One row per folder in `res/pred/` and task on {dataset_name}, over its complete "
                f"runs, {matching.value} match, in percent as mean ± SD {spread.selected_key}. "
                "Output tokens are per annotated sentence."
            ),
            mo.hstack([evaluation, spread], justify="start", gap=2),
            mo.ui.table(configuration_table, selection=None, pagination=False),
            mo.accordion({"How to read this": mo.md(_notes[evaluation.value])}),
        ]
    )
    return


@app.cell
def _(complete_runs, mean_sd, partial, pd):
    # mention detection configs, one row per model and reasoning effort
    _reasoning_order = ["off", "minimal", "low", "medium", "high"]
    _md = (
        complete_runs[complete_runs["task"].isin(["md", "md-test-sentences"])]
        if len(complete_runs)
        else complete_runs
    )

    def _change(new, old):
        return f"{100 * (new.mean() - old.mean()):+.1f}" if len(new) and len(old) else ""

    # models ordered by their best test F1, the reasoning efforts from off to high
    _best = (
        list(_md.groupby("model")[f"test_{partial}f1"].max().sort_values(ascending=False).index)
        if len(_md)
        else []
    )
    _rows = []
    for (_model, _reasoning), _group in _md.groupby(["model", "reasoning"]) if len(_md) else []:
        _whole = _group[_group["task"] == "md"]
        _all = _whole.get(f"all_sentences_{partial}f1", pd.Series(dtype=float)).dropna()
        _test = _whole[f"test_{partial}f1"].dropna()
        _sentences = _group[_group["task"] == "md-test-sentences"][f"test_{partial}f1"].dropna()
        _rows.append(
            {
                "model": _model,
                "reasoning": _reasoning,
                "all sentences": mean_sd(_all),
                "test: whole documents": mean_sd(_test),
                "test: single sentences": mean_sd(_sentences),
                "change": _change(_sentences, _test),
            }
        )
    input_table = pd.DataFrame(
        sorted(
            _rows,
            key=lambda r: (
                _best.index(r["model"]),
                _reasoning_order.index(r["reasoning"])
                if r["reasoning"] in _reasoning_order
                else 99,
            ),
        ),
        columns=[
            "model",
            "reasoning",
            "all sentences",
            "test: whole documents",
            "test: single sentences",
            "change",
        ],
    )
    return (input_table,)


@app.cell(hide_code=True)
def _(dataset, dataset_name, gold, input_table, matching, mo, test_sentences):
    _sentences = len(gold[dataset.value]["sentences"])
    mo.vstack(
        [
            mo.md(
                "## 4. Whole documents or single sentences?\n"
                f"Mention F1 on {dataset_name}, {matching.value} match, with uq-pet-token's "
                "scoring: does an LLM do better given the whole document or, like the "
                "fine-tuned models, one sentence at a time? **Change** is single sentences "
                "minus whole documents on the test split, in points."
            ),
            mo.ui.table(input_table, selection=None, pagination=False),
            mo.accordion(
                {
                    "How to read this": mo.md(f"""
    - **All sentences**: runs that annotated whole documents, scored on all {_sentences}
      sentences of the dataset.
    - **Test**: scored only on uq-pet-token's {test_sentences} test sentences, from the
      whole-document runs or from runs that got each test sentence on its own.
    - In percent, mean ± SD over runs; micro F1 over all mentions.
    - Models are ordered by their best test F1, reasoning efforts from off to high.
    """)
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Caveats

    - **Different averages.** Whole-document scores are macro over documents; test-split
      scores are micro over mentions. They do not compare directly.
    - **Picked on test.** The best LLM configuration is chosen on the test split; there is no
      validation split for the LLMs.
    - **Out-of-schema types.** A predicted type outside the dataset's schema counts as a wrong
      mention on the test split.
    - **Unequal supervision.** The fine-tuned models train on every training sentence; the
      LLMs see the 5 bootstrap sentences, the labelled sentences uq-pet-token's active
      learning starts from.
    """)
    return


if __name__ == "__main__":
    app.run()
