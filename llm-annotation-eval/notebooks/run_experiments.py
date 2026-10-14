import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import json
    import sys

    import marimo as mo

    return html, json, mo, sys


@app.cell
def _(html, mo):
    def pre(text):
        """Text as it is, wrapped, in a box of limited height."""
        return mo.Html(
            '<pre style="white-space: pre-wrap; max-height: 24rem; overflow: auto; '
            f'font-size: 0.8rem">{html.escape(text)}</pre>'
        )


    def stack(*items):
        """mo.vstack without the items that are None, which it would show as "None"."""
        return mo.vstack([i for i in items if i is not None])


    def bar(done, total):
        """A progress bar, done of total."""
        _width = 100 * done / total if total else 0
        return mo.Html(
            '<div style="height: 0.5rem; border-radius: 0.25rem; background: #8884">'
            f'<div style="height: 100%; width: {_width:.1f}%; border-radius: 0.25rem; '
            'background: #3b9c5a"></div></div>'
        )


    def percent(value):
        return "" if value is None else f"{value:.1%}"

    return bar, percent, pre, stack


@app.cell
def _(mo, sys):
    REPO_DIR = mo.notebook_dir().parent
    CONFIG_DIR = REPO_DIR / "configs"
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from eval import datasets, test_split
    from experiments import batch, checks, report, runner
    from experiments import config as run_config

    PRED_DIR = batch.PRED_DIR
    # notebook arguments, e.g. marimo edit notebooks/run_experiments.py -- --config configs/pet/md.yaml,
    # or as a script: python notebooks/run_experiments.py --config configs/pet/md.yaml --workers 8
    cli = dict(mo.cli_args())
    script_mode = mo.app_meta().mode == "script"
    return (
        CONFIG_DIR,
        PRED_DIR,
        REPO_DIR,
        batch,
        checks,
        cli,
        datasets,
        report,
        run_config,
        runner,
        script_mode,
        test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Experiments

    Pick a config file and change it below, check which runs it makes and what they would send,
    then run them while the scores so far update after every document. Runs are saved in
    `res/pred/<config>/run-<i>/` (see `src/experiments/runner.py`); a config that was run before
    resumes its unfinished runs. The results of all runs are in `notebooks/eval_table.py`.

    As a script, the notebook runs a config without the editor, like this:

    ```bash
    uv run python notebooks/run_experiments.py --config configs/pet/md.yaml --only 'gemma-4-31b-*' --workers 4
    ```

    `--only` takes a pattern of config names and can be repeated; add `--export` to write every
    run's `test-predictions.jsonl`, or `--check` to check the data against uq-pet-token
    (`--check-rebuild` also rebuilds its splits).
    """)
    return


@app.cell
def _(CONFIG_DIR, REPO_DIR, cli, mo):
    # every config file, as a path relative to the repository, and the one given as --config
    config_files = sorted(
        p.relative_to(REPO_DIR).as_posix()
        for p in [*CONFIG_DIR.rglob("*.yaml"), *CONFIG_DIR.rglob("*.yml")]
    )
    _default = str(cli.get("config", "configs/pet/md.yaml"))
    if _default not in config_files:
        if "config" not in cli:
            _default = config_files[0]
        elif (REPO_DIR / _default).is_file():
            config_files.append(_default)
        else:
            raise FileNotFoundError(f"--config {_default}: no such file")
    config_file = mo.ui.dropdown(config_files, value=_default, searchable=True, label="Config file")
    return (config_file,)


@app.cell
def _(REPO_DIR, config_file, mo):
    # changing the file starts over from what it holds
    config_text = (REPO_DIR / config_file.value).read_text(encoding="utf8")
    editor = mo.ui.code_editor(value=config_text, language="yaml", min_height=160)
    save = mo.ui.run_button(label="Save to the file", kind="warn")
    mo.vstack([
        mo.md("## 1. Configuration"),
        config_file,
        editor,
        mo.md(
            "Changes apply to the runs below without saving; see `configs/README.md` for the keys. "
            "A config with `model` is named after the file, so changing its model or prompts "
            "makes its existing runs impossible to resume: rename the file, or use `models`, "
            "whose entries are named by their `name`."
        ),
        save,
    ])
    return config_text, editor, save


@app.cell
def _(REPO_DIR, config_file, config_text, editor, mo, save):
    mo.stop(not save.value)
    if editor.value == config_text:
        mo.output.replace(mo.md(f"`{config_file.value}` is unchanged."))
    else:
        (REPO_DIR / config_file.value).write_text(editor.value, encoding="utf8")
        mo.output.replace(mo.callout(f"Saved `{config_file.value}`.", kind="success"))
    return


@app.cell
def _(REPO_DIR, config_file, editor, mo, run_config):
    _path = REPO_DIR / config_file.value
    try:
        configs = run_config.parse_configs(editor.value, _path.stem, config_file.value)
    except Exception as _error:
        configs = []
        mo.output.replace(mo.callout(f"The config is not valid: {_error}", kind="danger"))
    return (configs,)


@app.cell
def _(batch, cli, configs, mo):
    _only = cli.get("only", [])
    _only = [_only] if isinstance(_only, str) else _only
    _names = [c.name for c in configs]
    selection = mo.ui.multiselect(
        _names,
        value=[c.name for c in batch.select(configs, _only)],
        label="Configs to run",
        full_width=True,
    )
    workers = mo.ui.number(
        start=1, stop=32, step=1, value=int(cli.get("workers", 4)),
        label="Documents at the same time per run",
    )
    mo.vstack([selection, workers]) if configs else None
    return selection, workers


@app.cell
def _(configs, selection):
    selected = [c for c in configs if c.name in selection.value]
    return (selected,)


@app.cell
def _(datasets, test_split):
    _cache = {}


    def gold_data(dataset_name):
        """Gold documents and test sentences of a dataset, read once."""
        if dataset_name not in _cache:
            _dataset = datasets.DATASETS[dataset_name]
            _cache[dataset_name] = {
                "documents": datasets.read_documents(_dataset.gold_file),
                "test": test_split.load_test_split(_dataset.test_file),
            }
        return _cache[dataset_name]


    def input_documents(config):
        """The documents a run of the config annotates."""
        _dataset = datasets.DATASETS[config.dataset]
        if config.task == "md-test-sentences":
            _documents = test_split.sentence_documents(gold_data(config.dataset)["test"])
        else:
            _documents = gold_data(config.dataset)["documents"]
        return _documents[:config.max_documents]

    return gold_data, input_documents


@app.cell
def _(mo):
    # bumped after each run, so the run table reads the files again
    get_runs_changed, set_runs_changed = mo.state(0)
    return get_runs_changed, set_runs_changed


@app.cell
def _(
    PRED_DIR,
    batch,
    get_runs_changed,
    input_documents,
    json,
    mo,
    percent,
    runner,
    selected,
    stack,
):
    get_runs_changed()
    _problems = batch.problems(selected)
    _rows = []
    for _config in [] if _problems else selected:
        _planned = len(input_documents(_config))
        for _run in range(1, _config.repeats + 1):
            _directory = runner.run_dir(PRED_DIR, _config, _run)
            _done = len(runner.read_predictions(_directory, _config.task))
            _problem = runner.resume_problem(_directory, _config, _run)
            if _problem is not None:
                _state = "can't resume: settings or prompts changed"
            elif _done == 0:
                _state = "to do"
            elif _done < _planned:
                _state = "to resume"
            else:
                _state = "done"
            _metrics_file = _directory / "metrics.json"
            _metrics = json.loads(_metrics_file.read_text()) if _metrics_file.exists() else {}
            _rows.append({
                "config": _config.name,
                "run": _run,
                "dataset": _config.dataset,
                "task": _config.task,
                "model": _config.model,
                "reasoning": _config.reasoning_effort or "",
                "documents": f"{_done}/{_planned}",
                "state": _state,
                "MD F1": percent(_metrics.get("md", {}).get("macro", {}).get("f1", {}).get("mean")),
                "test F1": percent(_metrics.get("test", {}).get("entity_f1")),
            })
    stack(
        mo.md("### Runs"),
        *[mo.callout(p, kind="danger") for p in _problems],
        mo.ui.table(_rows, selection=None, pagination=False) if _rows else mo.md("No config selected."),
        mo.md(
            "A run that can't be resumed stops the run of its config; see its reason in the "
            "prompt preview."
        ) if any(r["state"].startswith("can't") for r in _rows) else None,
    )
    return


@app.cell
def _(mo, selected):
    preview_config = mo.ui.dropdown(
        {c.name: c for c in selected},
        value=selected[0].name if selected else None,
        label="Config",
    )
    return (preview_config,)


@app.cell
def _(input_documents, mo, preview_config):
    _documents = input_documents(preview_config.value) if preview_config.value else []
    preview_document = mo.ui.dropdown(
        {d.id: d for d in _documents},
        value=_documents[0].id if _documents else None,
        searchable=True,
        label="Document",
    )
    return (preview_document,)


@app.cell
def _(PRED_DIR, json, mo, pre, preview_config, preview_document, runner, stack):
    mo.stop(preview_config.value is None or preview_document.value is None, mo.md("## 2. Prompts"))
    _config = preview_config.value
    _document = preview_document.value
    _tabs = {}
    _parameters = None
    for _step, _annotator in runner.build_annotators(_config, None).items():
        _params = _annotator.get_params(_document)
        _instruction, _user = (m["content"][0]["text"] for m in _params.pop("messages"))
        _parameters = json.dumps(_params)
        _tabs[f"{_step}: instruction"] = stack(
            mo.md(f"The {_annotator.instruction_role} message, {len(_instruction.split()):,} "
                  "words, the same for every document."),
            pre(_instruction),
        )
        _tabs[f"{_step}: document"] = stack(
            mo.md(f"The user message, {len(_user.split()):,} words."),
            mo.md("The relation step gets the document with the predicted mentions; here the "
                  "gold mentions stand in for them.") if _step == "relations" else None,
            pre(_user),
        )
    _problems = [
        f"run {_run}: {_problem}"
        for _run in range(1, _config.repeats + 1)
        if (_problem := runner.resume_problem(runner.run_dir(PRED_DIR, _config, _run), _config, _run))
    ]
    stack(
        mo.md("## 2. Prompts\n\nWhat the selected runs would send, built by the same code that "
              "sends it (`runner.build_annotators`)."),
        mo.hstack([preview_config, preview_document], justify="start"),
        *[mo.callout(p, kind="warn") for p in _problems],
        mo.md(f"Request parameters besides the messages: `{_parameters}`"),
        mo.ui.tabs(_tabs),
    )
    return


@app.cell
def _(mo, selected):
    run_button = mo.ui.run_button(
        label=f"Run {len(selected)} config{'s' if len(selected) != 1 else ''}",
        kind="success",
        disabled=not selected,
    )
    mo.vstack([
        mo.md(
            "## 3. Run\n\nThe scores below update after every document: on the documents done "
            "so far (MD/RE F1, macro over documents, lenient), and on the test sentences among "
            "them (test F1 exact and partial, like uq-pet-token). Stop the cell to interrupt; the "
            "documents being annotated are finished and saved, and running again resumes."
        ),
        run_button,
    ])
    return (run_button,)


@app.cell
def _(
    PRED_DIR,
    bar,
    batch,
    cli,
    datasets,
    gold_data,
    input_documents,
    mo,
    percent,
    report,
    run_button,
    script_mode,
    selected,
    set_runs_changed,
    stack,
    workers,
):
    mo.stop(not (run_button.value or (script_mode and "config" in cli)))
    _problems = batch.problems(selected)
    if _problems:
        raise RuntimeError("\n".join(_problems))
    _clients = batch.clients(selected)

    # one row per run done, and the run going on
    finished_rows = []
    _live = {"title": "Starting", "done": 0, "planned": 0, "failed": [], "row": None}


    def _row(config, run, scores, state):
        return {
            "config": config.name,
            "run": run,
            "state": state,
            "documents": scores.get("documents", 0),
            "MD F1": percent(scores.get("md_f1")),
            "RE F1": percent(scores.get("re_f1")),
            "test sentences": scores.get("test_sentences", 0),
            "test F1": percent(scores.get("test_f1")),
            "test partial F1": percent(scores.get("test_partial_f1")),
            "tokens in": scores.get("prompt_tokens", 0),
            "tokens out": scores.get("completion_tokens", 0),
            "dropped lines": scores.get("dropped_lines", 0),
            "cut off": scores.get("truncated_answers", 0),
        }


    def _show():
        _rows = finished_rows + ([_live["row"]] if _live["row"] else [])
        _failed = f", skipped for now: {', '.join(_live['failed'])}" if _live["failed"] else ""
        mo.output.replace(stack(
            mo.md(f"**{_live['title']}**: {_live['done']}/{_live['planned']} documents{_failed}"),
            bar(_live["done"], _live["planned"]),
            mo.ui.table(_rows, selection=None, pagination=False) if _rows else None,
        ))


    for _i, _config in enumerate(selected, start=1):
        _dataset = datasets.DATASETS[_config.dataset]
        _gold = gold_data(_config.dataset)


        def _scores(directory, config=_config, gold=_gold, dataset=_dataset):
            return report.scores_so_far(
                directory, config.task, gold["documents"], gold["test"], dataset.types
            )


        def _on_document(run, directory, document_id, error, config=_config, scores=_scores):
            if error is not None:
                _live["failed"].append(document_id)
            _so_far = scores(directory)
            _live.update(
                title=f"{config.name}, run {run}/{config.repeats}", done=_so_far["documents"],
                row=_row(config, run, _so_far, "running"),
            )
            _show()


        _live.update(
            title=f"{_config.name} ({_i}/{len(selected)})", done=0, failed=[], row=None,
            planned=len(input_documents(_config)),
        )
        _show()
        for _result in batch.run_config(
            _config, _clients[_config.provider], int(workers.value), PRED_DIR,
            on_document=_on_document,
        ):
            _state = f"failed: {_result.error}" if _result.error else "done"
            _so_far = _scores(_result.directory) if _result.metrics else {}
            finished_rows.append(_row(_config, _result.run, _so_far, _state))
        _live["row"] = None
        set_runs_changed(lambda n: n + 1)

    _summary = report.write_summary(PRED_DIR)
    _live.update(title="Finished", done=_live["planned"])
    _show()
    _failed = [r["config"] for r in finished_rows if r["state"].startswith("failed")]
    if script_mode:
        report.print_summary(_summary)
        if _failed:
            raise RuntimeError(f"Failed: {', '.join(_failed)}. Run them again to resume.")
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export test predictions")
    rebuild = mo.ui.checkbox(label="also rebuild the splits with uq-pet-token's code (slow)")
    check_button = mo.ui.run_button(label="Check the data")
    mo.vstack([
        mo.md(
            "## 4. For the paper\n\n"
            "**Test predictions for uq-pet-token.** Writes `test-predictions.jsonl` next to every "
            "run's `pet.jsonl` (the test sentences of its dataset with predicted BIO tags). Runs "
            "write theirs when they are scored; this redoes all of them, e.g. after a change to "
            "`eval.test_split`."
        ),
        export_button,
        mo.md(
            "**Same data and scoring as uq-pet-token?** Compares `res/gold/` with the test gold "
            "stored in uq-pet-token's reported supervised runs, rescores those runs' predictions "
            "against their `results.csv`, checks that every run here used today's gold files "
            "and that no test sentence is in a prompt (see `src/experiments/checks.py`)."
        ),
        mo.hstack([check_button, rebuild], justify="start"),
    ])
    return check_button, export_button, rebuild


@app.cell
def _(PRED_DIR, REPO_DIR, cli, export_button, mo, report, script_mode):
    mo.stop(not (export_button.value or (script_mode and "export" in cli)))
    _written, _skipped = report.export_test_predictions(PRED_DIR)
    _message = f"Wrote `test-predictions.jsonl` for {len(_written)} prediction files."
    if _skipped:
        _message += " Skipped, as they don't cover the test split: " + ", ".join(
            f"`{p.relative_to(REPO_DIR)}`" for p in _skipped
        )
    print(_message) if script_mode else None
    mo.callout(mo.md(_message), kind="success")
    return


@app.cell
def _(check_button, checks, cli, mo, rebuild, script_mode):
    _from_cli = script_mode and ("check" in cli or "check-rebuild" in cli)
    mo.stop(not (check_button.value or _from_cli))
    _report = checks.run(rebuild=rebuild.value or "check-rebuild" in cli)
    _rows = [
        {"check": r.check, "ok": "ok" if r.ok else "FAIL", "result": r.message}
        for r in _report.results
    ]
    if script_mode:
        for _r in _rows:
            print(f"{_r['ok']:4} {_r['check']}: {_r['result']}")
        print(*_report.notes, sep="\n")
        if not _report.ok:
            raise RuntimeError("A data check failed.")
    mo.vstack([
        mo.callout(
            "All checks passed." if _report.ok else "A check failed.",
            kind="success" if _report.ok else "danger",
        ),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("\n".join(f"- {n}" for n in _report.notes)),
    ])
    return


if __name__ == "__main__":
    app.run()
