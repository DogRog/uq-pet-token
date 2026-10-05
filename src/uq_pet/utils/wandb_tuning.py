"""Publish fixed tuning trials as a W&B sweep with one summary run per configuration.

Random tuning scores the validation holdout, or the test split for the appendix's
test-tuned random oracle; names and labels follow the split that scored the trials.
"""

import contextlib
import hashlib
import io
import json
import os

import polars as pl

AXES = ("learning_rate", "batch_size", "k", "update_passes", "replay_ratio", "weight_decay")
SUPERVISED_METRIC = "validation_entity_f1"
SUPERVISED_AXES = ("learning_rate", "batch_size", "weight_decay")
WINNER_TAG = "winner"
MAX_BARS = 500


def tuning_metrics(split):
    """Return the AUC and final-F1 objective names that score random trials on a split."""
    return f"random_{split}_entity_f1_auc", f"random_{split}_final_entity_f1"


def objective_split(objective):
    """Return the split named by a random tuning objective (random_<split>_...)."""
    return "test" if objective.startswith("random_test_") else "validation"


def _sweep_parameters(plan, axes):
    parameters = {}
    for name in axes:
        values = plan["search_space"][name]
        parameters[name] = (
            {"distribution": "log_uniform_values", "min": values["low"], "max": values["high"]}
            if name == "learning_rate"
            else {"values": values}
        )
    return parameters


def sweep_definition(config, plan):
    """Describe the existing local sampler; W&B does not schedule training."""
    oracle = (
        "Selects the appendix's test-tuned random oracle, never a headline result. "
        if config.tuning_split == "test"
        else ""
    )
    return {
        "name": config.sweep_name,
        "description": f"Random-only {config.tuning_split} tuning; one run per configuration, "
        f"averaged across model seeds. {oracle}The saved local plan schedules trials; no W&B "
        "agents or UQ training are started.",
        "method": "random",
        "metric": {"name": config.objective, "goal": "maximize"},
        "parameters": _sweep_parameters(plan, AXES),
        "run_cap": config.num_configs,
    }


def supervised_sweep_definition(config, plan):
    """Describe the supervised sampler; W&B does not schedule training."""
    return {
        "name": config.sweep_name,
        "description": "Fully supervised validation; one run per configuration, averaged across "
        "model seeds. The saved local plan schedules trials; no W&B agents are started.",
        "method": "random",
        "metric": {"name": SUPERVISED_METRIC, "goal": "maximize"},
        "parameters": _sweep_parameters(plan, SUPERVISED_AXES),
        "run_cap": config.num_configs,
    }


def chart_workspace(entity, project, sweep_name, objective):
    """Create a saved view that ranks configurations by the objective, winner first."""
    split = objective_split(objective)
    auc, final = tuning_metrics(split)
    labels = {auc: f"{split.capitalize()} F1 AUC", final: f"Final {split} F1"}
    # The objective is the last parallel-coordinates axis, which sets the line colours.
    metrics = {name: labels[name] for name in (final, auc) if name != objective}
    return _tuning_workspace(
        entity,
        project,
        sweep_name,
        split=split,
        section=f"Random-selection hyperparameters and {split} F1",
        axes=AXES,
        metrics={**metrics, objective: labels[objective]},
    )


def supervised_chart_workspace(entity, project, sweep_name):
    return _tuning_workspace(
        entity,
        project,
        sweep_name,
        split="validation",
        section="Supervised hyperparameters and validation F1",
        axes=SUPERVISED_AXES,
        metrics={SUPERVISED_METRIC: "Mean validation entity F1"},
    )


def _tuning_workspace(entity, project, sweep_name, *, split, section, axes, metrics):
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    objective, objective_label = list(metrics.items())[-1]
    return ws.Workspace(
        entity=entity,
        project=project,
        name=f"{sweep_name} — {split} tuning",
        runset_settings=ws.RunsetSettings(
            filters=[
                expr.Config("run_role") == "configuration_summary",
                expr.Config("sweep_name") == sweep_name,
            ],
            # List the best configuration first; the winner also carries the winner tag.
            order=[ws.Ordering(ws.Summary(objective), ascending=False)],
        ),
        sections=[
            ws.Section(
                name=section,
                is_open=True,
                panels=[
                    wr.BarPlot(
                        title=f"{objective_label} by configuration: the top bar wins",
                        metrics=[wr.SummaryMetric(objective)],
                        orientation="h",
                        max_runs_to_show=MAX_BARS,
                        max_bars_to_show=MAX_BARS,
                        layout=wr.Layout(w=24, h=12),
                    ),
                    wr.ParallelCoordinatesPlot(
                        title=f"Mean {split} F1 across seeds: one line per configuration",
                        columns=[
                            wr.ParallelCoordinatesPlotColumn(
                                metric=wr.Config(name), log=name == "learning_rate"
                            )
                            for name in axes
                        ]
                        + [
                            wr.ParallelCoordinatesPlotColumn(
                                metric=wr.SummaryMetric(name), display_name=display_name
                            )
                            for name, display_name in metrics.items()
                        ],
                        layout=wr.Layout(w=24, h=12),
                    ),
                ],
            )
        ],
    )


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def ensure_sweep(config, plan, root):
    return _ensure_sweep(
        config,
        root,
        sweep_definition(config, plan),
        lambda entity: chart_workspace(
            entity, config.wandb_project, config.sweep_name, config.objective
        ),
    )


def ensure_supervised_sweep(config, plan, root):
    return _ensure_sweep(
        config,
        root,
        supervised_sweep_definition(config, plan),
        lambda entity: supervised_chart_workspace(entity, config.wandb_project, config.sweep_name),
    )


def _ensure_sweep(config, root, definition, make_workspace):
    """Create/reuse a remote sweep and saved chart without touching training state."""
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    key = f"{entity}/{config.wandb_project}"
    state_path = root / "wandb_sweeps.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if key in state and not _sweep_exists(api, state[key]):
        # Deleting the sweep or its project in W&B leaves a stale local record.
        wandb.termwarn(
            f"W&B sweep {key}/{state[key]['sweep_id']} no longer exists; "
            "creating a new sweep and chart and republishing completed trials."
        )
        del state[key]
        save_state(state_path, state)
    if key not in state:
        sweep_id = _create_sweep(definition, entity, config.wandb_project)
        state[key] = {
            "sweep_id": sweep_id,
            "entity": entity,
            "project": config.wandb_project,
            "published": {},
        }
        save_state(state_path, state)
    record = state[key]
    if "workspace_url" not in record:
        workspace = make_workspace(entity)
        workspace.save()
        record["workspace_url"] = workspace.url
        save_state(state_path, state)
    return state_path, state, key


def _sweep_exists(api, record) -> bool:
    try:
        api.sweep(f"{record['entity']}/{record['project']}/{record['sweep_id']}")
    except Exception as error:
        # Only a definite "not found" discards the record; network errors still raise.
        if "Could not find sweep" not in str(error):
            raise
        return False
    return True


def _create_sweep(definition, entity, project):
    """Create a sweep without its stdout banner or its exported entity and project."""
    import wandb

    # wandb.sweep() exports WANDB_ENTITY/PROJECT, so every later init would warn.
    exported = {name: os.environ.get(name) for name in ("WANDB_ENTITY", "WANDB_PROJECT")}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return wandb.sweep(definition, entity=entity, project=project)
    finally:
        for name, value in exported.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def publish_trial(config, root, entry, publication, score_function):
    """Publish completed tuning CSVs; deterministic IDs make retries safe."""
    split = config.tuning_split
    completed, metadata, rows = _completed_trial(config, root, entry, publication, split=split)
    if completed is None:
        return
    config_id = entry["config_id"]
    if not metadata.get("random_only"):
        raise ValueError(f"Refusing to publish non-{split}/random-only trial: {config_id}")
    scores = {metric: score_function(rows, metric) for metric in tuning_metrics(split)}
    if abs(scores[config.objective] - completed["score"]) > 1e-12:
        raise ValueError(f"Saved objective differs from raw {split} rows: {config_id}")
    _publish_summary(
        config,
        root,
        entry,
        publication,
        scores,
        job_type="random_tuning_summary",
        settings={"random_only": True, "evaluation_split": split},
    )


def mark_winner(publication, root):
    """Tag the winning configuration's summary run so the sweep shows which one won."""
    selection_path = root / "selection.json"
    if not selection_path.exists():
        return
    state_path, state, key = publication
    record = state[key]
    config_id = json.loads(selection_path.read_text())["config_id"]
    if record.get("winner") == config_id or config_id not in record["published"]:
        return
    import wandb

    run = wandb.Api().run(
        f"{record['entity']}/{record['project']}/{record['published'][config_id]}"
    )
    if WINNER_TAG not in run.tags:
        run.tags = [*run.tags, WINNER_TAG]
        run.update()
    record["winner"] = config_id
    save_state(state_path, state)


def publish_supervised_trial(config, root, entry, publication):
    """Publish a completed supervised trial's seed-mean validation F1 to its sweep."""
    completed, metadata, rows = _completed_trial(config, root, entry, publication)
    if completed is None:
        return
    values = [float(row["entity_f1"]) for row in rows]
    score = sum(values) / len(values)
    if abs(score - completed["score"]) > 1e-12:
        raise ValueError(f"Saved objective differs from raw validation rows: {entry['config_id']}")
    _publish_summary(
        config,
        root,
        entry,
        publication,
        {SUPERVISED_METRIC: score, "std_validation_entity_f1": completed["std_entity_f1"]},
        job_type="supervised_tuning_summary",
        settings={"epochs": config.epochs, "evaluation_split": "validation"},
    )


def _completed_trial(config, root, entry, publication, *, split="validation"):
    """Return a completed, unpublished tuning trial's marker, settings, and rows."""
    _, state, key = publication
    config_id = entry["config_id"]
    marker = root / "tuning" / config_id / "completed.json"
    if not marker.exists() or config_id in state[key]["published"]:
        return None, None, None
    completed = json.loads(marker.read_text())
    run_dir = root / completed["run_dir"]
    metadata = json.loads((run_dir / "config.json").read_text())
    if metadata.get("evaluation_split") != split:
        raise ValueError(f"Refusing to publish non-{split}/random-only trial: {config_id}")
    rows = pl.read_csv(run_dir / "results.csv").to_dicts()
    if sorted({row["seed"] for row in rows}) != sorted(config.model_seeds):
        raise ValueError(f"Incomplete seed coverage for {config_id}")
    return completed, metadata, rows


def _publish_summary(config, root, entry, publication, scores, *, job_type, settings):
    import wandb

    state_path, state, key = publication
    record = state[key]
    config_id = entry["config_id"]
    run_id = hashlib.sha256(f"{record['sweep_id']}/{config_id}".encode()).hexdigest()[:16]
    run = wandb.init(
        entity=record["entity"],
        project=record["project"],
        id=run_id,
        resume="allow",
        name=f"{config.sweep_name}-{config_id}",
        group=config.sweep_name,
        job_type=job_type,
        reinit="create_new",
        dir=str(root),
        settings={"sweep_id": record["sweep_id"], "silent": True, "console": "off"},
        config={
            **entry["parameters"],
            "config_id": config_id,
            "checkpoint": config.checkpoint,
            "model_seeds": config.model_seeds,
            **settings,
            "run_role": "configuration_summary",
            "sweep_name": config.sweep_name,
        },
    )
    try:
        run.log(scores)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)
    record["published"][config_id] = run_id
    save_state(state_path, state)
