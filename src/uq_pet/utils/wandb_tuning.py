"""Publish fixed validation trials as a W&B sweep with one summary run per configuration."""

import hashlib
import json

import polars as pl

AUC_METRIC = "random_validation_entity_f1_auc"
FINAL_METRIC = "random_validation_final_entity_f1"
AXES = ("learning_rate", "batch_size", "k", "update_passes", "replay_ratio", "weight_decay")
SUPERVISED_METRIC = "validation_entity_f1"
SUPERVISED_AXES = ("learning_rate", "batch_size", "weight_decay")


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
    return {
        "name": config.sweep_name,
        "description": "Random-only validation; one run per configuration, averaged across model seeds. "
        "The saved local plan schedules trials; no W&B agents or UQ training are started.",
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
    """Create a separate saved view with explicit parameter and F1 axes."""
    metrics = [FINAL_METRIC, AUC_METRIC] if objective == AUC_METRIC else [AUC_METRIC, FINAL_METRIC]
    return _tuning_workspace(
        entity,
        project,
        sweep_name,
        section="Random-selection hyperparameters and validation F1",
        axes=AXES,
        metrics={
            name: "Validation F1 AUC" if name == AUC_METRIC else "Final validation F1"
            for name in metrics
        },
    )


def supervised_chart_workspace(entity, project, sweep_name):
    return _tuning_workspace(
        entity,
        project,
        sweep_name,
        section="Supervised hyperparameters and validation F1",
        axes=SUPERVISED_AXES,
        metrics={SUPERVISED_METRIC: "Mean validation entity F1"},
    )


def _tuning_workspace(entity, project, sweep_name, *, section, axes, metrics):
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    return ws.Workspace(
        entity=entity,
        project=project,
        name=f"{sweep_name} — validation tuning",
        runset_settings=ws.RunsetSettings(
            filters=[
                expr.Config("run_role") == "configuration_summary",
                expr.Config("sweep_name") == sweep_name,
            ]
        ),
        sections=[
            ws.Section(
                name=section,
                is_open=True,
                panels=[
                    wr.ParallelCoordinatesPlot(
                        title="Mean validation F1 across seeds: one line per configuration",
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
                    )
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
    if key not in state:
        sweep_id = wandb.sweep(definition, entity=entity, project=config.wandb_project)
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


def publish_trial(config, root, entry, publication, score_function):
    """Publish completed validation CSVs; deterministic IDs make retries safe."""
    completed, metadata, rows = _completed_trial(config, root, entry, publication)
    if completed is None:
        return
    config_id = entry["config_id"]
    if not metadata.get("random_only"):
        raise ValueError(f"Refusing to publish non-validation/random-only trial: {config_id}")
    scores = {metric: score_function(rows, metric) for metric in (AUC_METRIC, FINAL_METRIC)}
    if abs(scores[config.objective] - completed["score"]) > 1e-12:
        raise ValueError(f"Saved objective differs from raw validation rows: {config_id}")
    _publish_summary(
        config,
        root,
        entry,
        publication,
        scores,
        job_type="random_tuning_summary",
        settings={"random_only": True},
    )


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
        settings={"epochs": config.epochs},
    )


def _completed_trial(config, root, entry, publication):
    """Return a completed, unpublished validation trial's marker, settings, and rows."""
    _, state, key = publication
    config_id = entry["config_id"]
    marker = root / "tuning" / config_id / "completed.json"
    if not marker.exists() or config_id in state[key]["published"]:
        return None, None, None
    completed = json.loads(marker.read_text())
    run_dir = root / completed["run_dir"]
    metadata = json.loads((run_dir / "config.json").read_text())
    if metadata.get("evaluation_split") != "validation":
        raise ValueError(f"Refusing to publish non-validation/random-only trial: {config_id}")
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
        settings={"sweep_id": record["sweep_id"], "quiet": True, "console": "off"},
        config={
            **entry["parameters"],
            "config_id": config_id,
            "checkpoint": config.checkpoint,
            "model_seeds": config.model_seeds,
            "evaluation_split": "validation",
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
