"""Publish fixed random-only trials as a W&B sweep with one run per configuration."""

import hashlib
import json

import polars as pl

AUC_METRIC = "random_validation_entity_f1_auc"
FINAL_METRIC = "random_validation_final_entity_f1"
AXES = ("learning_rate", "batch_size", "k", "update_passes", "replay_ratio", "weight_decay")


def sweep_definition(config, plan):
    """Describe the existing local sampler; W&B does not schedule training."""
    parameters = {}
    for name in AXES:
        values = plan["search_space"][name]
        parameters[name] = (
            {"distribution": "log_uniform_values", "min": values["low"], "max": values["high"]}
            if name == "learning_rate"
            else {"values": values}
        )
    return {
        "name": config.sweep_name,
        "description": "Random-only validation; one run per configuration, averaged across model seeds. "
        "The saved local plan schedules trials; no W&B agents or UQ training are started.",
        "method": "random",
        "metric": {"name": config.objective, "goal": "maximize"},
        "parameters": parameters,
        "run_cap": config.num_configs,
    }


def chart_workspace(entity, project, sweep_name, objective):
    """Create a separate saved view with explicit parameter and F1 axes."""
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    metrics = [FINAL_METRIC, AUC_METRIC] if objective == AUC_METRIC else [AUC_METRIC, FINAL_METRIC]
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
                name="Random-selection hyperparameters and validation F1",
                is_open=True,
                panels=[
                    wr.ParallelCoordinatesPlot(
                        title="Mean validation F1 across seeds: one line per configuration",
                        columns=[
                            wr.ParallelCoordinatesPlotColumn(
                                metric=wr.Config(name), log=name == "learning_rate"
                            )
                            for name in AXES
                        ]
                        + [
                            wr.ParallelCoordinatesPlotColumn(
                                metric=wr.SummaryMetric(name),
                                display_name="Validation F1 AUC"
                                if name == AUC_METRIC
                                else "Final validation F1",
                            )
                            for name in metrics
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
    """Create/reuse a remote sweep and saved chart without touching training state."""
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    key = f"{entity}/{config.wandb_project}"
    state_path = root / "wandb_sweeps.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if key not in state:
        sweep_id = wandb.sweep(
            sweep_definition(config, plan), entity=entity, project=config.wandb_project
        )
        state[key] = {
            "sweep_id": sweep_id,
            "entity": entity,
            "project": config.wandb_project,
            "published": {},
        }
        save_state(state_path, state)
    record = state[key]
    if "workspace_url" not in record:
        workspace = chart_workspace(
            entity, config.wandb_project, config.sweep_name, config.objective
        )
        workspace.save()
        record["workspace_url"] = workspace.url
        save_state(state_path, state)
    return state_path, state, key


def publish_trial(config, root, entry, publication, score_function):
    """Publish completed validation CSVs; deterministic IDs make retries safe."""
    import wandb

    state_path, state, key = publication
    record = state[key]
    config_id = entry["config_id"]
    marker = root / "tuning" / config_id / "completed.json"
    if not marker.exists():
        return
    completed = json.loads(marker.read_text())
    if config_id in record["published"]:
        return
    run_dir = root / completed["run_dir"]
    metadata = json.loads((run_dir / "config.json").read_text())
    if not metadata.get("random_only") or metadata.get("evaluation_split") != "validation":
        raise ValueError(f"Refusing to publish non-validation/random-only trial: {config_id}")
    rows = pl.read_csv(run_dir / "results.csv").to_dicts()
    if sorted({row["seed"] for row in rows}) != sorted(config.model_seeds):
        raise ValueError(f"Incomplete seed coverage for {config_id}")
    scores = {metric: score_function(rows, metric) for metric in (AUC_METRIC, FINAL_METRIC)}
    if abs(scores[config.objective] - completed["score"]) > 1e-12:
        raise ValueError(f"Saved objective differs from raw validation rows: {config_id}")
    run_id = hashlib.sha256(f"{record['sweep_id']}/{config_id}".encode()).hexdigest()[:16]
    run = wandb.init(
        entity=record["entity"],
        project=record["project"],
        id=run_id,
        resume="allow",
        name=f"{config.sweep_name}-{config_id}",
        group=config.sweep_name,
        job_type="random_tuning_summary",
        reinit="create_new",
        dir=str(root),
        settings={"sweep_id": record["sweep_id"], "quiet": True, "console": "off"},
        config={
            **entry["parameters"],
            "config_id": config_id,
            "checkpoint": config.checkpoint,
            "model_seeds": config.model_seeds,
            "evaluation_split": "validation",
            "random_only": True,
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
