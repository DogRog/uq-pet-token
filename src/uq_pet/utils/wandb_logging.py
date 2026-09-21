"""W&B credentials, evaluation records, and comparison run lifecycles."""

import os
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager

from rich.console import Console

_WANDB_ARMS = ("uncertainty", "random")
_WANDB_COMMON_FIELDS = (
    "seed",
    "round",
    "total_rounds",
    "n_acquired",
    "percent_acquired",
    "scoreable_pool_tokens",
    "token_budget",
)
_WANDB_HIDDEN_ARM_FIELDS = ("n_new", "n_replay")
_WANDB_TRACKED_ARM_FIELDS = (
    "entity_f1",
    "entity_macro_f1",
    "entity_precision",
    "entity_recall",
    "token_accuracy",
    "train_loss",
)


def require_wandb_credentials(environment: Mapping[str, str]) -> None:
    """Reject an online W&B run before downloading data or starting training."""
    if environment.get("WANDB_MODE", "").strip().lower() == "offline":
        return
    if environment.get("WANDB_API_KEY", "").strip():
        return
    raise ValueError("WANDB_API_KEY is missing; set it unless WANDB_MODE=offline")


def configure_wandb_metrics(wandb_run, uq_metric: str, *, random_only: bool = False) -> None:
    """Keep bookkeeping out of auto-panels and use acquisition as the x-axis."""
    arms = ("random",) if random_only else (uq_metric, "random")
    for field in _WANDB_COMMON_FIELDS:
        wandb_run.define_metric(f"evaluation/{field}", hidden=True)
    for field in _WANDB_HIDDEN_ARM_FIELDS:
        for arm in arms:
            wandb_run.define_metric(f"evaluation/{field}/{arm}", hidden=True)
    for field in _WANDB_TRACKED_ARM_FIELDS:
        for arm in arms:
            wandb_run.define_metric(
                f"evaluation/{field}/{arm}",
                step_metric="evaluation/percent_acquired",
            )


def make_wandb_evaluation_log(
    rows: list[dict], uq_metric: str, *, random_only: bool = False
) -> dict:
    """Build one W&B step with a separate scalar series for each arm."""
    arms = ("random",) if random_only else _WANDB_ARMS
    if len(rows) != len(arms):
        raise ValueError("a W&B evaluation step requires exactly one row per arm")

    rows_by_arm = {row["arm"]: row for row in rows}
    if set(rows_by_arm) != set(arms):
        raise ValueError(f"a W&B evaluation step requires {' and '.join(arms)} rows")

    first = rows_by_arm[arms[0]]
    for field in _WANDB_COMMON_FIELDS:
        if any(rows_by_arm[arm][field] != first[field] for arm in arms[1:]):
            raise ValueError(f"W&B evaluation rows disagree on {field}")

    payload = {f"evaluation/{field}": first[field] for field in _WANDB_COMMON_FIELDS}
    arm_fields = set(first) - set(_WANDB_COMMON_FIELDS) - {"arm"}
    for arm in arms:
        if set(rows_by_arm[arm]) - set(_WANDB_COMMON_FIELDS) - {"arm"} != arm_fields:
            raise ValueError("W&B evaluation rows have different metric fields")
        for field in sorted(arm_fields):
            display_arm = uq_metric if arm == "uncertainty" else arm
            payload[f"evaluation/{field}/{display_arm}"] = rows_by_arm[arm][field]
    return payload


@contextmanager
def wandb_comparison_logging(
    config,
    experiment,
    config_id,
    effective_precision,
    slots,
    *,
    console: Console,
    evaluation_split="test",
    random_only=False,
):
    """Own one W&B run per metric/seed in the parent, with separate acquisition axes."""
    if not config.wandb_enabled:
        yield None
        return

    import wandb

    runs = {}
    success = False
    offline = os.environ.get("WANDB_MODE", "").strip().lower() == "offline"
    with ExitStack() as cleanup:

        def log_evaluation(metric, rows):
            pair = rows[-1:] if random_only else rows[-2:]
            payload = make_wandb_evaluation_log(pair, metric, random_only=random_only)
            seed = pair[0]["seed"]
            key = (metric, seed)
            if key not in runs:
                acquisition = "random" if random_only else metric
                name = f"{config.wandb_run_name or config.sweep_name}-{config_id}-{acquisition}-seed{seed}"
                if random_only:
                    name += "-validation-random"
                settings = experiment.model_copy(
                    update={"uq_metric": metric, "model_seeds": [seed]}
                )
                logged_settings = settings.resolved_dict()
                if random_only:
                    logged_settings.pop("uq_metric", None)
                run = wandb.init(
                    project=config.wandb_project,
                    name=name,
                    group=config.sweep_name,
                    job_type="random_search",
                    reinit="create_new",
                    settings={"quiet": True, "console": "off"},
                    dir=str(slots[metric]),
                    config={
                        **logged_settings,
                        "wandb_run_name": name,
                        "seed": seed,
                        "config_id": config_id,
                        "sweep_name": config.sweep_name,
                        "evaluation_split": evaluation_split,
                        "random_only": random_only,
                        "effective_precision": effective_precision,
                        "shared_bootstrap_and_random": not random_only,
                    },
                )
                cleanup.callback(lambda run=run: run.finish(exit_code=0 if success else 1))
                runs[key] = run
                configure_wandb_metrics(run, metric, random_only=random_only)
                if not offline:
                    if len(runs) == 1:
                        console.print(f"W&B project: {run.get_project_url()}", markup=False)
                    console.print(f"W&B run ({metric}, seed {seed}): {run.get_url()}", markup=False)
            runs[key].log(payload)

        yield log_evaluation
        success = True
