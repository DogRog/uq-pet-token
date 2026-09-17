# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    return Path, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Best configurations from random-baseline sweeps

    Check trial completeness, then select the highest saved validation score in each
    completed model sweep. Ties use ascending configuration ID.
    """)
    return


@app.cell
def _(Path, json):
    results_root = Path(__file__).resolve().parents[1] / "results" / "random_baseline_search"
    completeness = []
    best_rows = []
    best_configs = {}
    for plan_path in sorted(results_root.glob("*/plan.json")):
        plan = json.loads(plan_path.read_text())
        if plan.get("mode") != "random_baseline_tuning":
            continue
        summary_path = plan_path.with_name("summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        trials = summary.get("trials", [])
        completed = [trial for trial in trials if trial["status"] == "complete"]
        is_complete = (
            summary.get("complete", False)
            and len(completed) == len(trials) == plan["num_configs"]
            and {trial["config_id"] for trial in completed}
            == {trial["config_id"] for trial in plan["configurations"]}
        )
        completeness.append(
            {
                "model": plan["fixed_config"]["checkpoint"],
                "sweep": plan_path.parent.name,
                "completed": len(completed),
                "planned": plan["num_configs"],
                "status": "Complete" if is_complete else "Incomplete",
            }
        )
        if is_complete:
            best = min(completed, key=lambda trial: (-trial["score"], trial["config_id"]))
            best_rows.append(
                {
                    "model": plan["fixed_config"]["checkpoint"],
                    "sweep": plan_path.parent.name,
                    "config_id": best["config_id"],
                    "validation_score": best["score"],
                    "objective": summary["objective"],
                    **best["parameters"],
                }
            )
            best_configs[plan_path.parent.name] = {**plan["fixed_config"], **best["parameters"]}
    return best_rows, completeness


@app.cell(hide_code=True)
def _(completeness, mo):
    mo.stop(not completeness, mo.md("No sweeps found in `results/random_baseline_search`."))
    mo.vstack(
        [
            mo.md("## Trial completeness"),
            mo.ui.table(completeness, selection=None),
            mo.md("Incomplete sweeps are excluded from the best configurations and download.")
            if any(row["status"] == "Incomplete" for row in completeness)
            else mo.md("All trials complete."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(best_rows, mo):
    mo.stop(not best_rows, mo.md("No completed sweeps available yet."))
    mo.vstack(
        [
            mo.md("## Best configurations"),
            mo.ui.table(
                best_rows,
                selection=None,
                format_mapping={
                    "learning_rate": "{:.8e}",
                    "validation_score": "{:.6f}",
                },
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
