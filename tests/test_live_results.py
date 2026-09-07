import ast
from pathlib import Path

import marimo as mo
import polars as pl
import pytest

from charts import make_learning_chart


def live_panel_builder(chart_function):
    """Exercise the notebook's renderer with the chart import held by its kernel."""
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "bert_token_uq.py"
    function = next(
        node
        for node in ast.walk(ast.parse(notebook.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "make_live_seed_panel"
    )
    namespace = {
        "mo": mo,
        "config": {"uq_metric": "entropy"},
        "make_learning_chart": chart_function,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["make_live_seed_panel"]


@pytest.mark.parametrize("records", [[], [{"seed": 2, "round": 0}]])
def test_waiting_seed_does_not_call_a_stale_chart_import(records):
    def previous_chart(rows, seed, metric):
        # The previous imported implementation failed on an empty, schema-less frame.
        pl.DataFrame(rows).filter(pl.col("seed") == seed)
        pytest.fail("a seed without round 0 must not render a chart")

    panel = live_panel_builder(previous_chart)(records, 0)
    assert "Waiting for round 0" in panel.text
    assert "420px" in panel.text


def test_ready_seed_renders_chart_inside_the_reserved_space():
    records = [{"seed": 2, "round": 1, "total_rounds": 1, "percent_acquired": 10.0}]
    calls = []

    def chart(rows, seed, metric):
        calls.append((rows, seed, metric))
        return mo.md("Updated chart")

    panel = live_panel_builder(chart)(records, 2)
    assert calls == [(records, 2, "entropy")]
    assert "Complete" in panel.text
    assert "Updated chart" in panel.text
    assert "420px" in panel.text


def test_learning_chart_accepts_empty_records_in_a_fresh_kernel():
    specification = make_learning_chart([], 0, "entropy").to_dict()
    assert len(specification["hconcat"]) == 3
    assert all(not rows for rows in specification["datasets"].values())
