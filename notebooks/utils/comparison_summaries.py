"""Read saved UQ-versus-random comparison summaries; no model or dataset loading."""

import json
from pathlib import Path

import polars as pl
from scipy import stats

UQ_METRICS = ("entropy", "least_confidence", "margin")
COMPARISON_SCHEMA = {
    "model_run": pl.String,
    "config_id": pl.String,
    "uq_metric": pl.String,
    "status": pl.String,
    "final_random_f1": pl.Float64,
    "final_uq_f1": pl.Float64,
    "gap_auc": pl.Float64,
    "final_gap": pl.Float64,
    "error": pl.String,
    "run_path": pl.String,
}


def load_test_comparisons(results_root: Path) -> pl.DataFrame:
    """One row per comparison in every test-split sweep under ``results_root``.

    Final random and UQ scores are seed-mean entity F1 at each seed's last round, read
    from the run's results.csv when the comparison completed.
    """
    rows = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("evaluation_split") != "test":
            continue
        for comparison in summary["comparisons"]:
            run_path = (
                summary_path.parent / comparison["run_dir"] if comparison.get("run_dir") else None
            )
            final_scores = {}
            if (
                comparison["status"] == "complete"
                and run_path is not None
                and (run_path / "results.csv").is_file()
            ):
                final_scores = dict(
                    pl.read_csv(run_path / "results.csv")
                    .filter(pl.col("round") == pl.col("round").max().over("seed"))
                    .group_by("arm")
                    .agg(pl.col("entity_f1").mean())
                    .iter_rows()
                )
            rows.append(
                {
                    "model_run": summary_path.parent.name,
                    "config_id": comparison["config_id"],
                    "uq_metric": comparison["uq_metric"],
                    "status": comparison["status"],
                    "final_random_f1": final_scores.get("random"),
                    "final_uq_f1": final_scores.get("uncertainty"),
                    "gap_auc": comparison.get("mean_test_entity_f1_gap_auc"),
                    "final_gap": comparison.get("mean_final_test_entity_f1_gap"),
                    "error": comparison.get("error", ""),
                    "run_path": str(run_path) if run_path is not None else "",
                }
            )
    return pl.DataFrame(rows, schema=COMPARISON_SCHEMA)


def uq_gap_ttests(search_root: Path) -> list[dict]:
    """Test each model's mean gap AUC across sampled configurations against zero.

    One row per model and UQ metric with at least two completed configurations: the mean
    gap and its 95% CI in pp, the two-sided one-sample t-test p-value, and Holm and
    Bonferroni adjustments across the three planned UQ metrics within each model.
    """
    rows = []
    for path in sorted(search_root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        model = path.parent.name.removesuffix("-random-5-seeds")
        for metric in UQ_METRICS:
            gaps = [
                row["mean_test_entity_f1_gap_auc"]
                for row in summary["comparisons"]
                if row["uq_metric"] == metric and row["status"] == "complete"
            ]
            if len(gaps) < 2:
                continue
            test = stats.ttest_1samp(gaps, 0, alternative="two-sided")
            ci = test.confidence_interval(confidence_level=0.95)
            rows.append(
                {
                    "Model": model,
                    "UQ metric": metric,
                    "Configurations": len(gaps),
                    "Mean improvement (pp)": 100 * sum(gaps) / len(gaps),
                    "95% CI lower (pp)": 100 * ci.low,
                    "95% CI upper (pp)": 100 * ci.high,
                    "p-value": float(test.pvalue),
                }
            )
    planned = len(UQ_METRICS)
    for model in {row["Model"] for row in rows}:
        # NaN p-values (identical gaps) sort last and stay NaN after adjustment.
        model_rows = sorted(
            (row for row in rows if row["Model"] == model),
            key=lambda row: (row["p-value"] != row["p-value"], row["p-value"]),
        )
        holm_p = 0.0
        for rank, row in enumerate(model_rows):
            p = row["p-value"]
            if p != p:
                row["Adjusted p-value (Holm)"] = p
                row["Adjusted p-value (Bonferroni)"] = p
                continue
            holm_p = max(holm_p, min(1.0, (planned - rank) * p))
            row["Adjusted p-value (Holm)"] = holm_p
            row["Adjusted p-value (Bonferroni)"] = min(1.0, planned * p)
    return rows
