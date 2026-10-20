import json

import polars as pl
import pytest

from analysis import paper_results
from analysis.paper_results import (
    curve_auc,
    floor_and_ceiling,
    headroom_summary,
    llm_test_scores,
    match_entities,
    pool_to_ceiling,
    pool_to_match,
    random_sweep_comparisons,
    read_llm_predictions,
    scores_at_budgets,
    tuning_winners,
)

GOLD = ["B-Action", "O", "B-Entity", "I-Entity", "O", "B-Condition"]


def test_exact_match_needs_same_words_and_type():
    predicted = ["B-Action", "O", "B-Entity", "O", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "exact")
    assert spans == [(0, 1, "Action", True), (2, 3, "Entity", False), (5, 6, "Entity", False)]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", False), (5, 6, "Condition", False)]


def test_partial_match_needs_one_shared_word_and_each_gold_matches_once():
    predicted = ["B-Action", "O", "B-Entity", "B-Entity", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "partial")
    assert spans == [
        (0, 1, "Action", True),
        (2, 3, "Entity", True),
        (3, 4, "Entity", False),
        (5, 6, "Entity", False),
    ]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", True), (5, 6, "Condition", False)]


def _write_tuning_sweep(root, name, dataset, complete):
    sweep = root / "random_baseline_search" / name
    sweep.mkdir(parents=True)
    plan = {
        "fixed_config": {"checkpoint": "roberta-base", "dataset": dataset},
        "search_space": {"k": [32, 64], "learning_rate": {"low": 1e-5, "high": 5e-4}},
    }
    (sweep / "plan.json").write_text(json.dumps(plan))
    (sweep / "summary.json").write_text(json.dumps({"complete": complete, "trials": []}))
    if complete:
        (sweep / "selection.json").write_text(
            json.dumps({"config_id": "config_0007", "validation_score": 0.25})
        )
        (sweep / "best_config.json").write_text(
            json.dumps({"checkpoint": "roberta-base", "k": 64, "learning_rate": 3e-5, "seed": 0})
        )


def test_tag_name_names_medical_types_and_keeps_the_rest():
    assert [paper_results.tag_name(tag) for tag in ("O", "B-T03", "I-T43", "B-Actor")] == [
        "O",
        "B-Activity",
        "I-Activity Data",
        "B-Actor",
    ]


def test_tuning_winners_lists_tuned_settings_of_finished_sweeps(tmp_path):
    _write_tuning_sweep(tmp_path, "done", "quishpi", complete=True)
    _write_tuning_sweep(tmp_path, "running", "pet", complete=False)
    assert tuning_winners(tmp_path).to_dicts() == [
        {
            "dataset": "Quishpi",
            "model": "RoBERTa",
            "config_id": "config_0007",
            "validation_score": 0.25,
            "k": 64,
            "learning_rate": 3e-5,
        }
    ]


def test_random_sweep_comparisons_compute_gaps_for_any_saved_score(tmp_path):
    sweep = tmp_path / "random_search" / "pet-sweep"
    run_dir = sweep / "runs" / "config_0003"
    run_dir.mkdir(parents=True)
    (sweep / "plan.json").write_text(json.dumps({"fixed_config": {"checkpoint": "roberta-base"}}))
    comparison = {
        "status": "complete",
        "run_dir": "runs/config_0003",
        "config_id": "config_0003",
        "uq_metric": "margin",
        "parameters": {"k": 32},
    }
    (sweep / "summary.json").write_text(json.dumps({"comparisons": [comparison]}))
    # Two seeds; each arm runs two rounds over 0-100% of the pool.
    pl.DataFrame(
        {
            "seed": [0, 0, 0, 0, 1, 1, 1, 1],
            "arm": ["random", "random", "uncertainty", "uncertainty"] * 2,
            "round": [0, 1] * 4,
            "percent_acquired": [0.0, 100.0] * 4,
            "entity_macro_f1": [0.2, 0.6, 0.4, 0.6, 0.2, 0.4, 0.2, 0.8],
        }
    ).write_csv(run_dir / "results.csv")
    row = random_sweep_comparisons(tmp_path, "entity_macro_f1").row(0, named=True)
    assert (row["dataset"], row["model"], row["config_id"], row["k"]) == (
        "PET",
        "RoBERTa",
        "config_0003",
        32,
    )
    # AUCs: random 0.4 and 0.3, uncertainty 0.5 and 0.5; finals 0.6/0.6 and 0.4/0.8.
    assert row["gap_auc"] == pytest.approx(0.15)
    assert row["final_gap"] == pytest.approx(0.2)
    assert row["random_auc"] == pytest.approx(0.35)
    assert row["random_final"] == pytest.approx(0.5)
    assert random_sweep_comparisons(tmp_path, "entity_partial_macro_f1").is_empty()


# Two PET test sentences; tag ids index data_prep.DATASET_TAGS["pet"] (1 B-Actor, 3 B-Activity).
TEST_SPLIT = (
    {
        "document_name": "doc-1.1",
        "sentence_id": 0,
        "tokens": ["Clerk", "checks"],
        "ner_tags": [1, 3],
    },
    {"document_name": "doc-1.1", "sentence_id": 1, "tokens": ["It", "ends"], "ner_tags": [0, 3]},
)


def _write_llm_run(root, config, run, tags, task="md-test-sentences", effort="high"):
    run_dir = root / config / f"run-{run}"
    (run_dir / task).mkdir(parents=True)
    settings = {"dataset": "pet", "task": task, "model": "org/llm", "reasoning_effort": effort}
    manifest = {"config": config, "run": run, "settings": settings}
    (run_dir / "run.json").write_text(json.dumps(manifest))
    lines = [
        json.dumps(
            {
                "document name": s["document_name"],
                "sentence-ID": s["sentence_id"],
                "tokens": s["tokens"],
                "ner-tags": sentence_tags,
            }
        )
        for s, sentence_tags in zip(TEST_SPLIT, tags, strict=True)
    ]
    (run_dir / task / "test-predictions.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir / task / "test-predictions.jsonl"


def test_llm_test_scores_rescore_finished_runs_of_the_task(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_results, "_test_split", lambda dataset: TEST_SPLIT)
    perfect = [["B-Actor", "B-Activity"], ["O", "B-Activity"]]
    # Off-schema types are kept and count as wrong.
    half = [["B-Actor", "B-actor data"], ["O", "O"]]
    _write_llm_run(tmp_path, "gemma-md-test-sentences-reasoning-high", 1, perfect)
    _write_llm_run(tmp_path, "gemma-md-test-sentences-reasoning-high", 2, half)
    _write_llm_run(tmp_path, "gemma-md", 1, perfect, task="md")
    (tmp_path / "unfinished-md-test-sentences" / "run-1").mkdir(parents=True)
    scores = llm_test_scores(tmp_path)
    assert scores.select("dataset", "config", "llm", "reasoning_effort", "run").rows() == [
        ("PET", "gemma · high", "gemma", "high", 1),
        ("PET", "gemma · high", "gemma", "high", 2),
    ]
    assert scores["entity_f1"].to_list() == pytest.approx([1.0, 0.4])


def test_read_llm_predictions_rejects_another_test_split(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_results, "_test_split", lambda dataset: TEST_SPLIT)
    predictions = _write_llm_run(tmp_path, "gemma", 1, [["O", "O"], ["O", "O"]])
    assert read_llm_predictions(predictions, "pet") == [["O", "O"], ["O", "O"]]
    monkeypatch.setattr(paper_results, "_test_split", lambda dataset: TEST_SPLIT[::-1])
    with pytest.raises(ValueError, match="test split"):
        read_llm_predictions(predictions, "pet")


def test_curve_auc_normalizes_trapezoids_per_seed_and_arm():
    results = pl.DataFrame(
        {
            "seed": [0, 0, 0, 0],
            "arm": ["random", "random", "uncertainty", "uncertainty"],
            "round": [0, 1, 0, 1],
            "percent_acquired": [0.0, 50.0, 0.0, 50.0],
            "entity_f1": [0.2, 0.6, 0.2, 0.8],
        }
    )

    assert curve_auc(results) == {
        (0, "random"): pytest.approx(0.4),
        (0, "uncertainty"): pytest.approx(0.5),
    }


def test_pool_to_match_uses_seed_mean_curves_and_shares_the_random_arm():
    rows = []
    for metric in ("entropy", "margin"):
        for seed, offset in ((0, -0.1), (1, 0.1)):
            for round_, percent, random_f1, uq_f1 in ((0, 0.0, 0.2, 0.2), (1, 50.0, 0.4, 0.6)):
                for arm, f1 in (("random", random_f1), ("uncertainty", uq_f1)):
                    rows.append(
                        {
                            "dataset": "PET",
                            "model": "BERT",
                            "uq_metric": metric,
                            "seed": seed,
                            "arm": arm,
                            "round": round_,
                            "percent_acquired": percent,
                            # The margin arm never gets above 0.5.
                            "entity_f1": f1 + offset
                            if metric == "entropy" or arm == "random"
                            else min(f1, 0.4) + offset,
                        }
                    )
    reach = pool_to_match(
        pl.DataFrame(rows), "entity_f1", pl.DataFrame({"dataset": ["PET"], "target": [0.5]})
    )
    assert dict(reach.sort("series").select("series", "pool").rows()) == {
        "entropy": 50.0,
        "margin": None,
        "random": None,
    }


def _curve(model, seed, arm, points, metric="entropy"):
    """Rows of one run: (percent acquired, entity F1) per round, in round order."""
    return [
        {
            "dataset": "PET",
            "model": model,
            "acquisition": "Top-K",
            "uq_metric": metric,
            "seed": seed,
            "arm": arm,
            "round": round_,
            "percent_acquired": percent,
            "entity_f1": f1,
        }
        for round_, (percent, f1) in enumerate(points)
    ]


CURVES = pl.DataFrame(
    _curve("BERT", 0, "random", [(0, 0.2), (40, 0.4), (100, 0.6)])
    + _curve("BERT", 0, "uncertainty", [(0, 0.2), (40, 0.6), (100, 0.6)])
    + _curve("BERT", 1, "random", [(0, 0.2), (50, 0.4), (100, 0.6)])
    + _curve("BERT", 1, "uncertainty", [(0, 0.2), (50, 0.5), (100, 0.6)])
)
CEILINGS = pl.DataFrame({"dataset": ["PET"], "model": ["BERT"], "ceiling": [0.7]})


def test_floor_and_ceiling_average_round_zero_and_supervised_seeds():
    supervised = pl.DataFrame(
        {"dataset": ["PET", "PET"], "model": ["BERT", "BERT"], "entity_f1": [0.6, 0.8]}
    )
    row = floor_and_ceiling(CURVES, supervised, "entity_f1").row(0, named=True)
    assert row["floor"] == pytest.approx(0.2)
    assert row["ceiling"] == pytest.approx(0.7)
    assert row["ceiling_sd"] == pytest.approx(0.1414, abs=1e-4)
    assert row["ceiling_seeds"] == 2


def test_scores_at_budgets_interpolate_between_each_runs_rounds():
    at = scores_at_budgets(CURVES, "entity_f1", [20, 50])
    scores = {(r["seed"], r["arm"], r["budget"]): r["score"] for r in at.iter_rows(named=True)}
    assert scores[0, "random", 20.0] == pytest.approx(0.3)
    assert scores[0, "random", 50.0] == pytest.approx(0.4 + 0.2 * 10 / 60)
    assert scores[1, "uncertainty", 20.0] == pytest.approx(0.32)
    assert scores[1, "uncertainty", 50.0] == pytest.approx(0.5)


def test_headroom_summary_pairs_seeds_and_divides_the_mean_gap_by_the_headroom():
    at = scores_at_budgets(CURVES, "entity_f1", [50, 100])
    summary = headroom_summary(at, CEILINGS, ["dataset"])
    half = summary.filter(pl.col("budget") == 50).row(0, named=True)
    random = (0.4 + 0.2 * 10 / 60 + 0.4) / 2
    uq = (0.6 + 0.5) / 2
    assert half["random"] == pytest.approx(random)
    assert half["gap"] == pytest.approx(uq - random)
    assert half["gap_low"] < half["gap"] < half["gap_high"]
    assert half["headroom"] == pytest.approx(0.7 - random)
    assert half["share_closed"] == pytest.approx((uq - random) / (0.7 - random))
    assert half["pairs"] == 2
    # At the full pool both arms sit at 0.6 under a 0.7 ceiling: no gap, headroom left.
    full = summary.filter(pl.col("budget") == 100).row(0, named=True)
    assert full["share_closed"] == pytest.approx(0.0)
    above = headroom_summary(at, CEILINGS.with_columns(ceiling=pl.lit(0.5)), ["dataset"])
    assert above.filter(pl.col("budget") == 100)["share_closed"].to_list() == [None]


def test_pool_to_ceiling_finds_the_first_round_at_the_target_or_leaves_it_empty():
    reach = pool_to_ceiling(CURVES, "entity_f1", CEILINGS, 0.8)
    pools = {(r["seed"], r["arm"]): r["pool"] for r in reach.iter_rows(named=True)}
    assert pools == {
        (0, "random"): 100.0,
        (0, "uncertainty"): 40.0,
        (1, "random"): 100.0,
        (1, "uncertainty"): 100.0,
    }
    assert pool_to_ceiling(CURVES, "entity_f1", CEILINGS, 1.0)["pool"].null_count() == 4
