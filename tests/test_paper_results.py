import json

from utils.paper_results import match_entities, tuning_winners

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
