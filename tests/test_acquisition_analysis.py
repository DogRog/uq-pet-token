import json

import polars as pl

from analysis.acquisition_analysis import pet_sentences, tag_selections


def saved_logs():
    selections, evaluations = [], []
    for seed in (0, 1):
        for arm in ("random", "uncertainty"):
            order = [0, 1, 2] if arm == "random" else [2, 1, 0]
            scores = [0.2, 0.6, 0.5] if arm == "random" else [0.2, 0.4, 0.7]
            for round_idx, acquired in enumerate((0, 2, 3)):
                evaluations.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "round": round_idx,
                        "n_acquired": acquired,
                        "n_new": (0, 2, 1)[round_idx],
                        "scoreable_pool_tokens": 3,
                        "entity_f1": scores[round_idx],
                    }
                )
            for index, token_id in enumerate(order):
                selections.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "round": 1 if index < 2 else 2,
                        "pool_idx": 0,
                        "word_idx": token_id,
                        "document_name": "doc",
                        "sentence_id": 0,
                        "token": ["A", "a", "run"][token_id],
                        "label": ["O", "O", "B-Activity"][token_id],
                    }
                )
    return selections, pl.DataFrame(evaluations)


def test_tag_selections_join_progress_and_order_labels_by_pool_tokens():
    selections, results = saved_logs()
    results = results.with_columns((100 * pl.col("n_acquired") / 3).alias("percent_acquired"))

    frame, label_order = tag_selections(selections, results, "run-a")

    assert label_order == ["O", "B-Activity"]
    assert frame.height == len(selections)
    assert frame["run_id"].unique().to_list() == ["run-a"]
    assert frame.filter(pl.col("round") == 2)["percent_acquired"].unique().to_list() == [100.0]


def test_pet_sentences_maps_document_and_sentence_to_words(tmp_path):
    path = tmp_path / "pet.jsonl"
    path.write_text(
        json.dumps({"document name": "doc", "sentence-ID": 0, "tokens": ["A", "a", "run"]}) + "\n"
    )

    assert pet_sentences(path) == {("doc", 0): ["A", "a", "run"]}
    assert pet_sentences(tmp_path / "missing.jsonl") == {}
