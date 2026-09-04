import pytest

from uq_pet.tuning import (
    SEARCH_SPACE,
    make_tuning_split,
    mean_entity_f1_gap_auc,
    suggest_search_config,
)


def test_tuning_split_withholds_labelled_validation_and_reindexes_pool():
    pool_inputs = [
        {
            "pool_idx": pool_idx,
            "document_name": "doc",
            "sentence_id": pool_idx,
            "tokens": [f"token-{pool_idx}", "tail"],
        }
        for pool_idx in range(4)
    ]
    pool_gold = {
        (pool_idx, word_idx): pool_idx + word_idx for pool_idx in range(4) for word_idx in range(2)
    }

    tuning_inputs, tuning_gold, validation = make_tuning_split(
        pool_inputs,
        pool_gold,
        validation_fraction=0.25,
        seed=7,
    )

    assert len(tuning_inputs) == 3
    assert len(validation) == 1
    assert [item["pool_idx"] for item in tuning_inputs] == [0, 1, 2]
    assert set(tuning_gold) == {
        (pool_idx, word_idx) for pool_idx in range(3) for word_idx in range(2)
    }
    assert validation[0]["ner_tags"] == [2, 3]
    assert all("ner_tags" not in item for item in tuning_inputs)


def test_tuning_split_is_reproducible_and_validates_fraction():
    pool_inputs = [
        {
            "pool_idx": pool_idx,
            "document_name": "doc",
            "sentence_id": pool_idx,
            "tokens": [str(pool_idx)],
        }
        for pool_idx in range(3)
    ]
    pool_gold = {(pool_idx, 0): pool_idx for pool_idx in range(3)}

    first = make_tuning_split(pool_inputs, pool_gold, validation_fraction=1 / 3, seed=3)
    second = make_tuning_split(pool_inputs, pool_gold, validation_fraction=1 / 3, seed=3)
    assert first == second

    with pytest.raises(ValueError, match="must be in"):
        make_tuning_split(pool_inputs, pool_gold, validation_fraction=1, seed=3)


def test_mean_entity_f1_gap_auc_pairs_arms_and_averages_seeds():
    gaps = {
        0: (0.0, 0.2, 0.2),
        1: (0.0, 0.1, 0.3),
    }
    results = []
    for seed, seed_gaps in gaps.items():
        for round_idx, (percent, gap) in enumerate(zip((0.0, 50.0, 100.0), seed_gaps, strict=True)):
            results.extend(
                [
                    {
                        "seed": seed,
                        "round": round_idx,
                        "arm": "uncertainty",
                        "percent_acquired": percent,
                        "entity_f1": 0.5 + gap,
                    },
                    {
                        "seed": seed,
                        "round": round_idx,
                        "arm": "random",
                        "percent_acquired": percent,
                        "entity_f1": 0.5,
                    },
                ]
            )

    assert mean_entity_f1_gap_auc(results) == pytest.approx(0.1375)


def test_mean_entity_f1_gap_auc_rejects_half_finished_round():
    with pytest.raises(ValueError, match="uncertainty and random"):
        mean_entity_f1_gap_auc(
            [
                {
                    "seed": 0,
                    "round": 0,
                    "arm": "uncertainty",
                    "percent_acquired": 0.0,
                    "entity_f1": 0.5,
                }
            ]
        )


def test_optuna_suggestions_use_shared_discrete_search_space():
    class FirstChoiceTrial:
        def suggest_categorical(self, name, choices):
            assert tuple(choices) == SEARCH_SPACE[name]
            return choices[0]

    config = suggest_search_config(FirstChoiceTrial())

    assert config == {name: values[0] for name, values in SEARCH_SPACE.items()}
