import importlib.util
from pathlib import Path

import pytest

SEARCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bert_token_uq_search.py"
SPEC = importlib.util.spec_from_file_location("bert_token_uq_search", SEARCH_PATH)
assert SPEC is not None and SPEC.loader is not None
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)
SEARCH_SPACE = search.SEARCH_SPACE
make_tuning_split = search.make_tuning_split
mean_entity_f1_gap_auc = search.mean_entity_f1_gap_auc
suggest_search_config = search.suggest_search_config


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


@pytest.mark.parametrize("name", ["tpe", "grid", "random"])
def test_search_config_chooses_optuna_sampler(name):
    config = search.SearchConfig.model_validate({"sampler": name, "model_seeds": [3]})
    expected = {
        "tpe": search.optuna.samplers.TPESampler,
        "grid": search.optuna.samplers.GridSampler,
        "random": search.optuna.samplers.RandomSampler,
    }
    assert isinstance(search.make_sampler(config.sampler, 7), expected[name])
    assert "sampler" not in config.experiment_config().active_learning_kwargs()
    assert config.experiment_config().model_seeds == [3]


def run_search(tmp_path, sampler, trials=2, seed_workers=1):
    import argparse
    import json

    parser = argparse.ArgumentParser()
    search.configure_parser(parser)
    args = parser.parse_args(
        [
            "--trials",
            str(trials),
            "--studies-dir",
            str(tmp_path),
            "--config-json",
            json.dumps({"sampler": sampler, "model_seeds": [0], "seed_workers": seed_workers}),
        ]
    )
    search.run(args, parser)


@pytest.fixture
def fake_experiment(monkeypatch):
    # Two grid points allow real Optuna persistence/exhaustion tests without training.
    monkeypatch.setattr(search, "SEARCH_SPACE", {"k": (8, 16)})
    pool = [
        {"pool_idx": i, "document_name": "doc", "sentence_id": i, "tokens": [str(i)]}
        for i in range(5)
    ]
    test_examples = [{"tokens": ["test-must-not-be-used"]}]
    monkeypatch.setattr(search, "download_pet_ner", lambda: "unused")
    monkeypatch.setattr(
        search,
        "load_pet_splits",
        lambda _: (
            [],
            pool,
            {(i, 0): 0 for i in range(5)},
            test_examples,
        ),
    )
    monkeypatch.setattr(search, "get_device", lambda: "cpu")
    calls = []

    def train(seed, inputs, gold, validation, **kwargs):
        assert validation != test_examples
        assert len(inputs) == 4 and len(validation) == 1
        assert all("ner_tags" not in item for item in inputs)
        assert {item["sentence_id"] for item in inputs}.isdisjoint(
            item["sentence_id"] for item in validation
        )
        assert "sampler" not in kwargs
        calls.append(kwargs["k"])
        rows = [
            {
                "seed": 0,
                "round": round_idx,
                "arm": arm,
                "percent_acquired": percent,
                "entity_f1": 0.5,
                "token_budget": 4,
                "scoreable_pool_tokens": 4,
            }
            for round_idx, percent in enumerate((0, 100))
            for arm in ("uncertainty", "random")
        ]
        return rows, []

    monkeypatch.setattr(search, "run_active_learning", train)
    return calls


@pytest.mark.parametrize("sampler", ["tpe", "grid", "random"])
def test_all_samplers_share_validation_and_outputs(tmp_path, fake_experiment, sampler):
    import json

    run_search(tmp_path, sampler)
    root = tmp_path / "bert-token-uq"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed_trials"] == 2
    assert summary["context"]["sampler"] == sampler
    best = search.ExperimentConfig.model_validate_json((root / "best_config.json").read_text())
    assert best.k in (8, 16)
    assert len(fake_experiment) == 2
    trials = json.loads((root / "trials.json").read_text())
    for trial in trials:
        output = Path(trial["user_attrs"]["run_dir"])
        assert (output / "results.csv").exists()
        assert json.loads((output / "selections.json").read_text()) == []
        config = json.loads((output / "config.json").read_text())
        assert config["search_context"]["sampler"] == sampler


def test_search_resumes_when_only_seed_concurrency_changes(tmp_path, fake_experiment):
    import json

    run_search(tmp_path, "random", trials=1)
    run_search(tmp_path, "random", trials=1, seed_workers=2)
    root = tmp_path / "bert-token-uq"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed_trials"] == 2
    assert "seed_workers" not in summary["context"]["fixed_config"]
    trial = json.loads((root / "trials.json").read_text())[-1]
    config = json.loads((Path(trial["user_attrs"]["run_dir"]) / "config.json").read_text())
    assert config["seed_workers"] == 2


def test_grid_resumes_then_exits_before_data_load_when_exhausted(
    tmp_path,
    fake_experiment,
    monkeypatch,
):
    run_search(tmp_path, "grid", trials=1)
    run_search(tmp_path, "grid", trials=10)
    assert sorted(fake_experiment) == [8, 16]

    def unexpected_download():
        pytest.fail("exhausted grid must not load data")

    monkeypatch.setattr(search, "download_pet_ner", unexpected_download)
    run_search(tmp_path, "grid", trials=10)
    assert len(fake_experiment) == 2


def test_resume_rejects_sampler_change_before_training(tmp_path, fake_experiment):
    run_search(tmp_path, "random", trials=1)
    with pytest.raises(SystemExit, match="2"):
        run_search(tmp_path, "tpe", trials=1)
    assert len(fake_experiment) == 1


def test_legacy_tpe_context_can_resume(tmp_path, fake_experiment):
    run_search(tmp_path, "tpe", trials=1)
    study = search.optuna.load_study(
        study_name="bert-token-uq",
        storage=f"sqlite:///{tmp_path / 'bert-token-uq' / 'study.db'}",
    )
    legacy_context = study.user_attrs["context"]
    legacy_context.pop("sampler")
    study.set_user_attr("context", legacy_context)
    study.set_user_attr("context_fingerprint", search._context_fingerprint(legacy_context))
    run_search(tmp_path, "tpe", trials=1)
    assert len(fake_experiment) == 2
    assert study.user_attrs["context"]["sampler"] == "tpe"
