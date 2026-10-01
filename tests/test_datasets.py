"""Check CoNLL-2003 loading and that the dataset choice reaches labels and saved plans."""

import json

import polars as pl
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from uq_pet import active_learning, pet_data, search, supervised
from uq_pet.config import ExperimentConfig, RandomSearchConfig, SupervisedConfig
from uq_pet.pet_data import CONLL_TAGS, load_conll_splits, load_splits


def fake_conll(n_train, n_test, names=CONLL_TAGS):
    features = Features(
        {
            "id": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=names)),
        }
    )

    def split(prefix, n):
        return Dataset.from_dict(
            {
                "id": [str(idx) for idx in range(n)],
                "tokens": [[f"{prefix}{idx}", "x"] for idx in range(n)],
                "ner_tags": [[1, 0] for _ in range(n)],
            },
            features=features,
        )

    return DatasetDict({"train": split("train", n_train), "test": split("test", n_test)})


def test_conll_uses_full_train_pool_and_test_split(monkeypatch):
    monkeypatch.setattr(pet_data, "load_dataset", lambda *args, **kwargs: fake_conll(405, 100))

    seed, pool, gold, test = load_conll_splits()

    assert (len(seed), len(pool), len(test)) == (5, 400, 100)
    assert all("ner_tags" not in example for example in pool)
    assert [example["pool_idx"] for example in pool] == list(range(400))
    assert len(gold) == 2 * 400 and set(gold.values()) == {0, 1}
    seed_tokens = {example["tokens"][0] for example in seed}
    pool_tokens = {example["tokens"][0] for example in pool}
    assert not seed_tokens & pool_tokens
    assert all(example["document_name"] == "test" for example in test)
    assert load_conll_splits() == (seed, pool, gold, test)


def test_conll_rejects_changed_label_order(monkeypatch):
    reordered = [CONLL_TAGS[0], *reversed(CONLL_TAGS[1:])]
    monkeypatch.setattr(
        pet_data, "load_dataset", lambda *args, **kwargs: fake_conll(400, 100, reordered)
    )
    with pytest.raises(ValueError, match="label order"):
        load_conll_splits()


def test_load_splits_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_splits("ontonotes")


def test_pet_plans_stay_unchanged_and_conll_plans_record_the_dataset():
    pet = search.sample_plan(RandomSearchConfig(num_configs=1))
    conll = search.sample_plan(RandomSearchConfig(num_configs=1, dataset="conll2003"))
    assert "dataset" not in pet["fixed_config"]
    assert conll["fixed_config"]["dataset"] == "conll2003"

    assert "dataset" not in supervised.grid_plan(SupervisedConfig())["fixed_config"]
    conll_supervised = SupervisedConfig(dataset="conll2003")
    assert supervised.grid_plan(conll_supervised)["fixed_config"]["dataset"] == "conll2003"
    assert conll_supervised.sweep_name == "distilbert-base-cased-conll2003-supervised"
    assert SupervisedConfig().sweep_name == "distilbert-base-cased-supervised"

    assert "conll2003" in ExperimentConfig(dataset="conll2003").resolved_wandb_run_name()
    assert "conll2003" not in ExperimentConfig().resolved_wandb_run_name()


def test_supervised_results_are_matched_by_dataset(tmp_path):
    for sweep, pool in (("a_conll", {"dataset": "conll2003"}), ("b_pet", {})):
        run_dir = tmp_path / sweep / "test" / "pool_100pct" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({"checkpoint": "bert", **pool}))
        pl.DataFrame({"seed": [0], "entity_f1": [0.5]}).write_csv(run_dir / "results.csv")
        (run_dir.parent / "completed.json").write_text(
            json.dumps({"run_dir": "test/pool_100pct/run"})
        )

    assert supervised.load_test_results(tmp_path, "bert")[0] == "b_pet"
    assert supervised.load_test_results(tmp_path, "bert", "conll2003")[0] == "a_conll"


def test_engine_uses_the_chosen_dataset_labels(monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    seen_labels = []

    def fake_loader(checkpoint, device, *, labels):
        seen_labels.append(labels)
        return TinyModel(), object()

    def fake_evaluate(*args, labels, **kwargs):
        seen_labels.append(labels)
        return {"entity_f1": 0.2, "token_accuracy": 0.8}

    pool = [
        {"pool_idx": idx, "document_name": "train", "sentence_id": idx, "tokens": ["w"]}
        for idx in range(2)
    ]
    gold = {(0, 0): 1, (1, 0): 5}
    monkeypatch.setattr(active_learning, "set_seed", lambda seed: None)
    monkeypatch.setattr(active_learning, "load_token_classifier", fake_loader)
    monkeypatch.setattr(active_learning, "prepare_inference_batches", lambda *a, **kw: [])
    monkeypatch.setattr(active_learning, "scoreable_token_keys", lambda *a, **kw: set(gold))
    monkeypatch.setattr(active_learning, "train_items", lambda *a, **kw: 0.5)
    monkeypatch.setattr(active_learning, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        active_learning,
        "score_token_uncertainty",
        lambda *a, excluded, **kw: {key: 1.0 for key in gold if key not in excluded},
    )

    _, selections = active_learning.run_active_learning(
        [{"tokens": ["seed"], "ner_tags": [0]}],
        pool,
        gold,
        [],
        dataset="conll2003",
        checkpoint="fake",
        model_seeds=[0],
        uq_metric="entropy",
        k=2,
        max_pool_percent=100,
        bootstrap_epochs=1,
        update_passes=1,
        replay_ratio=0.0,
        learning_rate=5e-5,
        weight_decay=0.01,
        batch_size=1,
        score_batch_size=1,
        max_length=8,
        device=torch.device("cpu"),
    )

    assert all(labels == CONLL_TAGS for labels in seen_labels)
    assert {row["label"] for row in selections} == {"B-PER", "B-LOC"}
