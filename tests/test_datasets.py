"""Check dataset loading, pool percentages, and that both reach labels and saved plans."""

import json

import httpx
import polars as pl
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub.errors import GatedRepoError

from uq_pet import active_learning, data_prep, search, supervised
from uq_pet.config import ExperimentConfig, RandomSearchConfig, SupervisedConfig
from uq_pet.data_prep import (
    CONLL_TAGS,
    MEDICAL_TAGS,
    QUISHPI_TAGS,
    brat_sentences,
    load_conll_splits,
    load_splits,
)


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


def test_conll_uses_full_test_and_nested_pool_percentages(monkeypatch):
    monkeypatch.setattr(data_prep, "load_dataset", lambda *args, **kwargs: fake_conll(405, 100))

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

    half_seed, half_pool, half_gold, half_test = load_conll_splits(pool_percent=50)
    assert (half_seed, half_test) == (seed, test)
    assert half_pool == pool[:200]
    assert half_gold == {key: tag for key, tag in gold.items() if key[0] < 200}


def test_pet_pool_percent_keeps_seed_and_test(tmp_path):
    path = tmp_path / "pet.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "document name": f"doc-{idx}",
                    "sentence-ID": 0,
                    "tokens": [f"w{idx}"],
                    "ner-tags": ["O"],
                }
            )
            for idx in range(50)
        )
    )
    seed, pool, gold, test = data_prep.load_pet_splits(path)
    quarter = data_prep.load_pet_splits(path, pool_percent=25)

    assert (len(seed), len(pool), len(test)) == (5, 35, 10)
    assert (quarter[0], quarter[1], quarter[3]) == (seed, pool[:9], test)
    with pytest.raises(ValueError, match="is empty"):
        data_prep.load_pet_splits(path, pool_percent=1)


def test_conll_rejects_changed_label_order(monkeypatch):
    reordered = [CONLL_TAGS[0], *reversed(CONLL_TAGS[1:])]
    monkeypatch.setattr(
        data_prep, "load_dataset", lambda *args, **kwargs: fake_conll(400, 100, reordered)
    )
    with pytest.raises(ValueError, match="label order"):
        load_conll_splits()


def brat_span(text, surface):
    """Byte offsets of a surface's first occurrence, as the Quishpi annotations count them."""
    start = text.encode().index(surface.encode())
    return f"{start} {start + len(surface.encode())}\t{surface}"


def test_brat_sentences_tag_spans_and_split_sentences():
    text = "The clerk’s team checks the order, i.e. the form.\nIf it is late, reject it..\nDone\nNext step."
    annotations = "\n".join(
        [
            f"T1\tEntity {brat_span(text, 'order')}",
            f"T2\tAction {brat_span(text, 'checks')}",
            f"T3\tCondition {brat_span(text, 'it is late')}",
            f"T4\tEntity {brat_span(text, 'it')}",
            f"T5\tEntity {brat_span(text, 'order')}",
            "A1\tEvent T2",
            "R1\tSequence Arg1:T2 Arg2:T3",
        ]
    )

    sentences = brat_sentences("doc", text, annotations, QUISHPI_TAGS)

    tagged = [
        [
            (word, QUISHPI_TAGS[tag])
            for word, tag in zip(row["tokens"], row["ner_tags"], strict=True)
        ]
        for row in sentences
    ]
    assert tagged == [
        [
            ("The", "O"),
            ("clerk’s", "O"),
            ("team", "O"),
            ("checks", "B-Action"),
            ("the", "O"),
            ("order", "B-Entity"),
            (",", "O"),
            ("i.e.", "O"),
            ("the", "O"),
            ("form", "O"),
            (".", "O"),
        ],
        [
            ("If", "O"),
            ("it", "B-Condition"),
            ("is", "I-Condition"),
            ("late", "I-Condition"),
            (",", "O"),
            ("reject", "O"),
            ("it", "O"),
            (".", "O"),
            (".", "O"),
        ],
        [("Done", "O")],
        [("Next", "O"), ("step", "O"), (".", "O")],
    ]
    assert [row["sentence_id"] for row in sentences] == [0, 1, 2, 3]
    assert {row["document_name"] for row in sentences} == {"doc"}


def test_brat_sentences_reject_spans_that_miss_their_text():
    with pytest.raises(ValueError, match="does not match"):
        brat_sentences("doc", "Check the order.", "T1\tEntity 0 5\torder", QUISHPI_TAGS)


def test_quishpi_is_split_like_pet(tmp_path, monkeypatch):
    names = tuple(f"doc-{idx}" for idx in range(50))
    monkeypatch.setattr(data_prep, "QUISHPI_DOCUMENTS", names)
    for name in names:
        text = f"{name} acts."
        (tmp_path / "texts").mkdir(exist_ok=True)
        (tmp_path / "judgeannotations").mkdir(exist_ok=True)
        (tmp_path / "texts" / f"{name}.txt").write_text(text)
        (tmp_path / "judgeannotations" / f"{name}.ann").write_text(
            f"T1\tAction {brat_span(text, 'acts')}"
        )
    pet_path = tmp_path / "pet.jsonl"
    pet_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "document name": name,
                    "sentence-ID": 0,
                    "tokens": [name, "acts", "."],
                    "ner-tags": ["O", "O", "O"],
                }
            )
            for name in names
        )
    )

    seed, pool, gold, test = data_prep.load_quishpi_splits(tmp_path)
    pet_seed, pet_pool, _, pet_test = data_prep.load_pet_splits(pet_path)
    quarter = data_prep.load_quishpi_splits(tmp_path, pool_percent=25)

    assert (seed, pool, test) == (
        [{**row, "ner_tags": [0, 1, 0]} for row in pet_seed],
        pet_pool,
        [{**row, "ner_tags": [0, 1, 0]} for row in pet_test],
    )
    assert all("ner_tags" not in example for example in pool)
    assert {QUISHPI_TAGS[tag] for key, tag in gold.items() if key[1] == 1} == {"B-Action"}
    assert (quarter[0], quarter[1], quarter[3]) == (seed, pool[:9], test)


@pytest.mark.parametrize(
    ("dataset", "download", "loader", "repo", "revision"),
    [
        (
            "quishpi",
            "download_quishpi",
            "load_quishpi_splits",
            data_prep.QUISHPI_REPO,
            data_prep.QUISHPI_REVISION,
        ),
        (
            "medical",
            "download_medical",
            "load_medical_splits",
            data_prep.MEDICAL_REPO,
            data_prep.MEDICAL_REVISION,
        ),
    ],
)
def test_new_datasets_record_their_pinned_revision(
    monkeypatch, dataset, download, loader, repo, revision
):
    monkeypatch.setattr(data_prep, download, lambda: "files")
    monkeypatch.setattr(data_prep, loader, lambda files, pool_percent: (files,))
    assert load_splits(dataset, 50) == (
        ("files",),
        {"dataset": repo, "dataset_revision": revision, "dataset_percent": 50},
    )


def test_medical_tags_are_named_by_their_begin_id():
    assert MEDICAL_TAGS[:5] == ["O", "B-T03", "I-T03", "B-T05", "I-T05"]
    assert len(MEDICAL_TAGS) == 39
    named = [MEDICAL_TAGS[data_prep._medical_tag(raw)] for raw in (0, 3, 4, 43, 44)]
    assert named == ["O", "B-T03", "I-T03", "B-T43", "I-T43"]
    with pytest.raises(ValueError, match="tag 7"):
        data_prep._medical_tag(7)


def test_medical_joins_both_files_and_is_split_like_pet(tmp_path):
    names = [f"doc-{idx}" for idx in range(50)]
    rows = [
        {
            "document_unique_id": name,
            "sentence_id": 0,
            "tokens": [name, "scan"],
            "ner_tags": [0, 43],
        }
        for name in names
    ]
    train, test_file = tmp_path / "train.json", tmp_path / "test.json"
    train.write_text("\n".join(json.dumps(row) for row in rows[:40]))
    test_file.write_text("\n".join(json.dumps(row) for row in rows[40:]))
    pet_path = tmp_path / "pet.jsonl"
    pet_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "document name": name,
                    "sentence-ID": 0,
                    "tokens": [name, "scan"],
                    "ner-tags": ["O", "O"],
                }
            )
            for name in names
        )
    )

    seed, pool, gold, test = data_prep.load_medical_splits([train, test_file])
    pet_seed, pet_pool, _, pet_test = data_prep.load_pet_splits(pet_path)

    object_tag = MEDICAL_TAGS.index("B-T43")
    assert seed == [{**row, "ner_tags": [0, object_tag]} for row in pet_seed]
    assert pool == pet_pool
    assert test == [{**row, "ner_tags": [0, object_tag]} for row in pet_test]
    assert {tag for key, tag in gold.items() if key[1] == 1} == {object_tag}


def test_medical_explains_gated_access(monkeypatch):
    def gated(*args, **kwargs):
        request = httpx.Request("GET", "https://huggingface.co")
        raise GatedRepoError("401", response=httpx.Response(401, request=request))

    monkeypatch.setattr(data_prep, "hf_hub_download", gated)
    with pytest.raises(RuntimeError, match="accept its conditions"):
        data_prep.download_medical()


def test_load_splits_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_splits("ontonotes")


def test_full_pet_plans_stay_unchanged_and_other_pools_are_recorded():
    pet = search.sample_plan(RandomSearchConfig(num_configs=1))
    conll = search.sample_plan(
        RandomSearchConfig(num_configs=1, dataset="conll2003", dataset_percent=50)
    )
    assert not {"dataset", "dataset_percent"} & set(pet["fixed_config"])
    assert conll["fixed_config"]["dataset"] == "conll2003"
    assert conll["fixed_config"]["dataset_percent"] == 50

    assert "dataset" not in supervised.sample_plan(SupervisedConfig())["fixed_config"]
    conll_supervised = SupervisedConfig(dataset="conll2003", dataset_percent=12.5)
    assert supervised.sample_plan(conll_supervised)["fixed_config"]["dataset_percent"] == 12.5
    assert conll_supervised.sweep_name == "distilbert-base-cased-conll2003-12p5pct-supervised"
    assert SupervisedConfig().sweep_name == "distilbert-base-cased-supervised"

    assert "conll2003-50pct" in (
        ExperimentConfig(dataset="conll2003", dataset_percent=50).resolved_wandb_run_name()
    )
    assert "conll2003" not in ExperimentConfig().resolved_wandb_run_name()
    assert "quishpi" in ExperimentConfig(dataset="quishpi").resolved_wandb_run_name()
    assert "dataset_percent" not in ExperimentConfig().active_learning_kwargs()


def test_supervised_results_are_matched_by_dataset_and_percent(tmp_path):
    pools = (
        ("a_conll_half", {"dataset": "conll2003", "dataset_percent": 50.0}),
        ("b_conll", {"dataset": "conll2003"}),
        ("c_pet", {}),
    )
    for sweep, pool in pools:
        run_dir = tmp_path / sweep / "test" / "pool_100pct" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({"checkpoint": "bert", **pool}))
        pl.DataFrame({"seed": [0], "entity_f1": [0.5]}).write_csv(run_dir / "results.csv")
        (run_dir.parent / "completed.json").write_text(
            json.dumps({"run_dir": "test/pool_100pct/run"})
        )

    assert supervised.load_test_results(tmp_path, "bert")[0] == "c_pet"
    assert supervised.load_test_results(tmp_path, "bert", "conll2003")[0] == "b_conll"
    assert supervised.load_test_results(tmp_path, "bert", "conll2003", 50.0)[0] == "a_conll_half"


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
