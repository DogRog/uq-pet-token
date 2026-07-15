import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

import uq_pet.data as data_module
from uq_pet.config import NER_TAGS
from uq_pet.data import sentence_key, split_pool_test, tag_ids_to_labels

FEATURES = Features(
    {
        "document name": Value("string"),
        "sentence-ID": Value("int8"),
        "tokens": Sequence(Value("string")),
        "tokens-IDs": Sequence(Value("int8")),
        "ner-tags": Sequence(ClassLabel(names=NER_TAGS)),
    }
)


@pytest.fixture
def fake_dataset(monkeypatch):
    examples = [
        {
            "document name": f"doc{i % 3}",
            "sentence-ID": i,
            "tokens": ["word", str(i)],
            "tokens-IDs": [0, 1],
            "ner-tags": [0, 1],
        }
        for i in range(20)
    ]
    dataset = Dataset.from_list(examples, features=FEATURES)
    monkeypatch.setattr(data_module, "load_pet_ner", lambda: dataset)
    return dataset


def test_split_pool_test_deterministic(fake_dataset):
    pool_a, test_a = split_pool_test(seed=3407)
    pool_b, test_b = split_pool_test(seed=3407)
    assert [sentence_key(ex) for ex in pool_a] == [sentence_key(ex) for ex in pool_b]
    assert [sentence_key(ex) for ex in test_a] == [sentence_key(ex) for ex in test_b]


def test_split_pool_test_sizes(fake_dataset):
    pool, test = split_pool_test(seed=3407, test_size=0.2)
    assert len(pool) == 16
    assert len(test) == 4
    pool_keys = {sentence_key(ex) for ex in pool}
    test_keys = {sentence_key(ex) for ex in test}
    assert pool_keys.isdisjoint(test_keys)


def test_sentence_key_disambiguates_documents():
    a = {"document name": "doc1", "sentence-ID": 3}
    b = {"document name": "doc2", "sentence-ID": 3}
    assert sentence_key(a) != sentence_key(b)
    assert sentence_key(a) == "doc1::3"


def test_tag_ids_to_labels():
    assert tag_ids_to_labels([0, 1, 2]) == ["O", "B-Actor", "I-Actor"]
