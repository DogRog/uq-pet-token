import json

import pytest
from datasets import Dataset

from uq_pet.config import RAW_DATASET_PATH
from uq_pet.dataset import (
    FEATURES,
    sentence_key,
    split_dataset,
    tag_ids_to_labels,
    to_examples,
)

needs_raw_data = pytest.mark.skipif(
    not RAW_DATASET_PATH.exists(), reason="data/raw/PETv1.1-entities.jsonl not downloaded"
)


def test_tag_ids_to_labels():
    assert tag_ids_to_labels([0, 1, 2]) == ["O", "B-Actor", "I-Actor"]


def test_sentence_key_disambiguates_documents(sample_examples):
    assert sentence_key(sample_examples[0]) == "doc-1::3"
    assert sentence_key(sample_examples[1]) == "doc-2::3"


def test_sentence_keys_are_unique(sample_examples):
    keys = [sentence_key(ex) for ex in sample_examples]
    assert len(set(keys)) == len(keys)


def test_to_examples_returns_json_serializable_python_types(sample_examples):
    ds = Dataset.from_list(sample_examples, features=FEATURES)
    examples = to_examples(ds)

    assert examples == sample_examples
    json.dumps(examples)  # would raise on numpy scalars
    for ex in examples:
        assert isinstance(ex["sentence-ID"], int)
        assert all(isinstance(t, int) for t in ex["ner-tags"])
        assert all(isinstance(t, str) for t in ex["tokens"])


@needs_raw_data
def test_split_sizes_and_first_key():
    """Canary for the idx-keyed LLM cache.

    Cache records are keyed by position in `pool`. If a datasets upgrade or a change
    to SEED/TEST_SIZE/N_FEW_SHOT_EXAMPLES reorders the split, every cached record
    silently points at the wrong sentence — this test fails first.
    """
    few_shot, pool, test = split_dataset()
    assert (len(few_shot), len(pool), len(test)) == (5, 328, 84)
    assert sentence_key(to_examples(pool)[0]) == "doc-1.1::9"


@needs_raw_data
def test_split_is_deterministic_and_disjoint():
    few_shot, pool, test = split_dataset()
    again = split_dataset()
    assert [sentence_key(e) for e in to_examples(pool)] == [
        sentence_key(e) for e in to_examples(again[1])
    ]

    keys = {name: {sentence_key(e) for e in to_examples(ds)} for name, ds in
            (("few_shot", few_shot), ("pool", pool), ("test", test))}
    assert not keys["few_shot"] & keys["pool"]
    assert not keys["pool"] & keys["test"]
    assert not keys["few_shot"] & keys["test"]
