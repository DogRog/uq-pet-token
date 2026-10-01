"""Dataset identity, download, and stable experiment splits for PET and CoNLL-2003."""

import hashlib
import random
import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset

SEED = 3407
TEST_SIZE = 0.2
SEED_SPLIT_SEED = 42
N_SEED_SENTENCES = 5

NER_TAGS = [
    "O",
    "B-Actor",
    "I-Actor",
    "B-Activity",
    "I-Activity",
    "B-Activity Data",
    "I-Activity Data",
    "B-Further Specification",
    "I-Further Specification",
    "B-XOR Gateway",
    "I-XOR Gateway",
    "B-Condition Specification",
    "I-Condition Specification",
    "B-AND Gateway",
    "I-AND Gateway",
]

CONLL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
DATASET_TAGS = {"pet": NER_TAGS, "conll2003": CONLL_TAGS}

# The Parquet conversion of the relocated Hub dataset, pinned so splits cannot drift.
CONLL_REPO = "eriktks/conll2003"
CONLL_REVISION = "ce85b39f9dd99f552d0739d456814e95fb6a39b0"

FEATURES = Features(
    {
        "document name": Value("string"),
        "sentence-ID": Value("int8"),
        "tokens": Sequence(Value("string")),
        "ner-tags": Sequence(ClassLabel(names=NER_TAGS)),
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PETv1.1-entities.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_URL = (
    "https://raw.githubusercontent.com/patriziobellan86/PETv1.1/master/PETv1.1-entities.jsonl"
)

TokenKey = tuple[int, int]


def download_pet_ner(path: Path = RAW_DATA_PATH) -> Path:
    """Download PET once, returning the existing file on later calls."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)
    return path


def _materialize(dataset: Dataset) -> list[dict]:
    return [
        {
            "document_name": str(row["document name"]),
            "sentence_id": int(row["sentence-ID"]),
            "tokens": [str(token) for token in row["tokens"]],
            "ner_tags": [int(tag) for tag in row["ner-tags"]],
        }
        for row in dataset.to_list()
    ]


def load_pet_splits(
    path: Path = RAW_DATA_PATH,
    *,
    n_seed_sentences: int = N_SEED_SENTENCES,
    split_seed: int = SEED,
    seed_split_seed: int = SEED_SPLIT_SEED,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Return labelled seed, label-free pool, private pool labels, and test data."""
    dataset = load_dataset("json", data_files={"full": str(path)}, features=FEATURES)["full"]
    outer = dataset.train_test_split(test_size=TEST_SIZE, seed=split_seed)
    inner = outer["train"].train_test_split(
        train_size=n_seed_sentences,
        shuffle=True,
        seed=seed_split_seed,
    )

    seed_examples = _materialize(inner["train"])
    raw_pool = _materialize(inner["test"])
    test_examples = _materialize(outer["test"])
    return seed_examples, *_label_free_pool(raw_pool), test_examples


def _label_free_pool(raw_pool: list[dict]) -> tuple[list[dict], dict[TokenKey, int]]:
    pool_inputs = [
        {
            "pool_idx": pool_idx,
            "document_name": example["document_name"],
            "sentence_id": example["sentence_id"],
            "tokens": example["tokens"],
        }
        for pool_idx, example in enumerate(raw_pool)
    ]
    pool_gold = {
        (pool_idx, word_idx): tag
        for pool_idx, example in enumerate(raw_pool)
        for word_idx, tag in enumerate(example["ner_tags"])
    }
    return pool_inputs, pool_gold


def _materialize_conll(dataset: Dataset, split: str) -> list[dict]:
    return [
        {
            "document_name": split,
            "sentence_id": int(row["id"]),
            "tokens": [str(token) for token in row["tokens"]],
            "ner_tags": [int(tag) for tag in row["ner_tags"]],
        }
        for row in dataset.to_list()
    ]


def load_conll_splits(
    *,
    n_seed_sentences: int = N_SEED_SENTENCES,
    split_seed: int = SEED,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Draw seed and pool from CoNLL train; evaluate on the full CoNLL test split."""
    dataset = load_dataset(CONLL_REPO, revision=CONLL_REVISION)
    if dataset["train"].features["ner_tags"].feature.names != CONLL_TAGS:
        raise ValueError("CoNLL-2003 label order differs from CONLL_TAGS")
    train = dataset["train"].shuffle(seed=split_seed)
    seed_examples = _materialize_conll(train.select(range(n_seed_sentences)), "train")
    raw_pool = _materialize_conll(train.select(range(n_seed_sentences, len(train))), "train")
    test_examples = _materialize_conll(dataset["test"], "test")
    return seed_examples, *_label_free_pool(raw_pool), test_examples


def load_splits(dataset: str) -> tuple[tuple, dict]:
    """Return a dataset's experiment splits and the identity recorded for resumes."""
    if dataset == "pet":
        path = download_pet_ner()
        splits = load_pet_splits(path)
        identity = {"dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    elif dataset == "conll2003":
        splits = load_conll_splits()
        identity = {"dataset": CONLL_REPO, "dataset_revision": CONLL_REVISION}
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return splits, identity


def split_tuning_pool(
    pool_inputs: list[dict],
    pool_gold: dict[TokenKey, int],
    *,
    validation_sentences: int,
    validation_seed: int,
) -> tuple[list[dict], dict[TokenKey, int], list[dict], dict]:
    """Hold out fixed pool sentences for tuning; never accept the outer test set.

    Choose sentence indices without labels, then materialize validation labels.
    Acquisition inputs are reindexed and remain label-free. The manifest maps
    those indices back to the original pool used for the final comparison.
    """
    if not 0 < validation_sentences < len(pool_inputs):
        raise ValueError("validation_sentences must leave a nonempty acquisition pool")
    held_out = set(
        random.Random(validation_seed).sample(range(len(pool_inputs)), validation_sentences)
    )
    train_indices = [index for index in range(len(pool_inputs)) if index not in held_out]
    validation_indices = sorted(held_out)
    inputs = [
        {**pool_inputs[original], "pool_idx": index} for index, original in enumerate(train_indices)
    ]
    gold = {
        (index, word): pool_gold[(original, word)]
        for index, original in enumerate(train_indices)
        for word in range(len(pool_inputs[original]["tokens"]))
    }
    validation = [
        {
            **{key: value for key, value in pool_inputs[index].items() if key != "pool_idx"},
            "ner_tags": [
                pool_gold[(index, word)] for word in range(len(pool_inputs[index]["tokens"]))
            ],
        }
        for index in validation_indices
    ]
    return (
        inputs,
        gold,
        validation,
        {
            "acquisition_original_pool_indices": train_indices,
            "validation_original_pool_indices": validation_indices,
        },
    )
