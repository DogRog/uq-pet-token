"""PET dataset identity, download, and stable experiment split."""

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
    return seed_examples, pool_inputs, pool_gold, test_examples
