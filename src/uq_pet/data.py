"""PET NER dataset download, loading, and pool/test splitting."""

import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset

from .config import NER_DATASET_URL, NER_TAGS, RAW_DATASET_PATH, SEED, TEST_SIZE


def download_pet_ner(dest: Path = RAW_DATASET_PATH, force: bool = False) -> Path:
    """Download the PET entities jsonl into data/raw/."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NER_DATASET_URL, dest)
    return dest


def load_pet_ner(path: Path = RAW_DATASET_PATH) -> Dataset:
    """Load the PET entities dataset (417 sentence-level examples) from data/raw/."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `uv run uq-pet download-data` first.")
    features = Features(
        {
            "document name": Value("string"),
            "sentence-ID": Value("int8"),
            "tokens": Sequence(Value("string")),
            "tokens-IDs": Sequence(Value("int8")),
            "ner-tags": Sequence(ClassLabel(names=NER_TAGS)),
        }
    )
    dataset = load_dataset("json", data_files={"full": str(path)}, features=features)
    return dataset["full"]


def split_pool_test(seed: int = SEED, test_size: float = TEST_SIZE) -> tuple[Dataset, Dataset]:
    """80/20 split: (experiment pool, held-out test)."""
    splits = load_pet_ner().train_test_split(test_size=test_size, seed=seed)
    return splits["train"], splits["test"]


def sentence_key(example: dict) -> str:
    """Unique sentence identifier; sentence-ID alone repeats across documents."""
    return f"{example['document name']}::{example['sentence-ID']}"


def tag_ids_to_labels(tag_ids: list) -> list[str]:
    return [NER_TAGS[tid] for tid in tag_ids]
