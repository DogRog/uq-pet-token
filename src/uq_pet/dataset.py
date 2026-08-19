"""PET NER dataset download, loading, and seed/pool/test splitting."""

import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset

from uq_pet.config import (
    N_SEED_EXAMPLES,
    NER_DATASET_URL,
    NER_TAGS,
    RAW_DATASET_PATH,
    SEED,
    SEED_SPLIT_SEED,
    TEST_SIZE,
)

FEATURES = Features(
    {
        "document name": Value("string"),
        "sentence-ID": Value("int8"),
        "tokens": Sequence(Value("string")),
        "ner-tags": Sequence(ClassLabel(names=NER_TAGS)),
    }
)


def download_pet_ner(raw_data_path: Path = RAW_DATASET_PATH, force: bool = False) -> Path:
    """Download the PET entities jsonl into data/raw/."""
    if raw_data_path.exists() and not force:
        return raw_data_path
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NER_DATASET_URL, raw_data_path)
    return raw_data_path


def load_pet_ner(path: Path = RAW_DATASET_PATH) -> Dataset:
    """Load the PET entities dataset (417 sentence-level examples) from data/raw/."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — it is downloaded on the first pipeline run.")
    return load_dataset("json", data_files={"full": str(path)}, features=FEATURES)["full"]


def split_dataset(
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    n_seed: int = N_SEED_EXAMPLES,
    seed_split_seed: int = SEED_SPLIT_SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split PET into labelled seed examples, experiment pool, and held-out test.

    Returns (seed_set, pool, test). The seed set is the only labelled data available
    before pool selection. With the defaults: 5 / 328 / 84.
    """
    outer = load_pet_ner().train_test_split(test_size=test_size, seed=seed)
    inner = outer["train"].train_test_split(train_size=n_seed, shuffle=True, seed=seed_split_seed)
    return inner["train"], inner["test"], outer["test"]


def to_examples(dataset: Dataset) -> list[dict]:
    """Materialize a Dataset as plain JSON-serializable dicts.

    Called once per split at the top of the pipeline so that nothing downstream has
    to care about arrow slices or numpy scalars — selection records get written to
    selection.json, which json.dump would otherwise reject.
    """
    return [
        {
            "document name": str(row["document name"]),
            "sentence-ID": int(row["sentence-ID"]),
            "tokens": [str(t) for t in row["tokens"]],
            "ner-tags": [int(t) for t in row["ner-tags"]],
        }
        for row in dataset.to_list()
    ]


def sentence_key(example: dict) -> str:
    """Stable identifier for one sentence: sentence-ID repeats across documents."""
    return f"{example['document name']}::{example['sentence-ID']}"


def tag_ids_to_labels(tag_ids: list[int]) -> list[str]:
    return [NER_TAGS[tid] for tid in tag_ids]
