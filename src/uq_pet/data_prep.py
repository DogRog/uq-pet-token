"""Dataset identity, download, and stable experiment splits for every supported dataset."""

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

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
QUISHPI_TAGS = ["O", "B-Action", "I-Action", "B-Entity", "I-Entity", "B-Condition", "I-Condition"]
# MedicalProcessInstruks stores unnamed integer tags: odd ids begin a type and the next id
# continues it. Only the types that occur are kept, named by their begin id.
MEDICAL_TYPE_IDS = (3, 5, 9, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43)
MEDICAL_TAGS = [
    "O",
    *(f"{prefix}-T{type_id:02d}" for type_id in MEDICAL_TYPE_IDS for prefix in ("B", "I")),
]
DATASET_TAGS = {
    "pet": NER_TAGS,
    "conll2003": CONLL_TAGS,
    "quishpi": QUISHPI_TAGS,
    "medical": MEDICAL_TAGS,
}

# The Parquet conversion of the relocated Hub dataset, pinned so splits cannot drift.
CONLL_REPO = "eriktks/conll2003"
CONLL_REVISION = "ce85b39f9dd99f552d0739d456814e95fb6a39b0"

# Quishpi et al.'s judge annotations of process texts, pinned at the bpm2020 branch head.
QUISHPI_REPO = "PADS-UPC/atdp-extractor"
QUISHPI_REVISION = "b9d34443a5ba145b9873217cefef0023f7e06dbf"
QUISHPI_DOCUMENTS = (
    "1-1_bicycle_manufacturing",
    "1-2_computer_repair",
    "10-2_process_b3",
    "1081511532_rev3",
    "1120589054_rev4",
    "1364308140_rev4",
    "2-1_sla_violation",
    "20818304_rev1",
    "3-1_2009-1_mc_finalice_sct_warranty_posession",
    "3-2_2009-2_conduct_directions_hearing",
    "3-6_2010-1_claims_notification",
    "4-1_intaker_workflow",
    "5-1_active_vos_tutorial",
    "6-1_acme",
    "7-1_calling_leads",
    "784358570_rev2",
    "8-1_hr_process_simple",
    "9-2_exercise_2",
)


# Gated: each user must accept its conditions on the Hub and log in before the first load.
MEDICAL_REPO = "dagrat/MedicalProcessInstruks"
MEDICAL_REVISION = "ee7a669935d77f6181058bd2caa8127d99e037a7"
MEDICAL_FILES = ("train_ner-v2.0.json", "test_ner-v2.0.json")


def _sentence_features(tags: list[str]) -> Features:
    return Features(
        {
            "document name": Value("string"),
            "sentence-ID": Value("int8"),
            "tokens": Sequence(Value("string")),
            "ner-tags": Sequence(ClassLabel(names=tags)),
        }
    )


FEATURES = _sentence_features(NER_TAGS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PETv1.1-entities.jsonl"
QUISHPI_DIR = PROJECT_ROOT / "data" / "raw" / "quishpi"
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
    pool_percent: float = 100.0,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Return labelled seed, label-free pool, private pool labels, and test data."""
    dataset = load_dataset("json", data_files={"full": str(path)}, features=FEATURES)["full"]
    return _split_sentences(
        dataset,
        n_seed_sentences=n_seed_sentences,
        split_seed=split_seed,
        seed_split_seed=seed_split_seed,
        pool_percent=pool_percent,
    )


def _split_sentences(
    dataset: Dataset,
    *,
    n_seed_sentences: int,
    split_seed: int,
    seed_split_seed: int,
    pool_percent: float,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    # PET's split: a seeded 80/20 train/test split, then seed sentences from the train part.
    outer = dataset.train_test_split(test_size=TEST_SIZE, seed=split_seed)
    inner = outer["train"].train_test_split(
        train_size=n_seed_sentences,
        shuffle=True,
        seed=seed_split_seed,
    )

    seed_examples = _materialize(inner["train"])
    raw_pool = _materialize(inner["test"])
    test_examples = _materialize(outer["test"])
    return seed_examples, *_label_free_pool(raw_pool, pool_percent), test_examples


def _label_free_pool(
    raw_pool: list[dict], pool_percent: float
) -> tuple[list[dict], dict[TokenKey, int]]:
    # The pool is already shuffled, so smaller percentages are nested prefixes.
    if not 0 < pool_percent <= 100:
        raise ValueError(f"pool_percent must be in (0, 100], got {pool_percent}")
    count = round(len(raw_pool) * pool_percent / 100)
    if count < 1:
        raise ValueError(f"{pool_percent:g}% of {len(raw_pool)} pool sentences is empty")
    raw_pool = raw_pool[:count]
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
    pool_percent: float = 100.0,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Draw seed and pool from CoNLL train; evaluate on the full CoNLL test split."""
    dataset = load_dataset(CONLL_REPO, revision=CONLL_REVISION)
    if dataset["train"].features["ner_tags"].feature.names != CONLL_TAGS:
        raise ValueError("CoNLL-2003 label order differs from CONLL_TAGS")
    train = dataset["train"].shuffle(seed=split_seed)
    seed_examples = _materialize_conll(train.select(range(n_seed_sentences)), "train")
    raw_pool = _materialize_conll(train.select(range(n_seed_sentences, len(train))), "train")
    test_examples = _materialize_conll(dataset["test"], "test")
    return seed_examples, *_label_free_pool(raw_pool, pool_percent), test_examples


def download_quishpi(directory: Path = QUISHPI_DIR) -> Path:
    """Download the pinned Quishpi texts and annotations, keeping files already present."""
    base_url = f"https://raw.githubusercontent.com/{QUISHPI_REPO}/{QUISHPI_REVISION}/input"
    for name in QUISHPI_DOCUMENTS:
        for folder, suffix in (("texts", ".txt"), ("judgeannotations", ".ann")):
            path = directory / folder / f"{name}{suffix}"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{base_url}/{folder}/{name}{suffix}", path)
    return directory


# Dotted abbreviations such as "i.e." stay one word so they do not end a sentence.
WORD_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}|\w+(?:['’-]\w+)*|[^\w\s]")
SENTENCE_ENDS = {".", "?", "!"}


def brat_sentences(document_name: str, text: str, annotations: str, tags: list[str]) -> list[dict]:
    """Split one brat document into word sentences with BIO tags from its T spans.

    Offsets are read as UTF-8 byte offsets, which is how the Quishpi annotations count
    them; every span must then match its surface text exactly. Attributes and relations
    are ignored. Longer spans claim words first, so a span overlapping a longer one is
    dropped. Sentences end after a run of ".", "?", or "!" and at line breaks.
    """
    encoded = text.encode()
    spans = set()
    for line in annotations.splitlines():
        if not line.startswith("T"):
            continue
        _, info, surface = line.split("\t")
        entity_type, start, end = info.split(" ")
        start = len(encoded[: int(start)].decode())
        end = len(encoded[: int(end)].decode())
        if text[start:end] != surface:
            raise ValueError(f"{document_name}: {line!r} does not match its text")
        spans.add((start, end, entity_type))

    words = [(match.start(), match.end(), match.group()) for match in WORD_PATTERN.finditer(text)]
    labels = ["O"] * len(words)
    for start, end, entity_type in sorted(spans, key=lambda span: (span[0] - span[1], span[0])):
        covered = [index for index, word in enumerate(words) if word[0] < end and word[1] > start]
        if not covered or any(labels[index] != "O" for index in covered):
            continue
        for index in covered:
            labels[index] = f"I-{entity_type}"
        labels[covered[0]] = f"B-{entity_type}"

    sentences: list[list[int]] = []
    for index, (start, _, word) in enumerate(words):
        previous = words[index - 1] if index else None
        if (
            previous is None
            or (previous[2] in SENTENCE_ENDS and word not in SENTENCE_ENDS)
            or "\n" in text[previous[1] : start]
        ):
            sentences.append([])
        sentences[-1].append(index)
    return [
        {
            "document_name": document_name,
            "sentence_id": sentence_id,
            "tokens": [words[index][2] for index in sentence],
            "ner_tags": [tags.index(labels[index]) for index in sentence],
        }
        for sentence_id, sentence in enumerate(sentences)
    ]


def load_quishpi_splits(
    directory: Path = QUISHPI_DIR,
    *,
    n_seed_sentences: int = N_SEED_SENTENCES,
    split_seed: int = SEED,
    seed_split_seed: int = SEED_SPLIT_SEED,
    pool_percent: float = 100.0,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Split Quishpi's sentences exactly as PET's are split."""
    rows = [
        {
            "document name": sentence["document_name"],
            "sentence-ID": sentence["sentence_id"],
            "tokens": sentence["tokens"],
            "ner-tags": sentence["ner_tags"],
        }
        for name in QUISHPI_DOCUMENTS
        for sentence in brat_sentences(
            name,
            (directory / "texts" / f"{name}.txt").read_text(encoding="utf-8"),
            (directory / "judgeannotations" / f"{name}.ann").read_text(encoding="utf-8"),
            QUISHPI_TAGS,
        )
    ]
    return _split_sentences(
        Dataset.from_list(rows, features=_sentence_features(QUISHPI_TAGS)),
        n_seed_sentences=n_seed_sentences,
        split_seed=split_seed,
        seed_split_seed=seed_split_seed,
        pool_percent=pool_percent,
    )


def download_medical() -> list[Path]:
    """Download the pinned MedicalProcessInstruks NER files into the Hub cache."""
    try:
        return [
            Path(
                hf_hub_download(MEDICAL_REPO, name, repo_type="dataset", revision=MEDICAL_REVISION)
            )
            for name in MEDICAL_FILES
        ]
    except GatedRepoError as error:
        raise RuntimeError(
            f"{MEDICAL_REPO} is gated: accept its conditions at "
            f"https://huggingface.co/datasets/{MEDICAL_REPO} and run `uv run hf auth login`"
        ) from error


def _medical_tag(raw_tag: int) -> int:
    if raw_tag == 0:
        return 0
    type_id = raw_tag if raw_tag % 2 else raw_tag - 1
    name = f"{'B' if raw_tag % 2 else 'I'}-T{type_id:02d}"
    if name not in MEDICAL_TAGS:
        raise ValueError(f"MedicalProcessInstruks tag {raw_tag} is not in MEDICAL_TAGS")
    return MEDICAL_TAGS.index(name)


def load_medical_splits(
    paths: list[Path],
    *,
    n_seed_sentences: int = N_SEED_SENTENCES,
    split_seed: int = SEED,
    seed_split_seed: int = SEED_SPLIT_SEED,
    pool_percent: float = 100.0,
) -> tuple[list[dict], list[dict], dict[TokenKey, int], list[dict]]:
    """Join the released train and test files and split their sentences as PET's are."""
    rows = [
        {
            "document name": row["document_unique_id"],
            "sentence-ID": row["sentence_id"],
            "tokens": row["tokens"],
            "ner-tags": [_medical_tag(tag) for tag in row["ner_tags"]],
        }
        for path in paths
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]
    return _split_sentences(
        Dataset.from_list(rows, features=_sentence_features(MEDICAL_TAGS)),
        n_seed_sentences=n_seed_sentences,
        split_seed=split_seed,
        seed_split_seed=seed_split_seed,
        pool_percent=pool_percent,
    )


def load_splits(dataset: str, pool_percent: float = 100.0) -> tuple[tuple, dict]:
    """Return a dataset's experiment splits and the identity recorded for resumes."""
    if dataset == "pet":
        path = download_pet_ner()
        splits = load_pet_splits(path, pool_percent=pool_percent)
        identity = {"dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    elif dataset == "conll2003":
        splits = load_conll_splits(pool_percent=pool_percent)
        identity = {"dataset": CONLL_REPO, "dataset_revision": CONLL_REVISION}
    elif dataset == "quishpi":
        splits = load_quishpi_splits(download_quishpi(), pool_percent=pool_percent)
        identity = {"dataset": QUISHPI_REPO, "dataset_revision": QUISHPI_REVISION}
    elif dataset == "medical":
        splits = load_medical_splits(download_medical(), pool_percent=pool_percent)
        identity = {"dataset": MEDICAL_REPO, "dataset_revision": MEDICAL_REVISION}
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    if pool_percent != 100:
        identity["dataset_percent"] = pool_percent
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
