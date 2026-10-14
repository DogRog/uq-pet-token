"""Writes the Quishpi and MedicalProcessInstruks sentences and test splits of uq-pet-token.

Runs in uq-pet-token's environment, so the sentences are cut and split by its own code
(``uq_pet.data_prep``) with its pinned datasets version:

    uv run --project ../uq-pet-token python scripts/uq-pet-token-splits.py

For each dataset it writes ``res/gold/<dataset>.jsonl`` (every sentence, in the dataset's order)
and ``res/gold/<dataset>-test.jsonl`` (the test split, in the split's order), one sentence per
line with BIO tags, in the format of PETv1.1-entities.jsonl. MedicalProcessInstruks is gated, so
its files are not tracked by git; log in with ``uv run --project ../uq-pet-token hf auth login``
first. PET's files are checked, not written: they must equal what uq-pet-token splits now.

With ``--check`` nothing is written: every dataset's files are compared with what uq-pet-token
splits now, and the exit code is 1 if one differs (``src/check-uq-pet-token-data.py`` runs this).
A dataset whose files are missing is skipped then.
"""
import argparse
import json
import pathlib
import subprocess
import sys

from uq_pet import data_prep

GOLD_DIR = pathlib.Path(__file__).resolve().parent.parent / "res" / "gold"


def rows(examples: list[dict], tags: list[str]) -> list[dict]:
    return [
        {
            "document name": e["document_name"],
            "sentence-ID": e["sentence_id"],
            "tokens": e["tokens"],
            "ner-tags": [tags[t] for t in e["ner_tags"]],
        }
        for e in examples
    ]


def split(dataset: str) -> tuple[list[dict], list[dict]]:
    """Every sentence in the dataset's order, and the test split in the split's order."""
    tags = data_prep.DATASET_TAGS[dataset]
    (seed, _, pool_gold, test), _ = data_prep.load_splits(dataset)
    if dataset == "pet":
        path = data_prep.download_pet_ner()
        everything = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    elif dataset == "quishpi":
        directory = data_prep.download_quishpi()
        everything = [
            sentence
            for name in data_prep.QUISHPI_DOCUMENTS
            for sentence in rows(
                data_prep.brat_sentences(
                    name,
                    (directory / "texts" / f"{name}.txt").read_text(encoding="utf-8"),
                    (directory / "judgeannotations" / f"{name}.ann").read_text(encoding="utf-8"),
                    tags,
                ),
                tags,
            )
        ]
    else:
        everything = [
            {
                "document name": row["document_unique_id"],
                "sentence-ID": row["sentence_id"],
                "tokens": row["tokens"],
                "ner-tags": [tags[data_prep._medical_tag(t)] for t in row["ner_tags"]],
            }
            for path in data_prep.download_medical()
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        ]
    test_rows = rows(test, tags)
    # the split only reorders the sentences
    keys = [(r["document name"], r["sentence-ID"]) for r in everything]
    assert len(set(keys)) == len(keys)
    assert len(seed) + len({i for i, _ in pool_gold}) + len(test_rows) == len(everything)
    by_key = dict(zip(keys, everything))
    assert all(by_key[(r["document name"], r["sentence-ID"])] == r for r in test_rows)
    return everything, test_rows


def write(path: pathlib.Path, sentences: list[dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        for row in sentences:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


FILES = {
    "pet": ("pet-v1.1.jsonl", "pet-v1.1-test.jsonl"),
    "quishpi": ("quishpi.jsonl", "quishpi-test.jsonl"),
    "medical": ("medical.jsonl", "medical-test.jsonl"),
}


def check() -> bool:
    """Compares every dataset's files with what uq-pet-token splits now; True if all are equal."""
    same = True
    for dataset, names in FILES.items():
        paths = [GOLD_DIR / name for name in names]
        if not all(path.exists() for path in paths):
            print(f"{dataset}: skipped, {', '.join(names)} missing")
            continue
        for path, rebuilt in zip(paths, split(dataset)):
            ours = read(path)
            differing = [i for i, (a, b) in enumerate(zip(ours, rebuilt)) if a != b]
            if ours == rebuilt:
                print(f"{dataset}: {path.name} unchanged, {len(ours)} sentences")
            else:
                same = False
                print(
                    f"{dataset}: {path.name} DIFFERS, {len(ours)} sentences here, "
                    f"{len(rebuilt)} in uq-pet-token, first differing lines {differing[:5]}"
                )
    return same


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="compare the files, write nothing"
    )
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=data_prep.PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if args.check:
        same = check()
        print(f"uq-pet-token commit {commit}")
        sys.exit(0 if same else 1)
    everything, test = split("pet")
    assert everything == read(GOLD_DIR / "pet-v1.1.jsonl"), "PET v1.1 differs from uq-pet-token's"
    assert test == read(GOLD_DIR / "pet-v1.1-test.jsonl"), "PET's test split differs"
    print(f"pet: {len(everything)} sentences, {len(test)} test, unchanged")
    for dataset in ("quishpi", "medical"):
        everything, test = split(dataset)
        full_name, test_name = FILES[dataset]
        write(GOLD_DIR / full_name, everything)
        write(GOLD_DIR / test_name, test)
        print(f"{dataset}: {len(everything)} sentences, {len(test)} test")
    print(f"uq-pet-token commit {commit}")


if __name__ == "__main__":
    main()
