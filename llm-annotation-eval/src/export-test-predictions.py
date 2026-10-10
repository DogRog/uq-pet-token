"""Writes test-predictions.jsonl next to every pet.jsonl in res/pred/, for uq-pet-token.

Each file holds the test split of uq-pet-token for the run's dataset (e.g. the 84 sentences of
res/gold/pet-v1.1-test.jsonl for PET, see eval.test_split) in the same format and order, with the
predicted BIO tags. Predictions that don't cover all test sentences are skipped. run-configs.py
writes the file for new runs itself.

    uv run python src/export-test-predictions.py
"""
import json
import pathlib

from eval import datasets, test_split
from experiments import runner

REPO_DIR = pathlib.Path(__file__).parent.parent
PRED_DIR = REPO_DIR / "res" / "pred"


def dataset_of(predictions_file: pathlib.Path) -> datasets.Dataset:
    """The dataset in the run's run.json; predictions from before the configs are PET's."""
    manifest_file = predictions_file.parent.parent / "run.json"
    if not manifest_file.exists():
        return datasets.DATASETS["pet"]
    manifest = json.loads(manifest_file.read_text(encoding="utf8"))
    return datasets.DATASETS[runner.manifest_settings(manifest)["dataset"]]


def main():
    tests = {}
    # runs of a config: <folder>/run-<i>/<task>/pet.jsonl; earlier predictions: <folder>/<task>/
    files = sorted(PRED_DIR.glob("*/run-*/*/pet.jsonl")) + sorted(PRED_DIR.glob("*/*/pet.jsonl"))
    written, skipped = 0, []
    for predictions_file in files:
        dataset = dataset_of(predictions_file)
        if dataset.name not in tests:
            tests[dataset.name] = test_split.load_test_split(dataset.test_file)
        if test_split.export(predictions_file, tests[dataset.name], dataset.types) is None:
            skipped.append(str(predictions_file.relative_to(REPO_DIR)))
        else:
            written += 1
    print(f"Wrote test-predictions.jsonl for {written} prediction files.")
    if skipped:
        print("Skipped, as they don't cover the test split:", *skipped, sep="\n  ")


if __name__ == "__main__":
    main()
