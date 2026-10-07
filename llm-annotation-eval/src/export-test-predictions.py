"""Writes test-predictions.jsonl next to every pet.jsonl in res/pred/, for uq-pet-token.

Each file holds the 84 sentences of uq-pet-token's PET test split (res/gold/pet-v1.1-test.jsonl,
see eval.test_split) in the same format and order, with the predicted BIO tags. Predictions that
don't cover all test sentences are skipped. run-configs.py writes the file for new runs itself.

    uv run python src/export-test-predictions.py
"""
import pathlib

from eval import test_split

REPO_DIR = pathlib.Path(__file__).parent.parent
TEST_FILE = REPO_DIR / "res" / "gold" / "pet-v1.1-test.jsonl"
PRED_DIR = REPO_DIR / "res" / "pred"


def main():
    test = test_split.load_test_split(TEST_FILE)
    # runs of a config: <folder>/run-<i>/<task>/pet.jsonl; earlier predictions: <folder>/<task>/
    files = sorted(PRED_DIR.glob("*/run-*/*/pet.jsonl")) + sorted(PRED_DIR.glob("*/*/pet.jsonl"))
    written, skipped = 0, []
    for predictions_file in files:
        if test_split.export(predictions_file, test) is None:
            skipped.append(str(predictions_file.relative_to(REPO_DIR)))
        else:
            written += 1
    print(f"Wrote test-predictions.jsonl for {written} prediction files.")
    if skipped:
        print("Skipped, as they don't cover the test split:", *skipped, sep="\n  ")


if __name__ == "__main__":
    main()
