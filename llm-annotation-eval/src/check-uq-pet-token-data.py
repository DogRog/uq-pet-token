"""Checks that results here and uq-pet-token's are on the same data and scored the same way.

    uv run python src/check-uq-pet-token-data.py [--uq-pet-token ../uq-pet-token] [--no-rebuild]

1. Splits: rebuilds every dataset's sentences and test split with uq-pet-token's own code and
   compares them with res/gold/ (scripts/uq-pet-token-splits.py --check, run in uq-pet-token's
   environment; it downloads the datasets once, the medical one needs a Hugging Face login).
   ``--no-rebuild`` skips this.
2. Reported test gold: every test run of uq-pet-token's supervised sweeps
   (results/supervised/<sweep>/test/*/*/predictions.npz, read by the results notebook) stores the
   gold tags and sentence lengths it was scored on; they must equal res/gold/<dataset>-test.jsonl.
3. Scoring: eval.test_split.evaluate on those runs' stored predictions must reproduce each model
   seed's metrics in their results.csv.
4. Runs here: the gold file every session of every run annotated and was scored against (its
   sha256 in run.json) must be today's file. Prediction folders without run.json can't be checked
   and are listed.
5. Prompts: no test sentence of five words or more appears in a dataset's prompt.

Also reports, without failing, the PET test sentences whose labels in res/gold/pet.jsonl (what PET
whole-document runs annotate) differ from PET v1.1, as only test scores use the latter.

The exit code is 1 if a check fails.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import typing

import numpy as np

from eval import datasets, test_split
from experiments import definitions

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
PRED_DIR = REPO_DIR / "res" / "pred"

# the metrics uq-pet-token's results.csv has per model seed, as eval.test_split.evaluate names them
METRICS = (
    "entity_f1", "entity_macro_f1", "entity_precision", "entity_recall", "entity_partial_f1",
    "entity_partial_macro_f1", "entity_partial_precision", "entity_partial_recall",
    "token_accuracy",
)
# test sentences with fewer words (not counting punctuation, e.g. headings like "# # Diabetes # #")
# may well occur in a prompt as a mention text
MIN_LEAK_WORDS = 5


class Checks:
    def __init__(self):
        self.failed: typing.List[str] = []

    def report(self, ok: bool, check: str, message: str):
        print(f"  {'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            self.failed.append(f"{check}: {message}")


def available(dataset: datasets.Dataset) -> bool:
    return dataset.sentences_file.exists() and dataset.test_file.exists()


def check_rebuild(checks: Checks, uq_dir: pathlib.Path):
    print("1. Splits rebuilt with uq-pet-token's code")
    script = REPO_DIR / "scripts" / "uq-pet-token-splits.py"
    result = subprocess.run(
        ["uv", "run", "--project", str(uq_dir), "python", str(script), "--check"],
        cwd=uq_dir, capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        print(f"     {line}")
    if result.returncode not in (0, 1):
        print(result.stderr[-2000:])
    checks.report(result.returncode == 0, "splits", "res/gold/ equals uq-pet-token's splits")


def stored_runs(uq_dir: pathlib.Path) -> typing.Iterator[typing.Tuple[str, str, pathlib.Path]]:
    """(sweep, dataset, predictions.npz) of every test run of uq-pet-token's supervised sweeps."""
    for config_file in sorted(uq_dir.glob("results/supervised/*/run_config.json")):
        dataset = json.loads(config_file.read_text(encoding="utf8"))["dataset"]
        for npz in sorted(config_file.parent.glob("test/*/*/predictions.npz")):
            yield config_file.parent.name, dataset, npz


def check_stored_runs(checks: Checks, uq_dir: pathlib.Path):
    print("2. Gold of uq-pet-token's reported test runs, and 3. scoring their predictions")
    runs = list(stored_runs(uq_dir))
    checks.report(
        bool(runs), "reported runs", f"{len(runs)} test runs in {uq_dir}/results/supervised/"
    )
    tests: typing.Dict[str, typing.List[test_split.Row]] = {}
    for sweep, name, npz in runs:
        dataset = datasets.DATASETS[name]
        if not available(dataset):
            print(f"  skip {sweep}: {dataset.test_file.name} missing")
            continue
        test = tests.setdefault(name, test_split.load_test_split(dataset.test_file))
        stored = np.load(npz, allow_pickle=True)
        labels = [str(label) for label in stored["labels"]]
        gold = [labels[i] for i in stored["gold"].tolist()]
        same_gold = (
            stored["sentence_lengths"].tolist() == [len(r["tokens"]) for r in test]
            and gold == [tag for r in test for tag in r["ner-tags"]]
        )
        checks.report(
            same_gold, "reported gold",
            f"{sweep}: test gold equals {dataset.test_file.name} ({len(test)} sentences, "
            f"{len(gold)} tokens)",
        )
        if not same_gold:
            continue
        seeds = list(csv.DictReader(open(npz.parent / "results.csv", encoding="utf8")))
        largest = 0.0
        for seed, row in enumerate(seeds):
            flat = [labels[i] for i in stored["predicted"][seed].tolist()]
            predicted, start = [], 0
            for r in test:
                predicted.append({**r, "ner-tags": flat[start:start + len(r["tokens"])]})
                start += len(r["tokens"])
            scores = test_split.evaluate(test, predicted)
            largest = max(largest, *(abs(scores[m] - float(row[m])) for m in METRICS))
        checks.report(
            largest < 1e-9, "scoring",
            f"{sweep}: {len(seeds)} model seeds rescored, largest difference to results.csv "
            f"{largest:.1e}",
        )


def check_runs_here(checks: Checks):
    print("4. Gold files of the runs here")
    current = {
        p.relative_to(REPO_DIR).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (REPO_DIR / "res" / "gold").glob("*.jsonl")
    }
    sessions, stale = 0, []
    for manifest_file in sorted(PRED_DIR.glob("*/run-*/run.json")):
        for session in json.loads(manifest_file.read_text(encoding="utf8"))["sessions"]:
            sessions += 1
            if current.get(session["gold_file"]) != session["gold_sha256"]:
                run = manifest_file.parent.relative_to(REPO_DIR)
                stale.append(f"{run} ({session['gold_file']})")
    checks.report(
        not stale, "runs here",
        f"{sessions} sessions in {len(list(PRED_DIR.glob('*/run-*/run.json')))} runs used today's "
        f"gold files" + "".join(f"\n         differs: {s}" for s in stale),
    )
    unrecorded = sorted({
        p.parent.parent.relative_to(REPO_DIR).as_posix() for p in PRED_DIR.glob("*/*/pet.jsonl")
        if not p.parent.parent.name.startswith("run-")
    })
    if unrecorded:
        print("  note predictions without run.json, gold unknown:", ", ".join(unrecorded))


def normalized(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def check_prompts(checks: Checks):
    print("5. Test sentences in the prompts")
    for dataset in datasets.DATASETS.values():
        if not available(dataset):
            print(f"  skip {dataset.name}: {dataset.test_file.name} missing")
            continue
        mention_definitions, values = definitions.mention_prompt(dataset)
        parts = [d.description for d in mention_definitions] + list(values.values())
        if dataset.documents_file is not None:  # PET, the only dataset with relations
            relation_definitions, relation_values = definitions.relation_prompt(dataset)
            parts += [d.description for d in relation_definitions]
            parts += list(relation_values.values())
        prompt = normalized(" ".join(parts))
        test = test_split.load_test_split(dataset.test_file)
        sentences = {test_split.sentence_id(r): normalized(" ".join(r["tokens"])) for r in test}
        long_enough = {i: s for i, s in sentences.items() if len(s.split()) >= MIN_LEAK_WORDS}
        leaked = [i for i, sentence in long_enough.items() if sentence in prompt]
        checks.report(
            not leaked, "prompts",
            f"{dataset.name}: none of {len(long_enough)} test sentences of {MIN_LEAK_WORDS}+ words "
            f"in the prompt" + (f", but found {leaked}" if leaked else ""),
        )


def report_pet_versions():
    print("Note: PET test sentences labelled differently in pet.jsonl and PET v1.1")
    pet = datasets.DATASETS["pet"]
    documents = {d.id: d for d in datasets.read_documents(pet.documents_file)}
    differing = []
    for row in test_split.load_test_split(pet.test_file):
        document = documents[row["document name"]]
        tags, _ = test_split.sentence_tags(document, row["sentence-ID"], pet.types)
        if tags != row["ner-tags"]:
            differing.append(test_split.sentence_id(row))
    print(
        f"  {len(differing)} of the test sentences: {', '.join(differing)}. Whole-document PET "
        "runs are scored against pet.jsonl in metrics.json, and against v1.1 only in test_*."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--uq-pet-token", type=pathlib.Path, default=REPO_DIR.parent / "uq-pet-token",
        help="uq-pet-token's folder (default: next to this repository)",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="don't rebuild the splits with uq-pet-token's code",
    )
    args = parser.parse_args()
    uq_dir = args.uq_pet_token.resolve()
    if not uq_dir.is_dir():
        sys.exit(f"{uq_dir} not found, see --uq-pet-token")

    checks = Checks()
    if args.no_rebuild:
        print("1. Splits rebuilt with uq-pet-token's code: skipped (--no-rebuild)")
    else:
        check_rebuild(checks, uq_dir)
    check_stored_runs(checks, uq_dir)
    check_runs_here(checks)
    check_prompts(checks)
    report_pet_versions()

    print()
    if checks.failed:
        print(f"{len(checks.failed)} checks failed:", *checks.failed, sep="\n  ")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
