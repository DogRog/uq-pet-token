"""Checks that results here and uq-pet-token's are on the same data and scored the same way.

``run`` does all checks, notebooks/run_experiments.py shows them:

1. Splits: rebuilds every dataset's sentences and test split with uq-pet-token's own code and
   compares them with res/gold/ (scripts/uq-pet-token-splits.py --check, run in uq-pet-token's
   environment; it downloads the datasets once, the medical one needs a Hugging Face login).
   Optional, as it is the slow one.
2. Reported test gold: every test run of uq-pet-token's supervised sweeps
   (results/supervised/<sweep>/test/*/*/predictions.npz, read by the results notebook) stores the
   gold tags and sentence lengths it was scored on; they must equal res/gold/<dataset>-test.jsonl.
3. Scoring: eval.test_split.evaluate on those runs' stored predictions must reproduce each model
   seed's metrics in their results.csv.
4. Runs here: the gold file every session of every run annotated and was scored against (its
   sha256 in run.json) must be today's file. Prediction folders without run.json can't be checked
   and are noted.
5. Prompts: no test sentence of five words or more appears in a dataset's prompt.

Also notes, without failing, the PET test sentences whose labels in res/gold/pet.jsonl (what PET
whole-document runs annotate) differ from PET v1.1, as only test scores use the latter.
"""
import csv
import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
import typing

import numpy as np

from eval import datasets, test_split
from experiments import definitions

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
PRED_DIR = REPO_DIR / "res" / "pred"
# this project is the llm-annotation-eval/ folder of uq-pet-token
UQ_PET_TOKEN_DIR = REPO_DIR.parent

# the metrics uq-pet-token's results.csv has per model seed, as eval.test_split.evaluate names them
METRICS = (
    "entity_f1", "entity_macro_f1", "entity_precision", "entity_recall", "entity_partial_f1",
    "entity_partial_macro_f1", "entity_partial_precision", "entity_partial_recall",
    "token_accuracy",
)
# test sentences with fewer words (not counting punctuation, e.g. headings like "# # Diabetes # #")
# may well occur in a prompt as a mention text
MIN_LEAK_WORDS = 5


@dataclasses.dataclass
class Result:
    # e.g. "splits", "scoring"
    check: str
    ok: bool
    message: str


@dataclasses.dataclass
class Report:
    results: typing.List[Result] = dataclasses.field(default_factory=list)
    # what the checks found that doesn't fail them
    notes: typing.List[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def add(self, check: str, ok: bool, message: str) -> None:
        self.results.append(Result(check, ok, message))


def _available(dataset: datasets.Dataset) -> bool:
    return all(
        path.exists()
        for path in (dataset.sentences_file, dataset.test_file, dataset.bootstrap_file)
    )


def check_splits(report: Report, uq_dir: pathlib.Path) -> None:
    script = REPO_DIR / "scripts" / "uq-pet-token-splits.py"
    result = subprocess.run(
        ["uv", "run", "--project", str(uq_dir), "python", str(script), "--check"],
        cwd=uq_dir, capture_output=True, text=True,
    )
    output = result.stdout.strip()
    if result.returncode not in (0, 1):
        output += "\n" + result.stderr.strip()[-2000:]
    message = f"res/gold/ equals uq-pet-token's splits\n{output}"
    report.add("splits", result.returncode == 0, message)


def _stored_runs(uq_dir: pathlib.Path) -> typing.Iterator[typing.Tuple[str, str, pathlib.Path]]:
    """(sweep, dataset, predictions.npz) of every test run of uq-pet-token's supervised sweeps."""
    for config_file in sorted(uq_dir.glob("results/supervised/*/run_config.json")):
        dataset = json.loads(config_file.read_text(encoding="utf8"))["dataset"]
        for npz in sorted(config_file.parent.glob("test/*/*/predictions.npz")):
            yield config_file.parent.name, dataset, npz


def check_reported_runs(report: Report, uq_dir: pathlib.Path) -> None:
    """Checks 2 and 3: the stored test gold, and rescoring the stored predictions."""
    runs = list(_stored_runs(uq_dir))
    report.add(
        "reported runs", bool(runs), f"{len(runs)} test runs in {uq_dir}/results/supervised/"
    )
    tests: typing.Dict[str, typing.List[test_split.Row]] = {}
    for sweep, name, npz in runs:
        dataset = datasets.DATASETS[name]
        if not _available(dataset):
            report.notes.append(f"{sweep} not checked: {dataset.test_file.name} is missing")
            continue
        test = tests.setdefault(name, test_split.load_test_split(dataset.test_file))
        stored = np.load(npz, allow_pickle=True)
        labels = [str(label) for label in stored["labels"]]
        gold = [labels[i] for i in stored["gold"].tolist()]
        same_gold = (
            stored["sentence_lengths"].tolist() == [len(r["tokens"]) for r in test]
            and gold == [tag for r in test for tag in r["ner-tags"]]
        )
        report.add(
            "reported gold", same_gold,
            f"{sweep}: test gold equals {dataset.test_file.name} ({len(test)} sentences, "
            f"{len(gold)} tokens)",
        )
        if not same_gold:
            continue
        with open(npz.parent / "results.csv", encoding="utf8") as f:
            seeds = list(csv.DictReader(f))
        largest = 0.0
        for seed, row in enumerate(seeds):
            flat = [labels[i] for i in stored["predicted"][seed].tolist()]
            predicted, start = [], 0
            for r in test:
                predicted.append({**r, "ner-tags": flat[start:start + len(r["tokens"])]})
                start += len(r["tokens"])
            scores = test_split.evaluate(test, predicted)
            largest = max(largest, *(abs(scores[m] - float(row[m])) for m in METRICS))
        report.add(
            "scoring", largest < 1e-9,
            f"{sweep}: {len(seeds)} model seeds rescored, largest difference to results.csv "
            f"{largest:.1e}",
        )


def check_runs_here(report: Report, pred_dir: pathlib.Path = PRED_DIR) -> None:
    current = {
        p.relative_to(REPO_DIR).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (REPO_DIR / "res" / "gold").glob("*.jsonl")
    }
    manifest_files = sorted(pred_dir.glob("*/run-*/run.json"))
    sessions, stale = 0, []
    for manifest_file in manifest_files:
        for session in json.loads(manifest_file.read_text(encoding="utf8"))["sessions"]:
            sessions += 1
            if current.get(session["gold_file"]) != session["gold_sha256"]:
                stale.append(f"{manifest_file.parent.name} of {manifest_file.parent.parent.name}")
    report.add(
        "runs here", not stale,
        f"{sessions} sessions in {len(manifest_files)} runs used today's gold files"
        + (f", not: {', '.join(stale)}" if stale else ""),
    )
    unrecorded = sorted({
        p.parent.parent.name for p in pred_dir.glob("*/*/pet.jsonl")
        if not p.parent.parent.name.startswith("run-")
    })
    if unrecorded:
        report.notes.append(
            f"Predictions without run.json, so with unknown gold: {', '.join(unrecorded)}"
        )


def _normalized(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def check_prompts(report: Report) -> None:
    for dataset in datasets.DATASETS.values():
        if not _available(dataset):
            report.notes.append(f"{dataset.name} prompt not checked: its files are missing")
            continue
        mention_definitions, values = definitions.mention_prompt(dataset)
        parts = [d.description for d in mention_definitions] + list(values.values())
        if dataset.documents_file is not None:  # PET, the only dataset with relations
            relation_definitions, relation_values = definitions.relation_prompt(dataset)
            parts += [d.description for d in relation_definitions]
            parts += list(relation_values.values())
        prompt = _normalized(" ".join(parts))
        test = test_split.load_test_split(dataset.test_file)
        sentences = {test_split.sentence_id(r): _normalized(" ".join(r["tokens"])) for r in test}
        long_enough = {i: s for i, s in sentences.items() if len(s.split()) >= MIN_LEAK_WORDS}
        leaked = [i for i, sentence in long_enough.items() if sentence in prompt]
        report.add(
            "prompts", not leaked,
            f"{dataset.name}: none of {len(long_enough)} test sentences of {MIN_LEAK_WORDS}+ "
            "words in the prompt" + (f", but found {leaked}" if leaked else ""),
        )


def note_pet_versions(report: Report) -> None:
    pet = datasets.DATASETS["pet"]
    documents = {d.id: d for d in datasets.read_documents(pet.documents_file)}
    differing = []
    for row in test_split.load_test_split(pet.test_file):
        document = documents[row["document name"]]
        tags, _ = test_split.sentence_tags(document, row["sentence-ID"], pet.types)
        if tags != row["ner-tags"]:
            differing.append(test_split.sentence_id(row))
    report.notes.append(
        f"{len(differing)} PET test sentences are labelled differently in pet.jsonl and PET v1.1: "
        f"{', '.join(differing)}. Whole-document PET runs are scored against pet.jsonl in "
        "metrics.json, and against v1.1 only in test_*."
    )


def run(uq_dir: pathlib.Path = UQ_PET_TOKEN_DIR, rebuild: bool = True) -> Report:
    """All checks; rebuild=False skips rebuilding the splits with uq-pet-token's code."""
    if not uq_dir.is_dir():
        raise FileNotFoundError(f"uq-pet-token not found at {uq_dir}")
    report = Report()
    if rebuild:
        check_splits(report, uq_dir)
    check_reported_runs(report, uq_dir)
    check_runs_here(report)
    check_prompts(report)
    note_pet_versions(report)
    return report
