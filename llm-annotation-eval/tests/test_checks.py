"""The checks that results here and uq-pet-token's share data and scoring (experiments.checks)."""
import json

from eval import datasets, test_split
from experiments import checks, definitions


def test_prompts_have_no_test_sentences():
    report = checks.Report()

    checks.check_prompts(report)

    assert report.results and report.ok


def test_a_test_sentence_in_a_prompt_fails(monkeypatch):
    pet = datasets.DATASETS["pet"]
    sentence = next(r for r in test_split.load_test_split(pet.test_file) if len(r["tokens"]) > 8)
    real = definitions.mention_prompt

    def leaking(dataset):
        mention_definitions, values = real(dataset)
        return mention_definitions, {**values, "example": " ".join(sentence["tokens"])}

    monkeypatch.setattr(definitions, "mention_prompt", leaking)
    report = checks.Report()

    checks.check_prompts(report)

    assert not report.ok
    assert test_split.sentence_id(sentence) in report.results[0].message


def test_runs_on_other_gold_files_fail(tmp_path):
    directory = tmp_path / "some-config" / "run-1"
    directory.mkdir(parents=True)
    session = {"gold_file": "res/gold/pet.jsonl", "gold_sha256": "0" * 64}
    (directory / "run.json").write_text(json.dumps({"sessions": [session]}))
    (tmp_path / "old-model" / "md").mkdir(parents=True)
    (tmp_path / "old-model" / "md" / "pet.jsonl").write_text("")
    report = checks.Report()

    checks.check_runs_here(report, tmp_path)

    assert not report.ok and "run-1 of some-config" in report.results[0].message
    assert "old-model" in report.notes[0]


def test_pet_versions_note_the_differing_test_sentences():
    report = checks.Report()

    checks.note_pet_versions(report)

    assert report.ok and report.notes[0].startswith("8 PET test sentences")
