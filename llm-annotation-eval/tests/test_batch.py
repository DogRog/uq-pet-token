"""Running configs (experiments.batch) and what the experiments notebook shows while they run."""
import json

import pytest

from eval import datasets, test_split
from experiments import batch, report, runner
from experiments.config import RunConfig

PET = datasets.DATASETS["pet"]


def answer(params):
    """Finds every "the" as an Actor, in every sentence of the document."""
    user = params["messages"][1]["content"][0]["text"]
    sentences = [line.split(": ", 1) for line in user.splitlines() if line[:1].isdigit()]
    return "\n".join(f"the\tActor\t{i}" for i, text in sentences if "the" in text.split())


def test_on_document_is_called_after_each_saved_document(tmp_path, fake_client):
    config = RunConfig(name="small", provider="nhr", model="m", task="md", max_documents=3)
    directory = runner.run_dir(tmp_path, config, 1)
    seen = []

    def on_document(document_id, error):
        predicted = [d.id for d in runner.read_predictions(directory, "md")]
        seen.append((document_id, error, document_id in predicted, len(predicted)))

    runner.annotate_run(
        config, 1, directory, PET.gold_file, fake_client(answer), workers=2, on_document=on_document
    )

    first = datasets.read_documents(PET.gold_file)[:3]
    assert sorted(i for i, *_ in seen) == sorted(d.id for d in first)
    # each document is saved before it is reported; the other worker's may be saved too
    assert all(error is None and saved for _, error, saved, _ in seen)
    counts = [n for *_, n in seen]
    assert counts == sorted(counts) and all(n >= i for i, n in enumerate(counts, start=1))


def test_scores_so_far_cover_the_documents_done(tmp_path, fake_client):
    config = RunConfig(name="small", provider="nhr", model="m", task="md", max_documents=2)
    directory = runner.run_dir(tmp_path, config, 1)
    gold = datasets.read_documents(PET.gold_file)
    test = test_split.load_test_split(PET.test_file)
    assert report.scores_so_far(directory, "md", gold, test, PET.types)["documents"] == 0

    runner.annotate_run(config, 1, directory, PET.gold_file, fake_client(answer, prompt_tokens=7))
    scores = report.scores_so_far(directory, "md", gold, test, PET.types)

    done = {d.id for d in gold[:2]}
    covered = [r for r in test if r["document name"] in done]
    assert scores["documents"] == 2 and scores["prompt_tokens"] == 14
    assert 0 < scores["md_f1"] < 1 and "re_f1" not in scores
    assert scores["test_sentences"] == len(covered)
    if covered:
        assert scores["test_f1"] <= scores["test_partial_f1"]


def test_run_config_scores_and_exports_each_run(tmp_path, fake_client):
    config = RunConfig(
        name="sentences", provider="nhr", model="m", task="md-test-sentences", max_documents=4,
        repeats=2,
    )
    progress = []

    results = batch.run_config(
        config, fake_client(answer), workers=2, pred_dir=tmp_path,
        on_document=lambda run, directory, document_id, error: progress.append(run),
    )

    assert [(r.run, r.error) for r in results] == [(1, None), (2, None)]
    assert progress == [1] * 4 + [2] * 4
    # 4 of the test sentences, scored only on them, so no test-predictions.jsonl
    assert results[0].metrics == {"documents": 4}
    predictions_file = runner.predictions_file(results[0].directory, config.task)
    assert not predictions_file.with_name("test-predictions.jsonl").exists()
    assert json.loads((results[1].directory / "metrics.json").read_text()) == {"documents": 4}


def test_a_failing_run_skips_the_rest_of_its_config(tmp_path, fake_client):
    config = RunConfig(
        name="failing", provider="nhr", model="m", task="md", max_documents=2, repeats=3
    )

    def rejected(params):
        raise ValueError("model not found")

    [result] = batch.run_config(config, fake_client(rejected), pred_dir=tmp_path)

    assert result.run == 1 and "model not found" in result.error and result.metrics is None


def test_problems_before_running(monkeypatch, tmp_path):
    configs = [RunConfig(name="a", provider="nhr", model=m) for m in ("m", "n")]
    assert batch.problems(configs[:1]) == []
    assert "unique" in batch.problems(configs)[0]

    missing = datasets.Dataset(
        "pet", PET.types, tmp_path / "missing.jsonl", PET.test_file, PET.bootstrap_file
    )
    monkeypatch.setitem(datasets.DATASETS, "pet", missing)
    [problem] = batch.problems(configs[:1])
    assert problem.startswith("missing.jsonl is missing, needed by a")


def test_select_by_name_patterns():
    names = ("gemma-md", "gemma-re", "qwen-md")
    configs = [RunConfig(name=n, provider="nhr", model="m") for n in names]
    assert [c.name for c in batch.select(configs, ["gemma-*"])] == ["gemma-md", "gemma-re"]
    assert batch.select(configs, []) == configs


def test_resume_problem_names_what_changed(tmp_path, fake_client):
    config = RunConfig(name="small", provider="nhr", model="m", task="md", max_documents=1)
    directory = runner.run_dir(tmp_path, config, 1)
    assert runner.resume_problem(directory, config, 1) is None

    runner.annotate_run(config, 1, directory, PET.gold_file, fake_client(answer))

    assert runner.resume_problem(directory, config, 1) is None
    changed = RunConfig(name="small", provider="nhr", model="other", task="md")
    assert "settings" in runner.resume_problem(directory, changed, 1)


def test_export_test_predictions_of_every_run(tmp_path, fake_client):
    covering = RunConfig(name="all", provider="nhr", model="m", task="md-test-sentences")
    partial = RunConfig(name="some", provider="nhr", model="m", task="md", max_documents=1)
    for config in (covering, partial):
        runner.annotate_run(
            config, 1, runner.run_dir(tmp_path, config, 1), PET.input_file(config.task),
            fake_client(answer),
        )

    written, skipped = report.export_test_predictions(tmp_path)

    assert [p.parent.parent.parent.name for p in written] == ["all"]
    test = test_split.load_test_split(PET.test_file)
    assert len(written[0].read_text().splitlines()) == len(test)
    assert [p.parent.parent.parent.name for p in skipped] == ["some"]


@pytest.mark.parametrize("task", ["md", "md-test-sentences"])
def test_prompt_preview_needs_no_client(task):
    config = RunConfig(name="preview", provider="nhr", model="m", task=task)
    document = datasets.read_documents(PET.gold_file)[0]

    annotators = runner.build_annotators(config, None)

    assert list(annotators) == ["mentions"]
    params = annotators["mentions"].get_params(document)
    assert [m["role"] for m in params["messages"]] == ["system", "user"]
