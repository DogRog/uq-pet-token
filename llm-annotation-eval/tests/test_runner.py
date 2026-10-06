import csv
import json

import pytest

from autoannotate import JsonExporter, JsonImporter, data
from experiments import report, runner
from experiments.config import RunConfig

CONFIG = RunConfig(name="test-config", provider="nhr", model="org/Model-1", reasoning_effort="low")


@pytest.fixture
def gold_file(tmp_path, make_doc):
    path = tmp_path / "gold.jsonl"
    documents = [
        make_doc(
            # the second sentence tells the documents apart in the prompt
            ["the clerk checks the form", f"{doc_id} ends"],
            mentions=[data.Mention("actor", (1,)), data.Mention("activity", (2,)), data.Mention("activity data", (3, 4))],
            relations=[data.Relation("actor performer", 1, 0), data.Relation("uses", 1, 2)],
            doc_id=doc_id,
        )
        for doc_id in ("doc-a", "doc-b", "doc-c")
    ]
    JsonExporter(str(path)).export(documents)
    return path


def answer(params, fail_on=None):
    """Answers like a decent LLM: finds the clerk and the checking, but misses the form."""
    instruction, user = (m["content"][0]["text"] for m in params["messages"])
    if fail_on is not None and fail_on in user:
        raise TimeoutError("request timed out")
    if "relations between" in instruction:
        return "actor performer\t1\t0"
    return "Mentions:\nclerk\tActor\t0\nchecks\tActivity\t0"


@pytest.fixture
def client(fake_client):
    return fake_client(answer, prompt_tokens=100, completion_tokens=10, reasoning="thinking", reasoning_tokens=4)


def read_calls(directory):
    return [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]


def read_manifest(directory):
    return json.loads((directory / "run.json").read_text())


def test_run_writes_predictions_calls_and_manifest(tmp_path, gold_file, client):
    directory = runner.run_dir(tmp_path / "pred", CONFIG, 1)

    runner.annotate_run(CONFIG, 1, directory, gold_file, client)

    predicted = JsonImporter(runner.predictions_file(directory)).do_import()
    assert [d.id for d in predicted] == ["doc-a", "doc-b", "doc-c"]
    assert predicted[0].relations == [data.Relation("actor performer", 1, 0)]

    calls = read_calls(directory)
    assert [(c["document_id"], c["step"]) for c in calls] == [
        ("doc-a", "mentions"), ("doc-a", "relations"), ("doc-b", "mentions"), ("doc-b", "relations"),
        ("doc-c", "mentions"), ("doc-c", "relations"),
    ]
    assert calls[0]["response"] == "Mentions:\nclerk\tActor\t0\nchecks\tActivity\t0"
    assert calls[0]["reasoning"] == "thinking"
    assert calls[0]["response_model"] == "org/Model-1-2026-10-01"
    assert calls[0]["num_predicted"] == 2
    assert [i["kind"] for i in calls[0]["issues"]] == ["ignored_line"]
    assert all(c["reasoning_effort"] == "low" for c in client.calls)

    manifest = read_manifest(directory)
    assert manifest["config"] == "test-config" and manifest["run"] == 1
    assert manifest["settings"] == {"provider": "nhr", "model": "org/Model-1", "reasoning_effort": "low"}
    assert manifest["api"]["instruction_role"] == "system"
    assert "relations between" in manifest["prompts"]["relations"]
    [session] = manifest["sessions"]
    assert session["outcome"] == "completed"
    assert session["documents_to_do"] == 3
    assert set(session["packages"]) == {"openai", "langextract"}
    assert len(session["gold_sha256"]) == 64
    totals = manifest["totals"]
    assert totals["documents"] == 3 and totals["discarded_calls"] == 0
    assert totals["all"]["prompt_tokens"] == 600
    assert totals["all"]["reasoning_tokens"] == 24
    assert totals["mentions"]["predicted"] == 6
    assert totals["mentions"]["issues"] == {"ignored_line": 3}
    assert totals["relations"]["finish_reasons"] == {"stop": 3}


def test_max_documents(tmp_path, gold_file, client):
    config = RunConfig(name="small", provider="nhr", model="m", max_documents=2)
    directory = runner.run_dir(tmp_path, config, 1)

    runner.annotate_run(config, 1, directory, gold_file, client)

    assert read_manifest(directory)["totals"]["documents"] == 2


def test_failed_run_is_logged_and_resumes(tmp_path, gold_file, fake_client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    failing = fake_client(lambda params: answer(params, fail_on="doc-b"))

    with pytest.raises(TimeoutError):
        runner.annotate_run(CONFIG, 1, directory, gold_file, failing)

    manifest = read_manifest(directory)
    assert manifest["sessions"][0]["outcome"] == "failed"
    assert "request timed out" in manifest["sessions"][0]["error"]
    assert manifest["totals"]["documents"] == 1
    assert read_calls(directory)[-1]["error"] == "TimeoutError: request timed out"

    runner.annotate_run(CONFIG, 1, directory, gold_file, fake_client(answer))

    manifest = read_manifest(directory)
    assert [s["outcome"] for s in manifest["sessions"]] == ["failed", "completed"]
    assert manifest["sessions"][1]["documents_to_do"] == 2
    assert manifest["totals"]["documents"] == 3
    assert manifest["totals"]["all"]["calls"] == 6
    # the failed call of the second document
    assert manifest["totals"]["discarded_calls"] == 1


def test_complete_run_is_not_started_again(tmp_path, gold_file, client, fake_client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    runner.annotate_run(CONFIG, 1, directory, gold_file, client)
    unused = fake_client(answer)

    runner.annotate_run(CONFIG, 1, directory, gold_file, unused)

    assert unused.calls == []
    assert len(read_manifest(directory)["sessions"]) == 1


def test_resuming_with_other_settings_fails(tmp_path, gold_file, client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    runner.annotate_run(CONFIG, 1, directory, gold_file, client)
    (directory / "end-to-end" / "pet.jsonl").unlink()
    changed = RunConfig(name=CONFIG.name, provider="nhr", model="org/Model-1", reasoning_effort="high")

    with pytest.raises(RuntimeError, match="settings of test-config changed"):
        runner.annotate_run(changed, 1, directory, gold_file, client)


def test_resuming_with_other_prompts_fails(tmp_path, gold_file, client, monkeypatch):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    runner.annotate_run(CONFIG, 1, directory, gold_file, client)
    (directory / "end-to-end" / "pet.jsonl").unlink()
    monkeypatch.setattr(runner.definitions, "mention_format_example", "another example")

    with pytest.raises(RuntimeError, match="prompts of test-config changed"):
        runner.annotate_run(CONFIG, 1, directory, gold_file, client)


def test_score_and_summarize_repeated_runs(tmp_path, gold_file, client, fake_client):
    pred_dir = tmp_path / "pred"
    config = RunConfig(name="repeated", provider="nhr", model="m", repeats=2)
    # the second run finds nothing
    clients = [client, fake_client("", prompt_tokens=50, completion_tokens=1)]
    for run, run_client in enumerate(clients, start=1):
        directory = runner.run_dir(pred_dir, config, run)
        runner.annotate_run(config, run, directory, gold_file, run_client)
        metrics = report.score_run(directory, gold_file)
        assert json.loads((directory / "metrics.json").read_text()) == metrics

    summary = report.write_summary(pred_dir)

    with open(pred_dir / "runs.csv") as f:
        runs = list(csv.DictReader(f))
    assert [(r["config"], r["run"], r["documents"], r["complete"]) for r in runs] == [
        ("repeated", "1", "3", "True"), ("repeated", "2", "3", "True"),
    ]
    # run 1 finds 2 of 3 mentions with 2 predictions, and 1 of 2 relations with 1 prediction
    assert float(runs[0]["md_precision"]) == 1.0
    assert float(runs[0]["md_recall"]) == pytest.approx(2 / 3)
    assert float(runs[0]["re_micro_recall"]) == 0.5
    assert runs[0]["reasoning_tokens"] == "24" and runs[1]["reasoning_tokens"] == ""
    assert runs[1]["dropped_mention_lines"] == "0"

    [row] = summary
    assert row["runs"] == 2 and row["documents"] == "3"
    assert row["md_recall"] == pytest.approx(1 / 3)
    assert row["md_recall_sd_runs"] == pytest.approx(0.4714, abs=1e-4)
    assert row["prompt_tokens"] == 450
    # only one run reported reasoning tokens
    assert row["reasoning_tokens"] == 24 and row["reasoning_tokens_sd_runs"] is None
    assert (pred_dir / "summary.csv").exists()
    report.print_summary(summary)


def test_summary_ignores_folders_without_run_logs(tmp_path):
    (tmp_path / "old-model" / "end-to-end").mkdir(parents=True)
    assert report.write_summary(tmp_path) == []
