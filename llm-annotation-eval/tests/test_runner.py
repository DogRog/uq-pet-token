import csv
import json

import httpx
import openai
import pytest

from autoannotate import JsonExporter, JsonImporter, data
from eval import datasets, test_split
from experiments import report, runner
from experiments.config import RunConfig

PET_TYPES = datasets.DATASETS["pet"].types

CONFIG = RunConfig(name="test-config", provider="nhr", model="org/Model-1", reasoning_effort="low")
REQUEST = httpx.Request("POST", "https://llm.test/v1/chat/completions")
TIMEOUT = openai.APITimeoutError(request=REQUEST)


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


@pytest.fixture
def test():
    """The first sentence of every gold document as a test sentence, see eval.test_split."""
    tokens = "the clerk checks the form".split()
    tags = ["O", "B-Actor", "B-Activity", "B-Activity Data", "I-Activity Data"]
    return [
        {"document name": doc, "sentence-ID": 0, "tokens": tokens, "ner-tags": tags}
        for doc in ("doc-a", "doc-b", "doc-c")
    ]


@pytest.fixture
def test_file(tmp_path, test):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in test))
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

    predicted = JsonImporter(runner.predictions_file(directory, "end-to-end")).do_import()
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
    # the gateway's response cache would make repeated runs copies of the first
    assert all(c["extra_body"] == {"cache": {"no-cache": True}} for c in client.calls)

    manifest = read_manifest(directory)
    assert manifest["config"] == "test-config" and manifest["run"] == 1
    assert manifest["settings"] == {
        "dataset": "pet", "task": "end-to-end", "provider": "nhr", "model": "org/Model-1", "reasoning_effort": "low"
    }
    assert manifest["api"]["instruction_role"] == "system"
    assert manifest["api"]["extra_body"] == {"cache": {"no-cache": True}}
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


def raise_for(*doc_ids, error):
    def _answer(params):
        if any(doc_id in params["messages"][1]["content"][0]["text"] for doc_id in doc_ids):
            raise error
        return answer(params)
    return _answer


def test_temporary_failure_skips_the_document_until_the_next_start(tmp_path, gold_file, fake_client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)

    runner.annotate_run(CONFIG, 1, directory, gold_file, fake_client(raise_for("doc-b", error=TIMEOUT)))

    manifest = read_manifest(directory)
    [session] = manifest["sessions"]
    assert session["outcome"] == "incomplete"
    assert session["failed_documents"] == ["doc-b"]
    assert manifest["totals"]["documents"] == 2
    assert [c["error"] for c in read_calls(directory) if "error" in c] == ["APITimeoutError: Request timed out."]

    runner.annotate_run(CONFIG, 1, directory, gold_file, fake_client(answer))

    manifest = read_manifest(directory)
    assert [s["outcome"] for s in manifest["sessions"]] == ["incomplete", "completed"]
    assert manifest["sessions"][1]["documents_to_do"] == 1
    assert manifest["totals"]["documents"] == 3
    assert manifest["totals"]["all"]["calls"] == 6
    assert manifest["totals"]["discarded_calls"] == 1


def test_rejected_request_stops_the_run(tmp_path, gold_file, fake_client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    rejected = openai.BadRequestError(
        "reasoning_effort=low is not supported", response=httpx.Response(400, request=REQUEST), body=None
    )

    with pytest.raises(openai.BadRequestError):
        runner.annotate_run(CONFIG, 1, directory, gold_file, fake_client(raise_for("doc-b", "doc-c", error=rejected)))

    manifest = read_manifest(directory)
    assert manifest["sessions"][0]["outcome"] == "failed"
    assert "not supported" in manifest["sessions"][0]["error"]
    assert manifest["totals"]["documents"] == 1


def test_failures_in_a_row_stop_the_run(tmp_path, gold_file, fake_client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)

    with pytest.raises(RuntimeError, match="3 documents in a row failed"):
        runner.annotate_run(CONFIG, 1, directory, gold_file, fake_client(raise_for("doc", error=TIMEOUT)))

    [session] = read_manifest(directory)["sessions"]
    assert session["outcome"] == "failed"
    assert len(session["failed_documents"]) == 3


def test_parallel_workers(tmp_path, gold_file, client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)

    runner.annotate_run(CONFIG, 1, directory, gold_file, client, workers=3)

    predicted = JsonImporter(runner.predictions_file(directory, "end-to-end")).do_import()
    assert sorted(d.id for d in predicted) == ["doc-a", "doc-b", "doc-c"]
    assert len(read_calls(directory)) == 6
    manifest = read_manifest(directory)
    assert manifest["sessions"][0]["workers"] == 3
    assert manifest["totals"]["all"]["calls"] == 6


def test_mention_detection_only(tmp_path, gold_file, client, test):
    pred_dir = tmp_path / "pred"
    config = RunConfig(name="md-only", provider="nhr", model="m", task="md")
    directory = runner.run_dir(pred_dir, config, 1)

    runner.annotate_run(config, 1, directory, gold_file, client)
    metrics = report.score_run(directory, gold_file, "md", test, PET_TYPES)
    [row] = report.write_summary(pred_dir)

    assert [d.id for d in JsonImporter(runner.predictions_file(directory, "md")).do_import()] == ["doc-a", "doc-b", "doc-c"]
    assert not (directory / "end-to-end").exists()
    assert {c["step"] for c in read_calls(directory)} == {"mentions"}
    manifest = read_manifest(directory)
    assert list(manifest["prompts"]) == ["mentions"]
    assert manifest["totals"]["relations"]["calls"] == 0
    assert set(metrics) == {"documents", "md", "test"}
    # the clerk and the checking match exactly, the form is missed
    assert metrics["test"]["entity_precision"] == 1.0
    assert metrics["test"]["entity_recall"] == pytest.approx(2 / 3)
    assert row["test_f1"] == pytest.approx(0.8)
    assert row["task"] == "md" and row["md_f1"] == pytest.approx(0.8)
    assert row["re_f1"] is None
    with open(pred_dir / "runs.csv") as f:
        assert next(csv.DictReader(f))["re_f1"] == ""
    report.print_summary([row])


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


def test_score_and_summarize_repeated_runs(tmp_path, gold_file, client, fake_client, test):
    pred_dir = tmp_path / "pred"
    config = RunConfig(name="repeated", provider="nhr", model="m", repeats=2)
    # the second run finds nothing
    clients = [client, fake_client("", prompt_tokens=50, completion_tokens=1)]
    for run, run_client in enumerate(clients, start=1):
        directory = runner.run_dir(pred_dir, config, run)
        runner.annotate_run(config, run, directory, gold_file, run_client)
        metrics = report.score_run(directory, gold_file, "end-to-end", test, PET_TYPES)
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


def test_test_sentences_are_annotated_on_their_own(tmp_path, gold_file, client, test, test_file):
    pred_dir = tmp_path / "pred"
    config = RunConfig(name="sentences", provider="nhr", model="m", task="md-test-sentences")
    directory = runner.run_dir(pred_dir, config, 1)

    runner.annotate_run(config, 1, directory, test_file, client, workers=2)
    metrics = report.score_run(directory, gold_file, "md-test-sentences", test, PET_TYPES)
    [row] = report.write_summary(pred_dir)

    predicted = JsonImporter(runner.predictions_file(directory, "md-test-sentences")).do_import()
    assert sorted(d.id for d in predicted) == ["doc-a#0", "doc-b#0", "doc-c#0"]
    # one request per sentence, with only that sentence
    assert len(client.calls) == 3
    prompts = [c["messages"][1]["content"][0]["text"] for c in client.calls]
    assert all(p.endswith("\n\n0: the clerk checks the form") for p in prompts)
    assert read_manifest(directory)["sessions"][0]["gold_file"].endswith("test.jsonl")
    # no gold documents to score single sentences against, only the test split
    assert set(metrics) == {"documents", "test"}
    assert metrics["test"]["entity_f1"] == pytest.approx(0.8)
    assert row["md_f1"] is None and row["test_f1"] == pytest.approx(0.8)
    predictions_file = runner.predictions_file(directory, "md-test-sentences")
    assert test_split.export(predictions_file, test, PET_TYPES) is not None


def test_runs_from_before_datasets_resume_as_pet(tmp_path, gold_file, client):
    directory = runner.run_dir(tmp_path, CONFIG, 1)
    runner.annotate_run(CONFIG, 1, directory, gold_file, client)
    # run.json as it was written before there were other datasets
    manifest = read_manifest(directory)
    del manifest["settings"]["dataset"]
    runner.write_json(directory / "run.json", manifest)
    (directory / "end-to-end" / "pet.jsonl").unlink()

    runner.annotate_run(CONFIG, 1, directory, gold_file, client)

    assert read_manifest(directory)["settings"]["dataset"] == "pet"


def test_quishpi_run(tmp_path, fake_client):
    pred_dir = tmp_path / "pred"
    sentences = [
        {"document name": doc, "sentence-ID": 0, "tokens": "the clerk checks the form".split(),
         "ner-tags": ["O", "O", "B-Action", "O", "B-Entity"]}
        for doc in ("doc-a", "doc-b")
    ]
    sentences_file = tmp_path / "quishpi.jsonl"
    sentences_file.write_text("".join(json.dumps(r) + "\n" for r in sentences))
    config = RunConfig(name="quishpi-md", provider="nhr", model="m", dataset="quishpi", task="md")
    directory = runner.run_dir(pred_dir, config, 1)
    quishpi_client = fake_client("clerk\tEntity\t0\nchecks\tAction\t0")

    runner.annotate_run(config, 1, directory, sentences_file, quishpi_client)
    types = datasets.DATASETS["quishpi"].types
    metrics = report.score_run(directory, sentences_file, "md", sentences, types)
    [row] = report.write_summary(pred_dir)

    instruction = quishpi_client.calls[0]["messages"][0]["content"][0]["text"]
    assert "**Condition**" in instruction and "**Activity**" not in instruction
    assert read_manifest(directory)["settings"]["dataset"] == "quishpi"
    # checks is right, the clerk is not an entity, the form is missed
    assert metrics["md"]["micro"]["precision"] == 0.5
    assert metrics["test"]["entity_precision"] == 0.5 and metrics["test"]["entity_recall"] == 0.5
    assert row["dataset"] == "quishpi" and row["test_f1"] == 0.5
