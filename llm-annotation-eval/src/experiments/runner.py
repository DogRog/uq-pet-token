"""Annotates the gold documents for one run of a config, and logs what is needed to report on it.

A run folder (res/pred/<config>/run-<i>/) contains:

- <task>/pet.jsonl: the predicted documents, each appended as soon as it is done; <task> is
  "end-to-end", "md" or "md-test-sentences", see experiments.config; the file is called pet.jsonl
  for every dataset
- calls.jsonl: one line per LLM call, with tokens, duration, raw response, reasoning and parser issues
- run.json: the manifest, with settings, prompts, code and package versions per session, and totals
- metrics.json: the scores, written by experiments.report

Starting a run again resumes it: documents already in pet.jsonl are skipped. Every start that has
documents left adds a session to run.json.

Documents are annotated in parallel by several workers. A document whose request fails temporarily
(timeout, server error, rate limit, after the client's own retries) is skipped and left for the next
start; other errors, e.g. a rejected request, stop the run, as do several failed documents in a row.
"""
import collections
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import threading
import typing
from concurrent import futures

import openai
from rich.progress import track

import autoannotate
from autoannotate import data
from eval import datasets, test_split
from experiments import definitions
from experiments.config import RunConfig

REPO_DIR = pathlib.Path(__file__).parent.parent.parent
# versions are recorded per session, the parsers depend on langextract
PACKAGES = ["openai", "langextract"]
# errors that may go away when the document is annotated again later
TRANSIENT_ERRORS = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
# stop a run after this many failed documents in a row, e.g. if the model is not served at the moment
MAX_CONSECUTIVE_FAILURES = 3


def run_dir(pred_dir: pathlib.Path, config: RunConfig, run: int) -> pathlib.Path:
    return pred_dir / config.name / f"run-{run}"


def predictions_file(directory: pathlib.Path, task: str) -> pathlib.Path:
    return directory / task / "pet.jsonl"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git() -> typing.Dict[str, typing.Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()

    try:
        # dirty: there were uncommitted changes, so the commit alone does not describe the code
        return {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain") != ""}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path.resolve())


def _read_jsonl(path: pathlib.Path) -> typing.List[typing.Dict[str, typing.Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


# workers append to the same files
_write_lock = threading.Lock()


def _append_jsonl(path: pathlib.Path, entry: typing.Dict[str, typing.Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _write_lock, open(path, "a", encoding="utf8") as f:
        f.write(line)


def write_json(path: pathlib.Path, value: typing.Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def manifest_settings(manifest: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """The settings in a run.json; runs from before there were other datasets annotated PET."""
    return {"dataset": "pet", **manifest["settings"]}


def _load_manifest(
        directory: pathlib.Path, config: RunConfig, run: int, prompts: typing.Dict[str, str]
) -> typing.Dict[str, typing.Any]:
    provider = autoannotate.get_provider(config.provider)
    expected = {"config": config.name, "run": run, "settings": config.settings(), "prompts": prompts}
    manifest_file = directory / "run.json"
    if not manifest_file.exists():
        return {
            **expected,
            "api": {
                "base_url": provider.base_url,
                "instruction_role": provider.instruction_role,
                "extra_body": provider.extra_body,
            },
            "sessions": [],
        }

    manifest = json.loads(manifest_file.read_text(encoding="utf8"))
    manifest["settings"] = manifest_settings(manifest)
    for key in ("settings", "prompts"):
        if manifest[key] != expected[key]:
            raise RuntimeError(
                f"The {key} of {config.name} changed since {directory} was started, so the run would mix "
                f"predictions made with different {key}. Rename the config to start a new run, "
                f"or delete {directory} to start this one over."
            )
    return manifest


def _call(
        annotator: autoannotate.BaseAnnotator, step: str, document: data.Document, calls_file: pathlib.Path
) -> data.Document:
    entry: typing.Dict[str, typing.Any] = {"document_id": document.id, "step": step, "started_at": _now()}
    try:
        annotation = annotator.annotate(document)
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        _append_jsonl(calls_file, entry)
        raise

    doc = annotation.doc
    entry.update({
        "duration_s": round(annotation.duration_s, 3),
        "response_model": annotation.response_model,
        "finish_reason": annotation.finish_reason,
        "prompt_tokens": annotation.prompt_tokens,
        "completion_tokens": annotation.completion_tokens,
        "reasoning_tokens": annotation.reasoning_tokens,
        "num_predicted": len(doc.mentions) if step == "mentions" else len(doc.relations),
        "issues": annotation.issues,
        "response": annotation.response,
        "reasoning": annotation.reasoning,
    })
    _append_jsonl(calls_file, entry)
    if annotation.finish_reason == "length":
        print(f"The {step} answer for {document.id} was cut off at the token limit.")
    return doc


def totals(directory: pathlib.Path, task: str) -> typing.Dict[str, typing.Any]:
    """Sums up calls.jsonl over the documents in the predictions.

    A document that was interrupted and annotated again when resuming only counts with the calls
    whose results were kept.
    """
    out_file = predictions_file(directory, task)
    done_ids = {d.id for d in autoannotate.JsonImporter(out_file).do_import()} if out_file.exists() else set()
    calls = _read_jsonl(directory / "calls.jsonl")
    latest = {}
    for call in calls:
        if "error" not in call:
            latest[(call["document_id"], call["step"])] = call
    kept = [call for (doc_id, _), call in latest.items() if doc_id in done_ids]

    def sums(steps: typing.List[str]) -> typing.Dict[str, typing.Any]:
        step_calls = [c for c in kept if c["step"] in steps]
        reasoning = [c["reasoning_tokens"] for c in step_calls if c["reasoning_tokens"] is not None]
        return {
            "calls": len(step_calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in step_calls),
            "completion_tokens": sum(c["completion_tokens"] for c in step_calls),
            # part of completion_tokens; None if the provider did not report it for any call
            "reasoning_tokens": sum(reasoning) if reasoning else None,
            # summed over calls, which may have run in parallel
            "duration_s": round(sum(c["duration_s"] for c in step_calls), 3),
            "finish_reasons": dict(collections.Counter(str(c["finish_reason"]) for c in step_calls)),
            "issues": dict(collections.Counter(i["kind"] for c in step_calls for i in c["issues"])),
        }

    return {
        "documents": len(done_ids),
        # failed calls, and calls of documents that were annotated again
        "discarded_calls": len(calls) - len(kept),
        "all": sums(["mentions", "relations"]),
        "mentions": {**sums(["mentions"]), "predicted": sum(c["num_predicted"] for c in kept if c["step"] == "mentions")},
        "relations": {**sums(["relations"]), "predicted": sum(c["num_predicted"] for c in kept if c["step"] == "relations")},
    }


def input_documents(task: str, input_file: pathlib.Path) -> typing.List[data.Document]:
    """The documents a run annotates.

    Args:
        task: see experiments.config.TASKS
        input_file: the gold documents (see eval.datasets.read_documents), or for
            md-test-sentences the test split
    """
    if task == "md-test-sentences":
        return test_split.sentence_documents(test_split.load_test_split(input_file))
    return datasets.read_documents(input_file)


def annotate_run(
        config: RunConfig, run: int, directory: pathlib.Path, gold_file: pathlib.Path, client, workers: int = 1
) -> None:
    """Annotates the gold documents that are not yet in the run's predictions.

    :param gold_file: the gold documents of the config's dataset, or for md-test-sentences its
        test split, see eval.datasets.Dataset.input_file
    :param workers: number of documents annotated at the same time
    """
    provider = autoannotate.get_provider(config.provider)
    dataset = datasets.DATASETS[config.dataset]
    mention_definitions, mention_values = definitions.mention_prompt(dataset)
    mention_annotator = autoannotate.LLMMentionAnnotator(
        client=client,
        language_model=config.model,
        template_values=mention_values,
        mention_definitions=mention_definitions,
        reasoning_effort=config.reasoning_effort,
        template=definitions.mention_template(),
        instruction_role=provider.instruction_role,
        extra_body=provider.extra_body,
    )
    with_relations = config.task == "end-to-end"
    prompts = {"mentions": mention_annotator.get_instruction()}
    if with_relations:
        relation_definitions, relation_values = definitions.relation_prompt(dataset)
        relation_annotator = autoannotate.LLMRelationsAnnotator(
            client=client,
            language_model=config.model,
            template_values=relation_values,
            type_definitions=relation_definitions,
            reasoning_effort=config.reasoning_effort,
            template=definitions.relation_template(),
            instruction_role=provider.instruction_role,
            extra_body=provider.extra_body,
        )
        prompts["relations"] = relation_annotator.get_instruction()
    manifest = _load_manifest(directory, config, run, prompts)

    documents = input_documents(config.task, gold_file)[:config.max_documents]
    out_file = predictions_file(directory, config.task)
    done_ids = {d.id for d in autoannotate.JsonImporter(out_file).do_import()} if out_file.exists() else set()
    todo = [d for d in documents if d.id not in done_ids]
    if not todo:
        print(f"{config.name} run {run}: all {len(documents)} documents are done.")
        return
    if done_ids:
        print(f"{config.name} run {run}: resuming, {len(done_ids)} documents are already done.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    session = {
        "started_at": _now(),
        "finished_at": None,
        # "completed"; "incomplete" if documents failed and were skipped; "failed" if the run was
        # stopped by an error; "interrupted", e.g. by Ctrl+C
        "outcome": "interrupted",
        "documents_planned": len(documents),
        "documents_to_do": len(todo),
        "failed_documents": [],
        # parallel requests, they affect the measured call durations
        "workers": workers,
        "gold_file": _display_path(gold_file),
        "gold_sha256": hashlib.sha256(gold_file.read_bytes()).hexdigest(),
        "git": _git(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
    }
    manifest["sessions"].append(session)
    manifest["documents_planned"] = len(documents)
    write_json(directory / "run.json", manifest)

    exporter = autoannotate.JsonExporter(str(out_file))
    calls_file = directory / "calls.jsonl"

    def annotate_document(document: data.Document) -> None:
        print(f"Annotating {document.name}...")
        document = document.copy(clear=["mentions", "relations", "entities"])
        document = _call(mention_annotator, "mentions", document, calls_file)
        if with_relations:
            document = _call(relation_annotator, "relations", document, calls_file)
        with _write_lock:
            exporter.append(document)

    pool = futures.ThreadPoolExecutor(workers)
    try:
        pending = {pool.submit(annotate_document, d): d for d in todo}
        consecutive_failures = 0
        for future in track(futures.as_completed(pending), total=len(pending), description=f"{config.name} run {run}"):
            try:
                future.result()
                consecutive_failures = 0
            except TRANSIENT_ERRORS as e:
                print(f"Skipping {pending[future].id}, it is left for the next start: {type(e).__name__}: {e}")
                session["failed_documents"].append(pending[future].id)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"{consecutive_failures} documents in a row failed, last with {e}") from e
        session["outcome"] = "incomplete" if session["failed_documents"] else "completed"
    except Exception as e:
        session["outcome"] = "failed"
        session["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        # don't start the documents left, wait for the running ones, their results are saved
        pool.shutdown(wait=True, cancel_futures=True)
        session["finished_at"] = _now()
        manifest["totals"] = totals(directory, config.task)
        write_json(directory / "run.json", manifest)
