"""
Annotates the gold documents for one run of a config, and logs what is needed to report on it.

A run folder (res/pred/<config>/run-<i>/) contains:

- end-to-end/pet.jsonl: the predicted documents, each appended as soon as it is done
- calls.jsonl: one line per LLM call, with tokens, duration, raw response, reasoning and parser issues
- run.json: the manifest, with settings, prompts, code and package versions per session, and totals
- metrics.json: the scores, written by experiments.report

Starting a run again resumes it: documents already in pet.jsonl are skipped. Every start that has
documents left adds a session to run.json.
"""
import collections
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import typing

from rich.progress import track

import autoannotate
from autoannotate import data
from experiments import definitions
from experiments.config import RunConfig

REPO_DIR = pathlib.Path(__file__).parent.parent.parent
# versions are recorded per session, the parsers depend on langextract
PACKAGES = ["openai", "langextract"]


def run_dir(pred_dir: pathlib.Path, config: RunConfig, run: int) -> pathlib.Path:
    return pred_dir / config.name / f"run-{run}"


def predictions_file(directory: pathlib.Path) -> pathlib.Path:
    return directory / "end-to-end" / "pet.jsonl"


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


def _append_jsonl(path: pathlib.Path, entry: typing.Dict[str, typing.Any]) -> None:
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_json(path: pathlib.Path, value: typing.Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def _load_manifest(
        directory: pathlib.Path, config: RunConfig, run: int, prompts: typing.Dict[str, str]
) -> typing.Dict[str, typing.Any]:
    provider = autoannotate.get_provider(config.provider)
    expected = {"config": config.name, "run": run, "settings": config.settings(), "prompts": prompts}
    manifest_file = directory / "run.json"
    if not manifest_file.exists():
        return {
            **expected,
            "api": {"base_url": provider.base_url, "instruction_role": provider.instruction_role},
            "sessions": [],
        }

    manifest = json.loads(manifest_file.read_text(encoding="utf8"))
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


def totals(directory: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """
    Sums up calls.jsonl over the documents in the predictions. A document that was interrupted
    and annotated again when resuming only counts with the calls whose results were kept.
    """
    done_ids = {d.id for d in autoannotate.JsonImporter(predictions_file(directory)).do_import()} \
        if predictions_file(directory).exists() else set()
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


def annotate_run(
        config: RunConfig, run: int, directory: pathlib.Path, gold_file: pathlib.Path, client
) -> None:
    """Annotates the gold documents that are not yet in the run's predictions."""
    provider = autoannotate.get_provider(config.provider)
    mention_annotator = autoannotate.LLMMentionAnnotator(
        client=client,
        language_model=config.model,
        example=definitions.mention_format_example,
        mention_definitions=definitions.mention_definitions,
        reasoning_effort=config.reasoning_effort,
        instruction_role=provider.instruction_role,
    )
    relation_annotator = autoannotate.LLMRelationsAnnotator(
        client=client,
        language_model=config.model,
        example=definitions.relation_format_example,
        type_definitions=definitions.relation_definitions,
        reasoning_effort=config.reasoning_effort,
        instruction_role=provider.instruction_role,
    )
    prompts = {"mentions": mention_annotator.get_instruction(), "relations": relation_annotator.get_instruction()}
    manifest = _load_manifest(directory, config, run, prompts)

    documents = autoannotate.JsonImporter(gold_file).do_import()[:config.max_documents]
    out_file = predictions_file(directory)
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
        # "completed", "interrupted" (e.g. Ctrl+C) or "failed"
        "outcome": "interrupted",
        "documents_planned": len(documents),
        "documents_to_do": len(todo),
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
    try:
        for document in track(todo, description=f"{config.name} run {run}"):
            print(f"Annotating {document.name}...")
            document = document.copy(clear=["mentions", "relations", "entities"])
            document = _call(mention_annotator, "mentions", document, calls_file)
            document = _call(relation_annotator, "relations", document, calls_file)
            exporter.append(document)
        session["outcome"] = "completed"
    except Exception as e:
        session["outcome"] = "failed"
        session["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        session["finished_at"] = _now()
        manifest["totals"] = totals(directory)
        write_json(directory / "run.json", manifest)
