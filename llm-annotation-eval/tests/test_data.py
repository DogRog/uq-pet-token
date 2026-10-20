import json

from autoannotate import data


def test_sentences_group_tokens_by_sentence_index(make_doc):
    doc = make_doc(["a b c", "d e"])
    assert [[t.text for t in s] for s in doc.sentences] == [["a", "b", "c"], ["d", "e"]]


def test_copy_clears_requested_fields_and_normalizes_types(make_doc):
    doc = make_doc(
        ["the clerk checks"],
        mentions=[data.Mention(type=" Actor ", token_document_indices=(1,))],
    )

    copied = doc.copy(clear=["entities"])

    assert doc.copy(clear=["mentions"]).mentions == []
    assert copied.mentions == [data.Mention(type="actor", token_document_indices=(1,))]
    assert copied.mentions[0].type == "actor"
    assert copied.tokens == doc.tokens and copied.tokens is not doc.tokens


def test_mention_equality_ignores_case_and_index_order():
    a = data.Mention("Actor", (2, 1))
    b = data.Mention("actor", (1, 2))
    assert a == b
    assert hash(a) == hash(b)
    assert a != data.Mention("activity", (1, 2))


def test_mention_match_requires_same_type_and_any_overlap():
    m = data.Mention("actor", (1, 2))
    assert m.match(data.Mention("Actor", (2, 3)))
    assert not m.match(data.Mention("actor", (3, 4)))
    assert not m.match(data.Mention("activity", (1, 2)))


def _document_with_annotations(make_doc):
    return make_doc(
        ["the clerk checks the form", "then he signs"],
        mentions=[data.Mention("actor", (1,)), data.Mention("activity", (2,)), data.Mention("activity data", (3, 4))],
    )


def test_export_import_round_trip(tmp_path, make_doc):
    doc = _document_with_annotations(make_doc)
    path = tmp_path / "docs.jsonl"

    data.JsonExporter(str(path)).export([doc, doc.copy(clear=[])])

    imported = data.JsonImporter(path).do_import()
    assert len(imported) == 2
    assert imported[0] == doc


def test_importer_lowercases_types(tmp_path, make_doc):
    doc = make_doc(["the clerk checks"])
    raw = data.DictExporter().export_document(doc)
    raw["mentions"] = [{"type": "Activity Data", "tokenDocumentIndices": [1]}]
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(raw))

    [imported] = data.JsonImporter(path).do_import()

    assert imported.mentions[0].type == "activity data"


def test_importer_ignores_relations_of_older_prediction_files(tmp_path, make_doc):
    raw = data.DictExporter().export_document(make_doc(["the clerk checks"]))
    assert "relations" not in raw
    raw["relations"] = [{"type": "uses", "headMentionIndex": 0, "tailMentionIndex": 0}]
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(raw))

    [imported] = data.JsonImporter(path).do_import()

    assert imported.tokens == make_doc(["the clerk checks"]).tokens


def test_append_creates_file_and_adds_one_line_per_document(tmp_path, make_doc):
    path = tmp_path / "pred.jsonl"
    exporter = data.JsonExporter(str(path))

    exporter.append(make_doc(["a"], doc_id="first"))
    exporter.append(make_doc(["b"], doc_id="second"))

    assert path.read_text().count("\n") == 2
    assert [d.id for d in data.JsonImporter(path).do_import()] == ["first", "second"]


def test_append_after_export_starts_a_new_line(tmp_path, make_doc):
    # export() writes no trailing newline, append() must not glue its line onto the last one
    path = tmp_path / "pred.jsonl"
    exporter = data.JsonExporter(str(path))
    exporter.export([make_doc(["a"], doc_id="first")])

    exporter.append(make_doc(["b"], doc_id="second"))

    assert [d.id for d in data.JsonImporter(path).do_import()] == ["first", "second"]
