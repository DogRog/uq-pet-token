import json
import pathlib

from autoannotate import data

GOLD_FILE = pathlib.Path(__file__).parent.parent / "res" / "gold" / "pet.jsonl"


def test_sentences_group_tokens_by_sentence_index(make_doc):
    doc = make_doc(["a b c", "d e"])
    assert [[t.text for t in s] for s in doc.sentences] == [["a", "b", "c"], ["d", "e"]]


def test_copy_clears_requested_fields_and_normalizes_types(make_doc):
    doc = make_doc(
        ["the clerk checks"],
        mentions=[data.Mention(type=" Actor ", token_document_indices=(1,))],
        relations=[data.Relation(type="Actor Performer", head_mention_index=0, tail_mention_index=0)],
    )

    copied = doc.copy(clear=["relations"])

    assert copied.relations == []
    assert copied.mentions == [data.Mention(type="actor", token_document_indices=(1,))]
    assert copied.mentions[0].type == "actor"
    assert copied.tokens == doc.tokens and copied.tokens is not doc.tokens


def test_remove_mention_drops_its_relations_and_reindexes_the_rest(make_doc):
    doc = make_doc(
        ["a b c d"],
        mentions=[data.Mention("actor", (0,)), data.Mention("activity", (1,)), data.Mention("activity data", (2,))],
        relations=[
            data.Relation("actor performer", head_mention_index=1, tail_mention_index=0),
            data.Relation("uses", head_mention_index=1, tail_mention_index=2),
        ],
    )

    doc.remove_mention(0)

    assert doc.mentions == [data.Mention("activity", (1,)), data.Mention("activity data", (2,))]
    assert doc.relations == [data.Relation("uses", head_mention_index=0, tail_mention_index=1)]


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
        relations=[data.Relation("uses", head_mention_index=1, tail_mention_index=2)],
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
    raw["relations"] = [{"type": " Actor Performer", "headMentionIndex": 0, "tailMentionIndex": 0}]
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(raw))

    [imported] = data.JsonImporter(path).do_import()

    assert imported.mentions[0].type == "activity data"
    assert imported.relations[0].type == "actor performer"


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


def test_gold_file_loads_with_consistent_annotations():
    documents = data.JsonImporter(GOLD_FILE).do_import()

    assert len(documents) == 45
    assert len({d.id for d in documents}) == len(documents)
    for doc in documents:
        assert [t.index_in_document for t in doc.tokens] == list(range(len(doc.tokens)))
        for mention in doc.mentions:
            assert all(0 <= i < len(doc.tokens) for i in mention.token_document_indices)
        for relation in doc.relations:
            assert 0 <= relation.head_mention_index < len(doc.mentions)
            assert 0 <= relation.tail_mention_index < len(doc.mentions)
