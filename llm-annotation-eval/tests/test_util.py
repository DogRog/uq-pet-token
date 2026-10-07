from autoannotate import data
from autoannotate import util


def test_ner_to_tag_replaces_spaces_and_adds_attributes():
    assert util.ner_to_tag("activity data", {}) == ("<activity_data>", "</activity_data>")
    assert util.ner_to_tag("actor", {"id": "3"}) == ("<actor id=3>", "</actor>")


def test_format_tags_mentions_with_their_list_index(make_doc):
    # list order differs from text order, ids must still be the index into document.mentions
    doc = make_doc(
        ["the clerk checks the form"],
        mentions=[data.Mention("activity", (2,)), data.Mention("activity data", (3, 4)), data.Mention("actor", (1,))],
    )

    assert util.format_document_text_with_entity_mentions(doc) == (
        "the <actor id=2> clerk </actor> <activity id=0> checks </activity> "
        "<activity_data id=1> the form </activity_data>"
    )


def test_format_only_tags_selected_types(make_doc):
    doc = make_doc(
        ["the clerk checks"],
        mentions=[data.Mention("actor", (1,)), data.Mention("activity", (2,))],
    )

    assert util.format_document_text_with_entity_mentions(doc, only_types=["Activity"]) == (
        "the clerk <activity id=1> checks </activity>"
    )


def test_document_from_string_splits_sentences():
    doc = util.document_from_string("The clerk checks the form. Then the manager signs it.", "doc")

    assert [t.text for t in doc.sentences[0]] == ["The", "clerk", "checks", "the", "form", "."]
    assert len(doc.sentences) == 2
    assert [t.index_in_document for t in doc.tokens] == list(range(len(doc.tokens)))
