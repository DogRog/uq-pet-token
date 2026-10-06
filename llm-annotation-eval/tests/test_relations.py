import pytest

from autoannotate import RelationParser, data


@pytest.fixture
def doc(make_doc):
    return make_doc(
        ["the clerk checks the form"],
        mentions=[data.Mention("actor", (1,)), data.Mention("activity", (2,)), data.Mention("activity data", (4,))],
    )


def parse(doc, *lines):
    return RelationParser().parse(doc, "\n".join(lines)).relations


def test_parses_type_head_and_tail(doc):
    assert parse(doc, "Actor Performer\t1\t0", "uses\t1\t2") == [
        data.Relation("actor performer", head_mention_index=1, tail_mention_index=0),
        data.Relation("uses", head_mention_index=1, tail_mention_index=2),
    ]


def test_explanations_may_contain_tabs(doc):
    assert parse(doc, "uses\t1\t2\tthe form\tis checked") == [data.Relation("uses", 1, 2)]


@pytest.mark.parametrize("line", [
    "uses 1 2",  # not tab separated
    "uses\t1",  # too few fields
    "uses\tone\t2",  # head not a number
    "uses\t1\t3",  # tail out of range
    "uses\t3\t1",  # head out of range
    "uses\t-2\t2",  # negative head
    "uses\t1\t-1",  # negative tail
])
def test_skips_invalid_lines(doc, line):
    assert parse(doc, line, "flow\t1\t1") == [data.Relation("flow", 1, 1)]


def test_replaces_existing_relations_and_keeps_mentions(doc):
    doc.relations.append(data.Relation("flow", 0, 1))

    parsed = RelationParser().parse(doc, "uses\t1\t2")

    assert parsed.relations == [data.Relation("uses", 1, 2)]
    assert parsed.mentions == doc.mentions
