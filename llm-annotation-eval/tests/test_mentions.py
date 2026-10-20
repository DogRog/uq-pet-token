import pytest

from autoannotate import MentionParser

SENTENCES = [
    "the clerk checks the form and the clerk signs the form .",
    "then the manager approves it .",
]


@pytest.fixture
def doc(make_doc):
    return make_doc(SENTENCES)


def parse_line(line):
    return MentionParser.parse_line(line, len(SENTENCES))


def parse(doc, *lines):
    parsed = MentionParser().parse(doc, "\n".join(lines))
    return [(m.type, m.token_document_indices) for m in parsed.mentions]


class TestParseLine:
    def test_three_fields(self):
        assert parse_line("Clerk\tActor\t0") == ("Clerk", "actor", 0)

    def test_fourth_field_is_an_ignored_explanation(self):
        assert parse_line("clerk\tActor\t0\tdoes the work") == ("clerk", "actor", 0)

    def test_empty_fields_are_dropped(self):
        assert parse_line("clerk\t\tActor\t0") == ("clerk", "actor", 0)

    @pytest.mark.parametrize("line", [
        "clerk\tActor",  # too few fields
        "clerk\tActor\t0\twhy\textra",  # too many fields
        "clerk\tActor\tfirst",  # sentence id not a number
        "clerk\tActor\t2",  # sentence id out of range
        "clerk\tActor\t-1",
    ])
    def test_invalid_lines_raise(self, line):
        with pytest.raises(ValueError):
            parse_line(line)


class TestParse:
    def test_places_mentions_on_tokens(self, doc):
        assert parse(doc, "clerk\tActor\t0", "the manager approves\tActivity\t1") == [
            ("actor", (1,)),
            ("activity", (13, 14, 15)),
        ]

    def test_matches_case_insensitively(self, doc):
        assert parse(doc, "CLERK\tActor\t0") == [("actor", (1,))]

    def test_sentence_index_selects_the_sentence(self, make_doc):
        doc = make_doc(["the clerk works", "the clerk rests"])
        assert parse(doc, "clerk\tActor\t1") == [("actor", (4,))]

    def test_repeated_text_resolves_to_successive_occurrences(self, doc):
        assert parse(doc, "clerk\tActor\t0", "clerk\tActor\t0") == [("actor", (1,)), ("actor", (7,))]

    def test_more_repetitions_than_occurrences_are_dropped(self, doc):
        assert parse(doc, *["clerk\tActor\t0"] * 3) == [("actor", (1,)), ("actor", (7,))]

    def test_lines_out_of_text_order_keep_llm_order(self, doc):
        assert parse(doc, "signs\tActivity\t0", "checks\tActivity\t0") == [
            ("activity", (8,)),
            ("activity", (2,)),
        ]

    def test_text_not_in_sentence_is_dropped(self, doc):
        assert parse(doc, "secretary\tActor\t1", "manager\tActor\t1") == [("actor", (14,))]

    def test_text_in_another_sentence_is_dropped(self, doc):
        assert parse(doc, "manager\tActor\t0") == []

    def test_skips_lines_without_tabs_and_malformed_lines(self, doc):
        lines = [
            "Here are the mentions:",
            "",
            "clerk\tActor\t7",
            "clerk\tActor\tx",
            "manager\tActor\t1\tbecause it is a role",
        ]
        assert parse(doc, *lines) == [("actor", (14,))]

    def test_keeps_document_but_drops_previous_annotations(self, doc):
        from autoannotate import data
        doc.mentions.append(data.Mention("activity", (2,)))

        parsed = MentionParser().parse(doc, "clerk\tActor\t0")

        assert parsed.id == doc.id and parsed.tokens == doc.tokens
        assert parsed.mentions == [data.Mention("actor", (1,))]

    @pytest.mark.xfail(strict=True, reason=(
        "LangExtract's aligner can report two lines as exact matches on the same span when the "
        "LLM output is not in text order, and MentionParser assumes exact matches never overlap"
    ))
    def test_repeated_text_after_a_later_mention_does_not_duplicate_a_mention(self, doc):
        # "the form" occurs at tokens 4 and 10, but the second clerk (token 7) is listed first
        mentions = parse(doc, "clerk\tActor\t0", "clerk\tActor\t0", "form\tActivity Data\t0", "form\tActivity Data\t0")
        assert len(set(mentions)) == len(mentions)



@pytest.mark.parametrize("answer, kind", [
    ("Here are the mentions:", "ignored_line"),
    ("clerk\tActor\tx", "dropped_malformed"),
    ("secretary\tActor\t1", "dropped_not_found"),
    ("clerk\tActor\t0\nclerk\tActor\t0\nclerk\tActor\t0", "dropped_no_free_occurrence"),
    ("signs\tActivity\t0\nchecks\tActivity\t0", "placed_out_of_order"),
    ("the manager approve\tActivity\t1", "placed_fuzzy"),
])
def test_reports_issues_by_kind(doc, answer, kind):
    parser = MentionParser()

    parser.parse(doc, answer)

    assert [i["kind"] for i in parser.issues] == [kind]
    assert parser.issues[0]["message"]


def test_no_issues_for_a_clean_answer(doc):
    parser = MentionParser()
    parser.parse(doc, "clerk\tActor\t0\nchecks\tActivity\t0")
    assert parser.issues == []
