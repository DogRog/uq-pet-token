import pathlib

import openai
import pytest

import autoannotate
from autoannotate import data

MENTION_DEFS = [autoannotate.MentionDefinition(type="Actor", description="who does it", priority=1)]
RELATION_DEFS = [autoannotate.RelationDefinition(type="uses", description="what is used")]
PROMPT_DIR = pathlib.Path(__file__).parent.parent / "prompts"
TEMPLATE_VALUES = {
    "procedure": "PROCEDURE", "disambiguation": "DISAMBIGUATION", "example": "EXAMPLE",
}


def mention_annotator(client, **kwargs):
    return autoannotate.LLMMentionAnnotator(
        client=client, language_model="test-model", mention_definitions=MENTION_DEFS,
        template_values=TEMPLATE_VALUES, template=PROMPT_DIR / "mentions.txt",
        **{"reasoning_effort": None, **kwargs},
    )


def relation_annotator(client, **kwargs):
    return autoannotate.LLMRelationsAnnotator(
        client=client, language_model="test-model", type_definitions=RELATION_DEFS,
        template_values=TEMPLATE_VALUES, template=PROMPT_DIR / "relations.txt",
        **{"reasoning_effort": None, **kwargs},
    )


def message_text(message):
    return message["content"][0]["text"]


class TestParams:
    def test_instruction_role_replaces_developer(self, fake_client, make_doc):
        params = mention_annotator(fake_client(""), instruction_role="system").get_params(make_doc(["a"]))
        assert [m["role"] for m in params["messages"]] == ["system", "user"]

    def test_developer_role_by_default(self, fake_client, make_doc):
        params = mention_annotator(fake_client("")).get_params(make_doc(["a"]))
        assert [m["role"] for m in params["messages"]] == ["developer", "user"]
        assert params["model"] == "test-model"

    def test_reasoning_effort_none_is_omitted(self, fake_client, make_doc):
        params = mention_annotator(fake_client("")).get_params(make_doc(["a"]))
        assert "reasoning_effort" not in params

    def test_reasoning_effort_is_passed_on(self, fake_client, make_doc):
        params = mention_annotator(fake_client(""), reasoning_effort="low").get_params(make_doc(["a"]))
        assert params["reasoning_effort"] == "low"


def test_mention_prompt_contains_definitions_example_and_numbered_sentences(fake_client, make_doc):
    messages = mention_annotator(fake_client("")).get_messages(make_doc(["the clerk works", "he rests"]))

    instruction, user = map(message_text, messages)
    assert " - **Actor**: who does it" in instruction
    assert all(value in instruction for value in TEMPLATE_VALUES.values())
    assert "{{" not in instruction
    assert user.endswith("0: the clerk works\n1: he rests")


def test_relation_prompt_contains_tagged_mentions(fake_client, make_doc):
    doc = make_doc(["the clerk works"], mentions=[data.Mention("actor", (1,))])

    instruction, user = map(message_text, relation_annotator(fake_client("")).get_messages(doc))

    assert " - **uses**: what is used" in instruction
    assert "{{" not in instruction
    assert user.endswith("the <actor id=0> clerk </actor> works")


def test_annotate_sends_one_request_and_parses_the_answer(fake_client, make_doc):
    client = fake_client("clerk\tActor\t0", prompt_tokens=100, completion_tokens=5)

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert len(client.calls) == 1
    assert annotation.doc.mentions == [data.Mention("actor", (1,))]
    assert (annotation.prompt_tokens, annotation.completion_tokens) == (100, 5)


def test_end_to_end_mentions_then_relations(fake_client, make_doc):
    doc = make_doc(["the clerk checks the form"])

    doc = mention_annotator(fake_client("clerk\tActor\t0\nchecks\tActivity\t0")).annotate(doc).doc
    doc = relation_annotator(fake_client("actor performer\t1\t0")).annotate(doc).doc

    assert doc.mentions == [data.Mention("actor", (1,)), data.Mention("activity", (2,))]
    assert doc.relations == [data.Relation("actor performer", head_mention_index=1, tail_mention_index=0)]


class TestProviders:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            autoannotate.get_provider("anthropic")

    def test_client_needs_api_key(self, monkeypatch):
        monkeypatch.delenv("NHR_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NHR_API_KEY"):
            autoannotate.get_provider("nhr").client()

    def test_client_uses_provider_base_url(self, monkeypatch):
        monkeypatch.setenv("NHR_API_KEY", "test-key")
        client = autoannotate.get_provider("nhr").client()
        assert isinstance(client, openai.OpenAI)
        assert str(client.base_url).startswith("https://hub.nhr.fau.de/api/llmgw/v1")

    @pytest.mark.parametrize("name, role", [("openai", "developer"), ("nhr", "system")])
    def test_instruction_roles(self, name, role):
        assert autoannotate.get_provider(name).instruction_role == role


def test_annotate_returns_what_the_run_log_needs(fake_client, make_doc):
    client = fake_client("Mentions:\nclerk\tActor\t0", reasoning="the clerk acts", reasoning_tokens=3, finish_reason="stop")

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.response == "Mentions:\nclerk\tActor\t0"
    assert annotation.reasoning == "the clerk acts"
    assert annotation.reasoning_tokens == 3
    assert annotation.finish_reason == "stop"
    assert annotation.response_model == "test-model-2026-10-01"
    assert annotation.duration_s >= 0
    assert [i["kind"] for i in annotation.issues] == ["ignored_line"]


def test_annotate_without_reasoning_or_reasoning_token_count(fake_client, make_doc):
    annotation = mention_annotator(fake_client("clerk\tActor\t0")).annotate(make_doc(["the clerk works"]))

    assert annotation.reasoning is None
    assert annotation.reasoning_tokens is None
    assert annotation.issues == []


def test_annotate_handles_missing_answer(fake_client, make_doc):
    # e.g. a reasoning model that used all tokens before answering
    client = fake_client(lambda params: None, finish_reason="length")

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.response == ""
    assert annotation.doc.mentions == []
    assert annotation.finish_reason == "length"


def test_issues_are_reset_for_every_document(fake_client, make_doc):
    annotator = mention_annotator(fake_client("chatter\nclerk\tActor\t0"))
    annotator.annotate(make_doc(["the clerk works"]))

    assert len(annotator.annotate(make_doc(["the clerk works"])).issues) == 1


@pytest.mark.parametrize("make_annotator", [mention_annotator, relation_annotator])
def test_instruction_is_the_first_message(fake_client, make_doc, make_annotator):
    annotator = make_annotator(fake_client(""))
    assert annotator.get_messages(make_doc(["a"]))[0]["content"][0]["text"] == annotator.get_instruction()


def test_extra_body_is_sent_only_when_set(fake_client, make_doc):
    params = mention_annotator(fake_client(""), extra_body={"cache": {"no-cache": True}}).get_params(make_doc(["a"]))
    assert params["extra_body"] == {"cache": {"no-cache": True}}
    assert "extra_body" not in mention_annotator(fake_client("")).get_params(make_doc(["a"]))


def test_batch_line_puts_extra_body_into_the_body(fake_client, make_doc):
    line = mention_annotator(fake_client(""), extra_body={"cache": {"no-cache": True}}).batch_line(make_doc(["a"]), "t1")
    assert line["body"]["cache"] == {"no-cache": True}
    assert "extra_body" not in line["body"]


@pytest.mark.parametrize("name, extra_body", [("openai", None), ("nhr", {"cache": {"no-cache": True}})])
def test_only_the_nhr_gateway_bypasses_its_response_cache(name, extra_body):
    assert autoannotate.get_provider(name).extra_body == extra_body
