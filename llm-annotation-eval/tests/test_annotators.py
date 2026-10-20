import pathlib

import openai
import pytest

import autoannotate
from autoannotate import data

MENTION_DEFS = [autoannotate.MentionDefinition(type="Actor", description="who does it", priority=1)]
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


def test_annotate_sends_one_request_and_parses_the_answer(fake_client, make_doc):
    client = fake_client("clerk\tActor\t0", prompt_tokens=100, completion_tokens=5)

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert len(client.calls) == 1
    assert annotation.doc.mentions == [data.Mention("actor", (1,))]
    assert (annotation.prompt_tokens, annotation.completion_tokens) == (100, 5)


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

    def test_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        provider = autoannotate.get_provider("openrouter")

        assert str(provider.client().base_url).startswith("https://openrouter.ai/api/v1")
        # the gated medical dataset must not reach providers that may store or train on prompts
        assert provider.extra_body == {"provider": {"data_collection": "deny"}}
        assert provider.cache_instruction and provider.session_routing

    @pytest.mark.parametrize("name", ["openai", "nhr"])
    def test_only_openrouter_marks_the_cache_and_routes_by_session(self, name):
        provider = autoannotate.get_provider(name)
        assert not provider.cache_instruction and not provider.session_routing

    @pytest.mark.parametrize(
        "name, role", [("openai", "developer"), ("nhr", "system"), ("openrouter", "system")]
    )
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
    assert annotation.cached_tokens is None and annotation.cache_write_tokens is None
    assert annotation.issues == []


def test_annotate_reports_prompt_cache_use(fake_client, make_doc):
    client = fake_client("clerk\tActor\t0", prompt_tokens=1300, cached_tokens=1280, cache_write_tokens=0)

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.cached_tokens == 1280
    assert annotation.cache_write_tokens == 0


def test_annotate_reports_cost_and_upstream_provider(fake_client, make_doc):
    client = fake_client("clerk\tActor\t0", cost=0.0021, upstream_provider="Alibaba")

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.cost == 0.0021
    assert annotation.upstream_provider == "Alibaba"


def test_annotate_without_cost_or_upstream_provider(fake_client, make_doc):
    annotator = mention_annotator(fake_client("clerk\tActor\t0"))

    annotation = annotator.annotate(make_doc(["the clerk works"]))

    assert annotation.cost is None and annotation.upstream_provider is None


def test_cache_marker_ends_the_instruction_only(fake_client, make_doc):
    # Anthropic and Alibaba models cache the prompt up to the marker, so it goes after the
    # instruction every call shares and before the document, which differs
    annotator = mention_annotator(fake_client(""), cache_instruction=True)

    instruction, user = annotator.get_params(make_doc(["a"]))["messages"]

    assert instruction["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in part for part in user["content"])
    # the messages themselves are left as they are
    messages = annotator.get_messages(make_doc(["a"]))
    assert all("cache_control" not in part for m in messages for part in m["content"])


def test_no_cache_marker_by_default(fake_client, make_doc):
    params = mention_annotator(fake_client("")).get_params(make_doc(["a"]))
    assert all("cache_control" not in part for m in params["messages"] for part in m["content"])


def test_annotate_reports_cached_tokens_without_cache_writes(fake_client, make_doc):
    # OpenAI reports cache reads only
    client = fake_client("clerk\tActor\t0", prompt_tokens=1300, cached_tokens=0)

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.cached_tokens == 0
    assert annotation.cache_write_tokens is None


def test_annotate_handles_missing_answer(fake_client, make_doc):
    # e.g. a reasoning model that used all tokens before answering
    client = fake_client(lambda params: None, finish_reason="length")

    annotation = mention_annotator(client).annotate(make_doc(["the clerk works"]))

    assert annotation.response == ""
    assert annotation.doc.mentions == []
    assert annotation.finish_reason == "length"


def test_answer_cut_off_by_a_provider_error_is_not_used(fake_client, make_doc):
    # OpenRouter passes on what an upstream provider generated before it failed
    client = fake_client("The MPON\tActor\t0\nperform\t", finish_reason="error")

    with pytest.raises(autoannotate.UpstreamError, match="stopped with an error"):
        mention_annotator(client).annotate(make_doc(["the clerk works"]))


def test_issues_are_reset_for_every_document(fake_client, make_doc):
    annotator = mention_annotator(fake_client("chatter\nclerk\tActor\t0"))
    annotator.annotate(make_doc(["the clerk works"]))

    assert len(annotator.annotate(make_doc(["the clerk works"])).issues) == 1


def test_instruction_is_the_first_message(fake_client, make_doc):
    annotator = mention_annotator(fake_client(""))
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
