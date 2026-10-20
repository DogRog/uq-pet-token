import types
import typing

import pytest

from autoannotate import data


def make_document(
        sentences: typing.List[str],
        mentions: typing.Sequence[data.Mention] = (),
        doc_id: str = "doc-1",
) -> data.Document:
    """Builds a document from whitespace-tokenized sentences."""
    tokens = []
    for sentence_index, sentence in enumerate(sentences):
        for text in sentence.split():
            tokens.append(data.Token(
                text=text,
                index_in_document=len(tokens),
                pos_tag="X",
                sentence_index=sentence_index,
            ))
    return data.Document(
        id=doc_id,
        name=doc_id,
        text=" ".join(t.text for t in tokens),
        category="",
        tokens=tokens,
        mentions=list(mentions),
        entities=[],
    )


@pytest.fixture
def make_doc():
    return make_document


class FakeClient:
    """
    Stands in for openai.OpenAI.

    :param answer: the answer to every chat completion, or a function from the request params to
                   the answer, which may raise to simulate a failed request
    :param reasoning: reasoning text returned next to the answer, like vLLM's reasoning_content
    :param reasoning_tokens: None to not report them, like most open-weight models
    :param cached_tokens: prompt tokens read from the provider's prompt cache; None to not report
                          them, like the NHR@FAU gateway
    :param cache_write_tokens: prompt tokens written to the cache, which OpenRouter reports next to
                               cached_tokens for some models
    :param cost: what the call cost in USD, which OpenRouter reports in the usage
    :param upstream_provider: who served the call, which OpenRouter reports as "provider"
    """

    def __init__(self, answer: typing.Union[str, typing.Callable[[typing.Dict], str]],
                 prompt_tokens: int = 11, completion_tokens: int = 7,
                 reasoning: typing.Optional[str] = None, reasoning_tokens: typing.Optional[int] = None,
                 finish_reason: str = "stop", cached_tokens: typing.Optional[int] = None,
                 cache_write_tokens: typing.Optional[int] = None,
                 cost: typing.Optional[float] = None,
                 upstream_provider: typing.Optional[str] = None):
        self.calls: typing.List[typing.Dict] = []
        self._answer = answer if callable(answer) else lambda params: answer
        details = types.SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
        prompt_details = None
        if cached_tokens is not None:
            # fields the openai package doesn't know end up in model_extra
            extra = {} if cache_write_tokens is None else {"cache_write_tokens": cache_write_tokens}
            prompt_details = types.SimpleNamespace(cached_tokens=cached_tokens, model_extra=extra)
        self._usage = types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=details,
            prompt_tokens_details=prompt_details,
            model_extra={"cost": cost} if cost is not None else {},
        )
        self._response_extra = {} if upstream_provider is None else {"provider": upstream_provider}
        self._model_extra = {"reasoning_content": reasoning} if reasoning is not None else {}
        self._finish_reason = finish_reason
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        message = types.SimpleNamespace(content=self._answer(params), model_extra=self._model_extra)
        choice = types.SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return types.SimpleNamespace(
            choices=[choice],
            usage=self._usage,
            model=params["model"] + "-2026-10-01",
            model_extra=self._response_extra,
        )


@pytest.fixture
def fake_client():
    return FakeClient
