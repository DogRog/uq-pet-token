import types
import typing

import pytest

from autoannotate import data


def make_document(
        sentences: typing.List[str],
        mentions: typing.Sequence[data.Mention] = (),
        relations: typing.Sequence[data.Relation] = (),
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
        relations=list(relations),
        entities=[],
    )


@pytest.fixture
def make_doc():
    return make_document


class FakeClient:
    """Stands in for openai.OpenAI, answers every chat completion with a fixed text."""

    def __init__(self, answer: str, prompt_tokens: int = 11, completion_tokens: int = 7):
        self.calls: typing.List[typing.Dict] = []
        self._answer = answer
        self._usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        message = types.SimpleNamespace(content=self._answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=self._usage)


@pytest.fixture
def fake_client():
    return FakeClient
