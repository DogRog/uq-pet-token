import abc
import dataclasses
import typing

import openai
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionDeveloperMessageParam

from autoannotate import data


@dataclasses.dataclass
class LLMAnnotation:
    doc: data.Document
    prompt_tokens: int
    completion_tokens: int


class BaseParser(abc.ABC):
    def parse(self, document: data.Document, string: str) -> data.Document:
        raise NotImplementedError()


class BaseAnnotator(abc.ABC):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 reasoning_effort: typing.Literal["none", "minimal", "low", "medium", "high"]):
        self.client = client
        self.language_model = language_model
        self.parser = self.get_parser()
        self.reasoning_effort = reasoning_effort

    @abc.abstractmethod
    def get_messages(
            self,
            document: data.Document
    ) -> typing.List[typing.Union[ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_parser(self) -> BaseParser:
        raise NotImplementedError()

    def get_params(self, document: data.Document):
        return {
            "messages": self.get_messages(document),
            "model": self.language_model,
            "reasoning_effort": self.reasoning_effort,
        }

    def batch_line(self, document: data.Document, task_id: str) -> typing.Dict:
        params = self.get_params(document)
        return {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                **params
            }
        }

    def annotate(self, document: data.Document) -> LLMAnnotation:
        resp = self.client.chat.completions.create(
            **self.get_params(document)
        )
        answer = resp.choices[0].message.content
        doc = self.parser.parse(document, answer)
        return LLMAnnotation(
            doc=doc,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
