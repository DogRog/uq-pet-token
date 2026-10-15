import abc
import dataclasses
import time
import typing

import openai
from openai.types.chat import ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam

from autoannotate import data


@dataclasses.dataclass
class LLMAnnotation:
    doc: data.Document
    prompt_tokens: int
    completion_tokens: int
    # None if the provider does not report it separately
    reasoning_tokens: typing.Optional[int]
    # part of prompt_tokens read from the provider's prompt cache; None if the provider does not
    # report it, like the NHR@FAU gateway
    cached_tokens: typing.Optional[int]
    # part of prompt_tokens written to the prompt cache, which some providers charge extra for;
    # None if not reported (OpenAI never does, OpenRouter for some models)
    cache_write_tokens: typing.Optional[int]
    # what the call cost in USD, None if the provider doesn't say (OpenRouter does)
    cost: typing.Optional[float]
    # who served the call behind a router like OpenRouter, None for a direct provider
    upstream_provider: typing.Optional[str]
    # raw answer as returned by the LLM, before parsing
    response: str
    # reasoning text, if the provider returns it separately from the answer
    reasoning: typing.Optional[str]
    # e.g. "stop", or "length" if the answer was cut off
    finish_reason: typing.Optional[str]
    # model id as reported by the provider, may be more specific than the requested one
    response_model: str
    duration_s: float
    # problems the parser found in the response, see BaseParser.report
    issues: typing.List[typing.Dict[str, str]]


class UpstreamError(Exception):
    """The provider stopped generating with an error, so the answer is cut off and not used.

    OpenRouter passes on what an upstream provider generated before it failed, with the finish
    reason "error"; annotating the document again later may work.
    """


class BaseParser(abc.ABC):
    def __init__(self):
        self.issues: typing.List[typing.Dict[str, str]] = []

    def parse(self, document: data.Document, string: str) -> data.Document:
        raise NotImplementedError()

    def report(self, kind: str, message: str) -> None:
        """Records a problem with the parsed response, e.g. a line that had to be dropped.

        :param kind: category to count problems by, prefixed with what happened to the line:
                     "dropped_..." (not used), "placed_..." (used anyway) or "ignored_line"
        """
        print(message)
        self.issues.append({"kind": kind, "message": message})


ReasoningEffort = typing.Optional[typing.Literal["none", "minimal", "low", "medium", "high"]]


class BaseAnnotator(abc.ABC):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 reasoning_effort: ReasoningEffort,
                 instruction_role: typing.Literal["developer", "system"] = "developer",
                 extra_body: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 cache_instruction: bool = False):
        self.client = client
        self.language_model = language_model
        self.parser = self.get_parser()
        # None omits the parameter, for models / providers that do not support it
        self.reasoning_effort = reasoning_effort
        self.instruction_role = instruction_role
        # provider specific fields added to the request body, see Provider.extra_body
        self.extra_body = extra_body
        # mark the end of the instruction for the prompt cache, see Provider.cache_instruction
        self.cache_instruction = cache_instruction

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
        messages = [
            self._instruction_message(m) if m["role"] == "developer" else m
            for m in self.get_messages(document)
        ]
        params = {
            "messages": messages,
            "model": self.language_model,
        }
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            params["extra_body"] = self.extra_body
        return params

    def _instruction_message(self, message):
        """The instruction in the provider's role, with the cache marker after it if wanted.

        Every call starts with the same instruction and only the document after it differs, so
        models that cache the prompt up to a marker (Anthropic's, Alibaba's) cache just that.
        """
        message = {**message, "role": self.instruction_role}
        if self.cache_instruction:
            *parts, last = message["content"]
            message["content"] = [*parts, {**last, "cache_control": {"type": "ephemeral"}}]
        return message

    def batch_line(self, document: data.Document, task_id: str) -> typing.Dict:
        params = self.get_params(document)
        extra_body = params.pop("extra_body", {})
        return {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                **params,
                **extra_body,
            }
        }

    def annotate(self, document: data.Document) -> LLMAnnotation:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            **self.get_params(document)
        )
        duration_s = time.perf_counter() - start
        choice = resp.choices[0]
        # None if e.g. a reasoning model used up all tokens before answering
        answer = choice.message.content or ""
        if choice.finish_reason == "error":
            raise UpstreamError(
                f"The provider stopped with an error after {resp.usage.completion_tokens} tokens: "
                f"{answer[-80:]!r}"
            )
        # a parser per call, as it collects the issues of one response and documents may be
        # annotated in parallel
        parser = self.get_parser()
        doc = parser.parse(document, answer)
        # OpenAI-compatible servers return reasoning under a non-standard field, if at all
        extra = choice.message.model_extra or {}
        details = resp.usage.completion_tokens_details
        prompt_details = resp.usage.prompt_tokens_details
        # not a field of the openai package, so among the extra fields
        prompt_extra = (prompt_details.model_extra or {}) if prompt_details is not None else {}
        # OpenRouter adds the cost to the usage and the upstream provider to the response
        usage_extra = getattr(resp.usage, "model_extra", None) or {}
        response_extra = getattr(resp, "model_extra", None) or {}
        return LLMAnnotation(
            doc=doc,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            reasoning_tokens=details.reasoning_tokens if details is not None else None,
            cached_tokens=prompt_details.cached_tokens if prompt_details is not None else None,
            cache_write_tokens=prompt_extra.get("cache_write_tokens"),
            cost=usage_extra.get("cost"),
            upstream_provider=response_extra.get("provider"),
            response=answer,
            reasoning=extra.get("reasoning_content") or extra.get("reasoning"),
            finish_reason=choice.finish_reason,
            response_model=resp.model,
            duration_s=duration_s,
            issues=parser.issues,
        )
