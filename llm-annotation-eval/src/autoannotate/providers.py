import dataclasses
import os
import typing

import openai


@dataclasses.dataclass(frozen=True)
class Provider:
    name: str
    base_url: typing.Optional[str]
    api_key_env: str
    # role of the instruction message: OpenAI prefers "developer", while chat templates
    # of open-weight models (e.g., served via vLLM) usually only know "system"
    instruction_role: typing.Literal["developer", "system"]
    # added to the body of every request
    extra_body: typing.Optional[typing.Dict[str, typing.Any]] = None
    # seconds to wait for a response; None: the openai client's default (10 minutes)
    timeout: typing.Optional[float] = None
    # mark the end of the instruction for the prompt cache, which some models only use with a
    # marker (see BaseAnnotator.get_params)
    cache_instruction: bool = False
    # send a session id per run, so the calls of a run reach the same upstream server and its cache
    session_routing: bool = False

    def client(self) -> openai.OpenAI:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} (e.g., in .env) to use the {self.name} provider.")
        if self.timeout is None:
            return openai.OpenAI(base_url=self.base_url, api_key=api_key)
        return openai.OpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout)


PROVIDERS: typing.Dict[str, Provider] = {
    "openai": Provider(
        name="OpenAI",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        instruction_role="developer",
    ),
    # OpenAI-compatible LLM gateway of NHR@FAU, see https://hpc.fau.de/request-llm-api-key/
    "nhr": Provider(
        name="NHR@FAU",
        base_url="https://hub.nhr.fau.de/api/llmgw/v1",
        api_key_env="NHR_API_KEY",
        instruction_role="system",
        # the gateway (LiteLLM) caches responses, an identical request would return the same
        # answer again within milliseconds, which makes repeated runs copies of the first
        extra_body={"cache": {"no-cache": True}},
        # with high reasoning, e.g. gpt-oss-120b takes over 10 minutes on the longest medical
        # guidelines, which the client's default timeout gives up on
        timeout=1800,
    ),
    # routes each request to one of the upstream providers that serve the model, see
    # https://openrouter.ai/docs/features/provider-routing; configs can pin them with "upstream"
    "openrouter": Provider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        instruction_role="system",
        # only providers that don't store or train on prompts: the medical dataset is gated
        extra_body={"provider": {"data_collection": "deny"}},
        timeout=1800,
        cache_instruction=True,
        session_routing=True,
    ),
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}', expected one of {list(PROVIDERS)}.")
    return PROVIDERS[name]
