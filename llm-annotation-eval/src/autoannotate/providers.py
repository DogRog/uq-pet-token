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

    def client(self) -> openai.OpenAI:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} (e.g., in .env) to use the {self.name} provider.")
        return openai.OpenAI(base_url=self.base_url, api_key=api_key)


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
    ),
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}', expected one of {list(PROVIDERS)}.")
    return PROVIDERS[name]
