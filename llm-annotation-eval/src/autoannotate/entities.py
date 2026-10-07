import pathlib
import typing

import openai
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import ChatCompletionDeveloperMessageParam
from openai.types.chat import ChatCompletionUserMessageParam

from autoannotate import base
from autoannotate import data
from autoannotate import prompts
from autoannotate import util


class EntityParser(base.BaseParser):
    def parse(self, document: data.Document, string: str) -> data.Document:
        document = document.copy(clear=["entities"])
        for line in string.splitlines(keepends=False):
            if " " not in line:
                try:
                    mention_id = int(line)
                    if mention_id >= len(document.mentions):
                        continue
                    mention_ids = []
                except ValueError:
                    print(f"Skipping non space-separated line '{line}'!")
                    continue
            else:
                mention_ids = []
                for i in line.split(" "):
                    try:
                        mention_id = int(i)
                        if mention_id >= len(document.mentions):
                            continue
                        mention_ids.append(mention_id)
                    except ValueError:
                        pass
            mentions = []
            for i in mention_ids:
                if i >= len(document.mentions):
                    continue
                mentions.append(document.mentions[i])
            mention_types = set(m.type for m in mentions)
            if len(mention_types) > 1:
                print(f"Extracted multi-type entity, with mentions {mentions}.")
            document.entities.append(data.Entity(mention_indices=tuple(mention_ids)))
        for i, mention in enumerate(document.mentions):
            if any([i in e.mention_indices for e in document.entities]):
                continue
            document.entities.append(data.Entity(mention_indices=(i,)))
        return document


class LLMEntitiesAnnotator(base.BaseAnnotator):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 reasoning_effort: base.ReasoningEffort,
                 instruction_role: typing.Literal["developer", "system"] = "developer"):
        super().__init__(client, language_model, reasoning_effort, instruction_role)
        prompt_folder = pathlib.Path(__file__).parent.resolve() / "resources" / "prompts"
        self._prompt_template = prompts.Prompt(prompt_folder / "annotate-entities.txt")

    def get_messages(
            self,
            document: data.Document
    ) -> typing.List[typing.Union[ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam]]:
        text = self._format_text(document)
        prompt = self._prompt_template()

        return [
            ChatCompletionDeveloperMessageParam(
                role="developer",
                content=[
                    ChatCompletionContentPartTextParam(
                        text=prompt,
                        type="text"
                    )
                ]
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(
                        text=f"Here is the text in question: \n\n{text}",
                        type="text"
                    )
                ]
            )
        ]

    @staticmethod
    def _format_text(document: data.Document) -> str:
        return util.format_document_text_with_entity_mentions(
            document=document,
            only_types=["Activity Data", "Actor"]
        )

    def get_parser(self) -> base.BaseParser:
        return EntityParser()
