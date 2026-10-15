import pathlib
import typing

import openai
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam,
)

from autoannotate import base, data, definitions, prompts, util


class RelationParser(base.BaseParser):
    def parse(self, document: data.Document, string: str) -> data.Document:
        document = document.copy(clear=["relations"])
        for line in string.splitlines(keepends=False):
            if line.strip() == "":
                continue
            if "\t" not in line:
                self.report("ignored_line", f"Skipping non-tab-separated line '{line}'.")
                continue
            split_line = line.split("\t")
            if len(split_line) >= 3:
                # anything after the indices is an explanation (which may contain tabs) and unused
                relation_type, head_index, tail_index = split_line[:3]
            else:
                self.report(
                    "dropped_malformed",
                    f"Skipping line '{line}': expected 3 or 4 tab-separated fields, got {len(split_line)}."
                )
                continue
            relation_type = relation_type.lower().strip()
            try:
                head_index = int(head_index)
                tail_index = int(tail_index)
            except ValueError:
                self.report("dropped_malformed", f"Skipping line '{line}': mention ids are not numbers.")
                continue
            # negative indices would silently refer to mentions from the end of the list
            if not (0 <= head_index < len(document.mentions) and 0 <= tail_index < len(document.mentions)):
                self.report(
                    "dropped_unknown_mention",
                    f"Skipping line '{line}': there are only {len(document.mentions)} mentions."
                )
                continue
            document.relations.append(
                data.Relation(
                    type=relation_type,
                    head_mention_index=head_index,
                    tail_mention_index=tail_index,
                )
            )
        return document


class LLMRelationsAnnotator(base.BaseAnnotator):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 type_definitions: typing.List[definitions.RelationDefinition],
                 template_values: typing.Dict[str, str],
                 reasoning_effort: base.ReasoningEffort,
                 template: pathlib.Path,
                 instruction_role: typing.Literal["developer", "system"] = "developer",
                 extra_body: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 cache_instruction: bool = False):
        super().__init__(
            client, language_model, reasoning_effort, instruction_role, extra_body,
            cache_instruction,
        )
        self._prompt_template = prompts.Prompt(template)
        self._template_values = template_values
        self._defs = type_definitions

    def get_instruction(self) -> str:
        """The instruction message, the same for every document.

        The template gets the type definitions and the dataset's template values, e.g. its
        procedure, disambiguation hints and format example.
        """
        type_definitions = "\n".join(f" - {f.for_llm()}" for f in self._defs)
        return self._prompt_template(type_definitions=type_definitions, **self._template_values)

    def get_messages(
            self,
            document: data.Document
    ) -> typing.List[typing.Union[ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam]]:
        text = util.format_document_text_with_entity_mentions(document)
        prompt = self.get_instruction()

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

    def get_parser(self) -> base.BaseParser:
        return RelationParser()
