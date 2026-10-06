import pathlib
import traceback
import typing

import openai
from openai.types.chat import ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionContentPartTextParam

from autoannotate import base, definitions, data, prompts


class MentionParser(base.BaseParser):
    @staticmethod
    def parse_line(
            line: str, document: data.Document
    ) -> typing.List[data.Mention]:
        split_line = line.split("\t")
        split_line = tuple(e for e in split_line if e.strip() != "")

        if len(split_line) < 3 or len(split_line) > 4:
            print("Malformed line:", line)
            raise ValueError(
                f"Skipping line {split_line}, as it is not formatted "
                f"properly, expected between 3 and 4 arguments."
            )

        if len(split_line) == 3:
            mention_text, mention_type, sentence_id = split_line
        else:
            mention_text, mention_type, sentence_id, explanation = split_line
            # print(f"Explanation for {mention_text}: {explanation}")

        try:
            sentence_id = int(sentence_id)
        except ValueError:
            print(f"No numerical sentence id in line: {line}")
            raise ValueError(f"Invalid sentence index '{sentence_id}', skipping line.")

        sentence = document.sentences[sentence_id]

        mention_text = mention_text.lower()
        mention_tokens = mention_text.split(" ")

        res = []
        found = False
        for i, token in enumerate(sentence):
            candidates = sentence[i: i + len(mention_tokens)]
            candidate_text = " ".join(c.text.lower() for c in candidates)

            if candidate_text.lower() != mention_text.lower():
                continue

            res.append(
                data.Mention(
                    token_document_indices=tuple(
                        c.index_in_document for c in candidates
                    ),
                    type=mention_type.lower().strip(),
                )
            )
            found = True
        if not found:
            print(f"Did not find predicted mention '{mention_text}'.")
        return res

    def parse(self, document: data.Document, string: str) -> data.Document:
        parsed_mentions: typing.List[data.Mention] = []
        num_parse_errors = 0
        for line in string.splitlines(keepends=False):
            line = line.strip()
            if line == "":
                continue
            if "\t" not in line:
                continue

            try:
                mentions_from_line = self.parse_line(line, document)
                parsed_mentions.extend(mentions_from_line)
            except Exception:
                num_parse_errors += 1
                print("Error during parsing of line, skipping line. Error was:")
                print(traceback.format_exc())

        doc = data.Document(
            id=document.id,
            text=document.text,
            name=document.name,
            category=document.category,
            tokens=[t.copy() for t in document.tokens],
            mentions=parsed_mentions,
            relations=[],
            entities=[],
        )
        return doc


class LLMMentionAnnotator(base.BaseAnnotator):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 mention_definitions: typing.List[definitions.MentionDefinition],
                 example: typing.Optional[str],
                 reasoning_effort: typing.Literal["none", "minimal", "low", "medium", "high"]):
        super().__init__(client, language_model, reasoning_effort)
        prompt_folder = pathlib.Path(__file__).parent.resolve() / "resources" / "prompts"
        self._prompt_template = prompts.Prompt(prompt_folder / "annotate-mentions.txt")
        self._defs = mention_definitions
        self._example = example

    def get_messages(
            self,
            document: data.Document
    ) -> typing.List[typing.Union[ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam]]:
        text = self._format(document)
        example = self._example if self._example is not None else ""
        type_definitions = "\n".join(f" - {f.for_llm()}" for f in self._defs)
        prompt = self._prompt_template(type_definitions=type_definitions, example=example)

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
        return MentionParser()

    @staticmethod
    def _format(document: data.Document) -> str:
        return "\n".join(f"{i}: {' '.join(t.text for t in s)}" for i, s in enumerate(document.sentences))
