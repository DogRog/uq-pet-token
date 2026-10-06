import pathlib
import traceback
import typing

import langextract as lx
import openai
from langextract import resolver
from openai.types.chat import ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionContentPartTextParam

from autoannotate import base, definitions, data, prompts


class MentionParser(base.BaseParser):
    """
    Maps the LLM's ``text<TAB>type<TAB>sentence`` lines back to tokens.

    The text of a line does not say which occurrence in the sentence is meant, so the lines
    of each sentence are placed with LangExtract's aligner: in the order the LLM produced
    them (repeated texts resolve to successive occurrences), with at most one occurrence
    per line, and with a fuzzy fallback for texts the LLM did not copy exactly.
    """

    @staticmethod
    def parse_line(line: str, document: data.Document) -> typing.Tuple[str, str, int]:
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

        if not 0 <= sentence_id < len(document.sentences):
            raise ValueError(f"Sentence index {sentence_id} out of range, skipping line.")

        return mention_text.strip(), mention_type.lower().strip(), sentence_id

    @staticmethod
    def align_sentence(
            sentence: typing.List[data.Token], lines: typing.List[typing.Tuple[str, str]]
    ) -> typing.List[typing.Optional[data.Mention]]:
        """
        Places the (text, type) lines predicted for one sentence on its tokens.

        :return: one entry per line, in the same order, None if the line could not be placed
        """
        # character span of every token in the sentence text we align against
        token_spans = []
        pos = 0
        for token in sentence:
            token_spans.append((pos, pos + len(token.text), token.index_in_document))
            pos += len(token.text) + 1
        sentence_text = " ".join(t.text for t in sentence)

        extractions = [
            lx.data.Extraction(extraction_class=mention_type, extraction_text=mention_text)
            for mention_text, mention_type in lines
        ]
        # partial matches would silently cut mentions short, rather fall back to fuzzy matching
        [aligned] = resolver.WordAligner().align_extractions(
            [extractions], sentence_text, accept_match_lesser=False
        )

        def token_indices(extraction: lx.data.Extraction) -> typing.Tuple[int, ...]:
            start, end = extraction.char_interval.start_pos, extraction.char_interval.end_pos
            return tuple(idx for s, e, idx in token_spans if s < end and e > start)

        # exact matches never overlap each other, so they claim their tokens first
        taken_token_indices: typing.Set[int] = set()
        for extraction in aligned:
            if extraction.alignment_status == lx.data.AlignmentStatus.MATCH_EXACT:
                taken_token_indices.update(token_indices(extraction))

        res: typing.List[typing.Optional[data.Mention]] = []
        for extraction in aligned:
            if extraction.char_interval is None:
                print(f"Did not find predicted mention '{extraction.extraction_text}'.")
                res.append(None)
                continue

            indices = token_indices(extraction)
            if len(indices) == 0:
                print(f"Did not find predicted mention '{extraction.extraction_text}'.")
                res.append(None)
                continue
            if extraction.alignment_status != lx.data.AlignmentStatus.MATCH_EXACT:
                if any(idx in taken_token_indices for idx in indices):
                    # the fuzzy fallback may land on an occurrence another line already
                    # claimed, e.g. if the LLM listed a text more often than it occurs
                    print(f"Did not find a free occurrence of predicted mention '{extraction.extraction_text}'.")
                    res.append(None)
                    continue
                matched_text = " ".join(t.text for t in sentence if t.index_in_document in indices)
                if matched_text.lower() == extraction.extraction_text.lower():
                    # the aligner also falls back to fuzzy matching for lines out of text order
                    print(f"Predicted mention '{extraction.extraction_text}' is out of text order, placed it anyway.")
                else:
                    print(f"Fuzzy matched predicted mention '{extraction.extraction_text}' to '{matched_text}'.")
                taken_token_indices.update(indices)

            res.append(data.Mention(token_document_indices=indices, type=extraction.extraction_class))
        return res

    def parse(self, document: data.Document, string: str) -> data.Document:
        # lines grouped by sentence, in the order the LLM produced them
        lines_by_sentence: typing.Dict[int, typing.List[typing.Tuple[int, str, str]]] = {}
        num_parse_errors = 0
        for line_index, line in enumerate(string.splitlines(keepends=False)):
            line = line.strip()
            if line == "":
                continue
            if "\t" not in line:
                continue

            try:
                mention_text, mention_type, sentence_id = self.parse_line(line, document)
                lines_by_sentence.setdefault(sentence_id, []).append((line_index, mention_text, mention_type))
            except Exception:
                num_parse_errors += 1
                print("Error during parsing of line, skipping line. Error was:")
                print(traceback.format_exc())

        # keep the mentions in the order of the LLM output
        mentions_by_line: typing.Dict[int, data.Mention] = {}
        for sentence_id, lines in lines_by_sentence.items():
            aligned = self.align_sentence(
                document.sentences[sentence_id],
                [(mention_text, mention_type) for _, mention_text, mention_type in lines]
            )
            for (line_index, _, _), mention in zip(lines, aligned):
                if mention is not None:
                    mentions_by_line[line_index] = mention
        parsed_mentions = [mentions_by_line[i] for i in sorted(mentions_by_line)]

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
                 reasoning_effort: base.ReasoningEffort,
                 instruction_role: typing.Literal["developer", "system"] = "developer"):
        super().__init__(client, language_model, reasoning_effort, instruction_role)
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
