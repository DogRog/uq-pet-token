import pathlib
import typing

import langextract as lx
import openai
from langextract import resolver
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam,
)

from autoannotate import base, data, definitions, prompts


class MentionParser(base.BaseParser):
    """Maps the LLM's ``text<TAB>type<TAB>sentence`` lines back to tokens.

    The text of a line does not say which occurrence in the sentence is meant, so the lines
    of each sentence are placed with LangExtract's aligner: in the order the LLM produced
    them (repeated texts resolve to successive occurrences), with at most one occurrence
    per line, and with a fuzzy fallback for texts the LLM did not copy exactly.
    """

    @staticmethod
    def parse_line(line: str, num_sentences: int) -> typing.Tuple[str, str, int]:
        fields = [f.strip() for f in line.split("\t") if f.strip() != ""]
        if len(fields) not in (3, 4):
            raise ValueError(f"Expected 3 or 4 tab-separated fields, got {len(fields)}.")
        # a 4th field is an explanation, which is ignored
        mention_text, mention_type, sentence_id = fields[:3]

        try:
            sentence_id = int(sentence_id)
        except ValueError:
            raise ValueError(f"Invalid sentence index '{sentence_id}'.")

        if not 0 <= sentence_id < num_sentences:
            raise ValueError(f"Sentence index {sentence_id} out of range.")

        return mention_text, mention_type.lower(), sentence_id

    def align_sentence(
            self, sentence: typing.List[data.Token], lines: typing.List[typing.Tuple[str, str]]
    ) -> typing.List[typing.Optional[data.Mention]]:
        """Places the (text, type) lines predicted for one sentence on its tokens.

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
            if extraction.char_interval is None:
                return ()
            start, end = extraction.char_interval.start_pos, extraction.char_interval.end_pos
            return tuple(idx for s, e, idx in token_spans if s < end and e > start)

        # exact matches never overlap each other, so they claim their tokens first
        taken_token_indices: typing.Set[int] = set()
        for extraction in aligned:
            if extraction.alignment_status == lx.data.AlignmentStatus.MATCH_EXACT:
                taken_token_indices.update(token_indices(extraction))

        res: typing.List[typing.Optional[data.Mention]] = []
        for extraction in aligned:
            indices = token_indices(extraction)
            if len(indices) == 0:
                self.report("dropped_not_found", f"Did not find predicted mention '{extraction.extraction_text}'.")
                res.append(None)
                continue
            if extraction.alignment_status != lx.data.AlignmentStatus.MATCH_EXACT:
                if any(idx in taken_token_indices for idx in indices):
                    # the fuzzy fallback may land on an occurrence another line already
                    # claimed, e.g. if the LLM listed a text more often than it occurs
                    self.report(
                        "dropped_no_free_occurrence",
                        f"Did not find a free occurrence of predicted mention '{extraction.extraction_text}'."
                    )
                    res.append(None)
                    continue
                matched_text = " ".join(t.text for t in sentence if t.index_in_document in indices)
                if matched_text.lower() == extraction.extraction_text.lower():
                    # the aligner also falls back to fuzzy matching for lines out of text order
                    self.report(
                        "placed_out_of_order",
                        f"Predicted mention '{extraction.extraction_text}' is out of text order, placed it anyway."
                    )
                else:
                    self.report(
                        "placed_fuzzy",
                        f"Fuzzy matched predicted mention '{extraction.extraction_text}' to '{matched_text}'."
                    )
                taken_token_indices.update(indices)

            res.append(data.Mention(token_document_indices=indices, type=extraction.extraction_class))
        return res

    def parse(self, document: data.Document, string: str) -> data.Document:
        sentences = document.sentences
        # (text, type, sentence id) per line, in the order the LLM produced them
        lines: typing.List[typing.Tuple[str, str, int]] = []
        for line in string.splitlines(keepends=False):
            line = line.strip()
            if line == "":
                continue
            if "\t" not in line:
                self.report("ignored_line", f"Skipping non-tab-separated line '{line}'.")
                continue

            try:
                lines.append(self.parse_line(line, len(sentences)))
            except ValueError as e:
                self.report("dropped_malformed", f"Skipping line '{line}': {e}")

        # aligned sentence by sentence, but kept in the order of the LLM output
        mentions: typing.List[typing.Optional[data.Mention]] = [None] * len(lines)
        for sentence_id in dict.fromkeys(s for _, _, s in lines):
            positions = [i for i, (_, _, s) in enumerate(lines) if s == sentence_id]
            aligned = self.align_sentence(sentences[sentence_id], [lines[i][:2] for i in positions])
            for i, mention in zip(positions, aligned):
                mentions[i] = mention

        doc = document.copy(clear=["mentions", "relations", "entities"])
        doc.mentions = [m for m in mentions if m is not None]
        return doc


class LLMMentionAnnotator(base.BaseAnnotator):
    def __init__(self, client: openai.OpenAI, language_model: str,
                 mention_definitions: typing.List[definitions.MentionDefinition],
                 example: typing.Optional[str],
                 reasoning_effort: base.ReasoningEffort,
                 instruction_role: typing.Literal["developer", "system"] = "developer",
                 extra_body: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__(client, language_model, reasoning_effort, instruction_role, extra_body)
        prompt_folder = pathlib.Path(__file__).parent.resolve() / "resources" / "prompts"
        self._prompt_template = prompts.Prompt(prompt_folder / "annotate-mentions.txt")
        self._defs = mention_definitions
        self._example = example

    def get_instruction(self) -> str:
        """The instruction message, the same for every document."""
        example = self._example if self._example is not None else ""
        type_definitions = "\n".join(f" - {f.for_llm()}" for f in self._defs)
        return self._prompt_template(type_definitions=type_definitions, example=example)

    def get_messages(
            self,
            document: data.Document
    ) -> typing.List[typing.Union[ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam]]:
        text = self._format(document)
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
        return MentionParser()

    @staticmethod
    def _format(document: data.Document) -> str:
        return "\n".join(f"{i}: {' '.join(t.text for t in s)}" for i, s in enumerate(document.sentences))
