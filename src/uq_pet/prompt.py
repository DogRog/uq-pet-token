"""Prompt construction, and parsers for the response formats it asks for.

The templates below define both output contracts — the default tag array and the
opt-in JSON object carrying tags plus self-uncertainty — so their parsers live here too
rather than in llm.py. One module owns both halves of each contract.
"""

import ast
import hashlib
import json
import math
import re
from collections.abc import Iterable
from string import Template
from textwrap import dedent

from uq_pet.config import NER_TAGS

TAG_LEGEND = "\n".join(f"  {i} = {tag}" for i, tag in enumerate(NER_TAGS))

SYSTEM_TEMPLATE = Template(
    dedent("""\
    You are a strict Named Entity Recognition (NER) system for Process Extraction.
    Assign exactly one tag ID to each token in the sentence.

    ENTITY DEFINITIONS:
    - Actor: The person, system, or role performing the action.
    - Activity: The task or action being executed.
    - Activity Data: The object, document, or data manipulated by the activity.
    - Further Specification: Additional context, tools, or locations (e.g., 'via email').
    - XOR Gateway: Words indicating an exclusive branching point (e.g., 'If', 'otherwise').
    - Condition Specification: The condition required to take a branch (e.g., 'the claim is valid').
    - AND Gateway: Words indicating parallel execution (e.g., 'in parallel').
    - O: Tokens outside of any process entity.

    DATASET RULES:
    - Determiners ('The', 'a', 'an') MUST be included in the entity if they precede it.
    - Multi-word entities must start with 'B-' (Beginning) and continue with 'I-' (Inside).
    - Single-word entities get the 'B-' tag.

    TAG IDS:
    $tag_legend

    === EXAMPLES ===
    $examples
    === END OF EXAMPLES ===
    """)
)

USER_TEMPLATE = Template(
    dedent("""\
    Tokens to tag:
    $tokens

    Output MUST be an array of integers with exactly $n_tokens elements, one per
    token, in order. Output nothing except the array.
    """)
)

SELF_UNCERTAINTY_SYSTEM_SUFFIX = dedent("""\

    SELF-REPORTED UNCERTAINTY:
    Along with the tag IDs, report how uncertain you are that the complete tag array
    is correct. Use 0 for completely certain and 1 for completely uncertain. Judge
    semantic and boundary ambiguity in the tagging itself, not your confidence in
    following the output format.
    """)

SELF_UNCERTAINTY_USER_TEMPLATE = Template(
    dedent("""\
    Tokens to tag:
    $tokens

    Output MUST be one JSON object in exactly this shape:
    {"tags": [one integer per token], "uncertainty": <number from 0 to 1>}

    The "tags" array MUST have exactly $n_tokens elements, one per token, in order.
    Output no markdown, prose, or additional keys.
    """)
)

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]?")


def build_system_prompt(few_shot: Iterable[dict], *, self_report_uncertainty: bool = False) -> str:
    """Build the constant prompt prefix — identical for every sentence, so it caches."""
    examples = "\n\n".join(f"{ex['tokens']}\n{ex['ner-tags']}" for ex in few_shot)
    prompt = SYSTEM_TEMPLATE.substitute(tag_legend=TAG_LEGEND, examples=examples)
    if self_report_uncertainty:
        prompt += SELF_UNCERTAINTY_SYSTEM_SUFFIX
    return prompt


def build_user_prompt(tokens: list[str], *, self_report_uncertainty: bool = False) -> str:
    """Build the user prompt for one sentence of PET tokens."""
    template = SELF_UNCERTAINTY_USER_TEMPLATE if self_report_uncertainty else USER_TEMPLATE
    return template.substitute(tokens=tokens, n_tokens=len(tokens))


def build_user_prompts(
    examples: Iterable[dict], *, self_report_uncertainty: bool = False
) -> list[str]:
    """One user prompt per example, in order — index i addresses example i."""
    return [
        build_user_prompt(ex["tokens"], self_report_uncertainty=self_report_uncertainty)
        for ex in examples
    ]


def prompt_fingerprint(system_prompt: str) -> str:
    """Short digest of the rendered prompt, stamped onto every new cache record.

    Editing SYSTEM_TEMPLATE changes this, which lets load_cache tell records produced
    under a different prompt from ones it can still trust.
    """
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:12]


def _json_objects(text: str) -> Iterable[dict]:
    """JSON objects embedded in a response, tolerating fences or surrounding prose."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_tag_ids(text: str | None) -> list[int] | None:
    """Parse the model's tag-ID array; None when nothing usable is present.

    Tolerates fenced code blocks, prose around the array, and a truncated tail (some
    responses hit the max_tokens ceiling mid-array), since this is used to sanity-check
    cached responses rather than to produce labels.
    """
    if not text:
        return None

    # Self-UQ responses wrap the same tag array in a JSON object. Keep accepting the
    # legacy bare array so old caches and non-self-UQ configs retain their contract.
    for obj in _json_objects(text):
        parsed = obj.get("tags")
        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(t, int) and not isinstance(t, bool) for t in parsed)
        ):
            return list(parsed)

    match = _ARRAY_RE.search(text)
    if match is None:
        return None

    body = match.group(0)
    try:
        parsed = ast.literal_eval(body if body.endswith("]") else body + "]")
    except (ValueError, SyntaxError):
        # A truncated tail can leave a trailing comma or a half-written number.
        parsed = [int(t) for t in re.findall(r"-?\d+", body)]

    if not isinstance(parsed, (list, tuple)):
        return None
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in parsed):
        return None
    return list(parsed) or None


def parse_self_uncertainty(text: str | None) -> float | None:
    """Parse a self-reported sequence uncertainty in [0, 1].

    Only the explicit JSON field is accepted. Missing, non-numeric, non-finite, and
    out-of-range values are unusable rather than silently clamped into the ranking.
    """
    if not text:
        return None
    for obj in _json_objects(text):
        score = obj.get("uncertainty")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        value = float(score)
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            return value
    return None
