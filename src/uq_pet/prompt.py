"""Prompt construction, and the parser for the response format it asks for.

The templates below define the model's output contract ("an array of exactly N
integers"), so `parse_tag_ids` lives here too rather than in llm.py — one module owns
both halves of the contract.
"""

import ast
import hashlib
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

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]?")


def build_system_prompt(few_shot: Iterable[dict]) -> str:
    """Build the constant prompt prefix — identical for every sentence, so it caches."""
    examples = "\n\n".join(f"{ex['tokens']}\n{ex['ner-tags']}" for ex in few_shot)
    return SYSTEM_TEMPLATE.substitute(tag_legend=TAG_LEGEND, examples=examples)


def build_user_prompt(tokens: list[str]) -> str:
    """Build the user prompt for one sentence of PET tokens."""
    return USER_TEMPLATE.substitute(tokens=tokens, n_tokens=len(tokens))


def build_user_prompts(examples: Iterable[dict]) -> list[str]:
    """One user prompt per example, in order — index i addresses example i."""
    return [build_user_prompt(ex["tokens"]) for ex in examples]


def prompt_fingerprint(system_prompt: str) -> str:
    """Short digest of the rendered prompt, stamped onto every new cache record.

    Editing SYSTEM_TEMPLATE changes this, which lets load_cache tell records produced
    under a different prompt from ones it can still trust.
    """
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:12]


def parse_tag_ids(text: str | None) -> list[int] | None:
    """Parse the model's tag-ID array; None when nothing usable is present.

    Tolerates fenced code blocks, prose around the array, and a truncated tail (some
    responses hit the max_tokens ceiling mid-array), since this is used to sanity-check
    cached responses rather than to produce labels.
    """
    if not text:
        return None
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
