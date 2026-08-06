import pytest

from uq_pet.config import NER_TAGS
from uq_pet.prompt import (
    build_system_prompt,
    build_user_prompt,
    build_user_prompts,
    parse_self_uncertainty,
    parse_tag_ids,
    prompt_fingerprint,
)


def test_system_prompt_has_no_unfilled_placeholders(sample_examples):
    assert "$" not in build_system_prompt(sample_examples)


def test_system_prompt_lists_every_tag(sample_examples):
    prompt = build_system_prompt(sample_examples)
    for i, tag in enumerate(NER_TAGS):
        assert f"{i} = {tag}" in prompt


def test_system_prompt_embeds_the_few_shot_examples(sample_examples):
    prompt = build_system_prompt(sample_examples)
    assert "'clerk'" in prompt
    assert "[1, 2, 3]" in prompt


def test_user_prompt_declares_the_token_count():
    assert "exactly 3 elements" in build_user_prompt(["a", "b", "c"])


def test_build_user_prompts_preserves_order(sample_examples):
    prompts = build_user_prompts(sample_examples)
    assert len(prompts) == len(sample_examples)
    assert "exactly 1 elements" in prompts[3]  # the single-token example


def test_self_uncertainty_prompt_requests_the_json_contract(sample_examples):
    system = build_system_prompt(sample_examples, self_report_uncertainty=True)
    user = build_user_prompt(["a", "b"], self_report_uncertainty=True)

    assert "0 for completely certain and 1 for completely uncertain" in system
    assert '{"tags": [one integer per token], "uncertainty":' in user
    assert "exactly 2 elements" in user


def test_self_uncertainty_mode_changes_the_prompt_fingerprint(sample_examples):
    regular = build_system_prompt(sample_examples)
    self_uq = build_system_prompt(sample_examples, self_report_uncertainty=True)
    assert prompt_fingerprint(regular) != prompt_fingerprint(self_uq)


def test_prompt_fingerprint_is_stable_and_sensitive(sample_examples):
    prompt = build_system_prompt(sample_examples)
    assert prompt_fingerprint(prompt) == prompt_fingerprint(prompt)
    assert prompt_fingerprint(prompt) != prompt_fingerprint(prompt + " ")
    assert len(prompt_fingerprint(prompt)) == 12


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[0, 1, 2]", [0, 1, 2]),
        ("[0,1,2]", [0, 1, 2]),
        ("```json\n[3, 4]\n```", [3, 4]),
        ("Here are the tags:\n[5, 6]\nHope that helps.", [5, 6]),
        ("[0, 1,", [0, 1]),  # truncated by max_tokens
        ("[7", [7]),
        ("[]", None),
        ("no array here", None),
        ("", None),
        (None, None),
        ("[0, 'B-Actor']", None),  # wrong element type
    ],
)
def test_parse_tag_ids(text, expected):
    assert parse_tag_ids(text) == expected


def test_parse_tag_ids_on_a_real_cached_response():
    assert parse_tag_ids("[0, 0, 5, 6, 0, 3, 0]") == [0, 0, 5, 6, 0, 3, 0]


def test_parse_tag_ids_from_self_uncertainty_response():
    text = '{"tags": [0, 3, 4], "uncertainty": 0.72}'
    assert parse_tag_ids(text) == [0, 3, 4]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"tags": [0], "uncertainty": 0}', 0.0),
        ('```json\n{"tags": [0], "uncertainty": 0.72}\n```', 0.72),
        ('{"tags": [0], "uncertainty": 1}', 1.0),
        ('{"tags": [0], "uncertainty": -0.1}', None),
        ('{"tags": [0], "uncertainty": 1.1}', None),
        ('{"tags": [0], "uncertainty": "0.5"}', None),
        ('{"tags": [0], "uncertainty": true}', None),
        ('{"tags": [0]}', None),
        ("[0]", None),
        (None, None),
    ],
)
def test_parse_self_uncertainty(text, expected):
    assert parse_self_uncertainty(text) == expected
