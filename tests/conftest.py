"""Shared fixtures. Every test here runs offline: no network, no API key, no weights."""

import os

import pytest
from typeguard import install_import_hook

from uq_pet.config import LLMConfig

# Runtime type-checking of the package under test. Every module boundary in uq_pet was
# redrawn at once, so signature drift is the most likely kind of bug.
install_import_hook("uq_pet")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def make_choice(tokens_and_logprobs: list[tuple[str, float]], finish_reason: str = "stop") -> dict:
    """A cache-record 'choice' built from (token, logprob) pairs."""
    return {
        "text": "[" + ", ".join(t for t, _ in tokens_and_logprobs) + "]",
        "finish_reason": finish_reason,
        "logprobs": [{"token": t, "logprob": lp, "top": {}} for t, lp in tokens_and_logprobs],
    }


def make_text_choice(text: str, finish_reason: str = "stop") -> dict:
    """A choice carrying only the response text — no logprobs at all.

    What the output-only metrics see in a cache produced by a gateway that returns no
    logprobs, and what proves they never reach for any.
    """
    return {"text": text, "finish_reason": finish_reason}


@pytest.fixture
def tag_records():
    """Factory: {idx: [tag array per sample]} -> cache records with text-only choices.

    A sample given as a string is passed through verbatim, so a test can hand the parser
    something unparsable.
    """

    def build(samples_by_idx: dict[int, list[list[int] | str]]) -> list[dict]:
        return [
            {
                "idx": idx,
                "key": f"doc::{idx}",
                "choices": [
                    make_text_choice(s if isinstance(s, str) else str(s)) for s in samples
                ],
            }
            for idx, samples in samples_by_idx.items()
        ]

    return build


@pytest.fixture
def sample_examples() -> list[dict]:
    """Four PET-shaped sentences; two documents share sentence-ID 3."""
    return [
        {
            "document name": "doc-1",
            "sentence-ID": 3,
            "tokens": ["The", "clerk", "signs"],
            "ner-tags": [1, 2, 3],
        },
        {
            "document name": "doc-2",
            "sentence-ID": 3,
            "tokens": ["He", "sends", "it"],
            "ner-tags": [1, 3, 5],
        },
        {
            "document name": "doc-2",
            "sentence-ID": 4,
            "tokens": ["If", "valid"],
            "ner-tags": [9, 11],
        },
        {
            "document name": "doc-3",
            "sentence-ID": 0,
            "tokens": ["Done"],
            "ner-tags": [0],
        },
    ]


TEST_LLM_CONFIG = LLMConfig(model="test-model", n_samples=2)


@pytest.fixture
def llm_config() -> LLMConfig:
    """The config that `sample_records` were notionally produced by."""
    return TEST_LLM_CONFIG


@pytest.fixture
def sample_records(sample_examples) -> list[dict]:
    """Cache records for sample_examples: two healthy, one error.

    `model`/`params` match TEST_LLM_CONFIG so load_cache accepts them, and token
    counts match the corresponding example so alignment checks pass.
    """
    header = {
        "split": "pool",
        "model": TEST_LLM_CONFIG.model,
        "params": TEST_LLM_CONFIG.sampling_params(),
        "prompt_sha": "sha",
    }
    return [
        {
            "idx": 0,
            "key": "doc-1::3",
            **header,
            "choices": [
                make_choice([("[", -0.1), ("1", -1.0), (",", -0.1), ("2", -3.0), ("3", -2.0)]),
                make_choice([("1", -2.0), ("2", -2.0), ("3", -2.0)]),
            ],
        },
        {
            "idx": 1,
            "key": "doc-2::3",
            **header,
            "choices": [
                {"text": "[1, 3, 5]", "finish_reason": "stop", "logprobs": []},
            ],
        },
        # Error records are never written to a real cache; kept here to prove the
        # readers tolerate one if a hand-edited file contains it.
        {"idx": 2, "key": "doc-2::4", "error": "RateLimitError()"},
    ]


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache.jsonl"
