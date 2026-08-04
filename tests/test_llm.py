"""llm.py tests — no client is ever constructed, no key is ever read."""

import json
from types import SimpleNamespace

import pytest

from uq_pet import llm
from uq_pet.config import LLMConfig
from uq_pet.llm import (
    failed_indices,
    load_cache,
    pack_choice,
    score_split,
    truncated_count,
    verify_cache_alignment,
)


def write_jsonl(path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# --- load_cache ---------------------------------------------------------------


def test_load_cache_missing_file_is_empty(tmp_path):
    assert load_cache(tmp_path / "nope.jsonl") == {}


def test_load_cache_keys_by_idx(cache_file, sample_records):
    write_jsonl(cache_file, sample_records)
    assert sorted(load_cache(cache_file)) == [0, 1, 2]


def test_load_cache_skips_truncated_tail(cache_file, sample_records):
    write_jsonl(cache_file, sample_records[:2])
    with cache_file.open("a") as f:
        f.write('{"idx": 3, "choi')
    assert sorted(load_cache(cache_file)) == [0, 1]


def test_load_cache_last_line_wins(cache_file):
    write_jsonl(cache_file, [{"idx": 0, "v": "old"}, {"idx": 0, "v": "new"}])
    assert load_cache(cache_file)[0]["v"] == "new"


def test_load_cache_drops_mismatched_model(cache_file, sample_records):
    write_jsonl(cache_file, sample_records)
    # Only the error record survives — it carries no model to disagree with.
    assert sorted(load_cache(cache_file, model="other-model")) == [2]


def test_load_cache_drops_mismatched_params(cache_file, sample_records):
    write_jsonl(cache_file, sample_records)
    assert sorted(load_cache(cache_file, params={"temperature": 0.0})) == [2]


def test_load_cache_keeps_matching_model_and_params(cache_file, sample_records, llm_config):
    write_jsonl(cache_file, sample_records)
    kept = load_cache(cache_file, model=llm_config.model, params=llm_config.sampling_params())
    assert sorted(kept) == [0, 1, 2]


def test_load_cache_grandfathers_legacy_records(cache_file):
    """The 328 existing records predate key/prompt_sha and must still load."""
    write_jsonl(cache_file, [{"idx": 0, "model": "m", "params": {"n": 5}, "choices": []}])
    kept = load_cache(cache_file, model="m", params={"n": 5}, prompt_sha="abc", keys=["doc::1"])
    assert list(kept) == [0]


def test_load_cache_drops_key_idx_contradiction(cache_file, sample_records):
    write_jsonl(cache_file, sample_records)
    kept = load_cache(cache_file, keys=["WRONG", "doc-2::3", "doc-2::4"])
    assert sorted(kept) == [1, 2]


def test_load_cache_drops_mismatched_prompt_sha(cache_file):
    write_jsonl(cache_file, [{"idx": 0, "prompt_sha": "aaa"}, {"idx": 1, "prompt_sha": "bbb"}])
    assert sorted(load_cache(cache_file, prompt_sha="bbb")) == [1]


# --- pack_choice --------------------------------------------------------------


def fake_choice(with_logprobs: bool):
    token = SimpleNamespace(token="1", logprob=-0.5, top_logprobs=[])
    return SimpleNamespace(
        message=SimpleNamespace(content="[1]"),
        finish_reason="stop",
        logprobs=SimpleNamespace(content=[token]) if with_logprobs else None,
    )


def test_pack_choice_extracts_text_and_logprobs():
    packed = pack_choice(fake_choice(with_logprobs=True))
    assert packed["text"] == "[1]"
    assert packed["finish_reason"] == "stop"
    assert packed["logprobs"] == [{"token": "1", "logprob": -0.5, "top": {}}]


def test_pack_choice_without_logprobs_omits_the_field():
    assert "logprobs" not in pack_choice(fake_choice(with_logprobs=False))


def harmony_choice(*tokens: str, text: str | None, finish_reason: str = "stop"):
    """A choice whose logprobs carry a harmony token stream, one token per string."""
    return SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
        logprobs=SimpleNamespace(
            content=[SimpleNamespace(token=t, logprob=-0.5, top_logprobs=[]) for t in tokens]
        ),
    )


def packed_tokens(choice) -> list[str]:
    return [t["token"] for t in pack_choice(choice).get("logprobs", [])]


def test_pack_choice_keeps_only_the_final_channel():
    choice = harmony_choice(
        "<|channel|>",
        "analysis",
        "<|message|>",
        "So",
        " 2",
        " it",
        " is",
        ".",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "[1",
        ",",
        " 0",
        "]",
        "<|return|>",
        text="[1, 0]",
    )
    assert packed_tokens(choice) == ["[1", ",", " 0", "]"]


def test_pack_choice_drops_a_sample_truncated_inside_its_reasoning():
    # The failure mode that emptied a whole run: reasoning digits are not tags.
    choice = harmony_choice(
        "<|channel|>",
        "analysis",
        "<|message|>",
        "token",
        " 2",
        " is",
        " I-Actor",
        text=None,
        finish_reason="length",
    )
    assert packed_tokens(choice) == []


def test_pack_choice_leaves_a_stream_without_channels_untouched():
    choice = harmony_choice("[1", ",", " 0", "]", text="[1, 0]")
    assert packed_tokens(choice) == ["[1", ",", " 0", "]"]


# --- score_split --------------------------------------------------------------


KEYS = ["doc-1::3", "doc-2::3", "doc-2::4"]


def test_score_split_never_builds_a_client_when_cache_is_complete(
    monkeypatch, cache_file, sample_records, llm_config
):
    """The guarantee that re-runs and CI need no NHR_FAU_API_KEY."""
    monkeypatch.setattr(
        llm, "make_client", lambda cfg: pytest.fail("client built despite a complete cache")
    )
    write_jsonl(cache_file, sample_records)

    records = score_split(
        llm_config, "system", ["a", "b", "c"], KEYS, cache_file, "sha", progress=False
    )
    assert [r["idx"] for r in records] == [0, 1, 2]


def test_score_split_respects_limit(monkeypatch, cache_file, sample_records, llm_config):
    monkeypatch.setattr(llm, "make_client", lambda cfg: pytest.fail("should not be called"))
    write_jsonl(cache_file, sample_records)

    records = score_split(
        llm_config, "system", ["a", "b", "c"], KEYS, cache_file, "sha", limit=2, progress=False
    )
    assert [r["idx"] for r in records] == [0, 1]


def test_score_split_scores_only_the_uncached(monkeypatch, cache_file, sample_records, llm_config):
    """A partial cache means exactly the missing indices are requested.

    The new records are appended, so the next run finds a complete cache.
    """
    write_jsonl(cache_file, sample_records[:1])
    requested = []

    def fake_score_one(client, cfg, system_prompt, user_prompt, idx, key, prompt_sha, split):
        requested.append(idx)
        return {
            "idx": idx,
            "key": key,
            "split": split,
            "model": cfg.model,
            "params": cfg.sampling_params(),
            "prompt_sha": prompt_sha,
            "choices": [{"text": "[0]", "finish_reason": "stop", "logprobs": []}],
        }

    monkeypatch.setattr(llm, "make_client", lambda cfg: object())
    monkeypatch.setattr(llm, "score_one", fake_score_one)

    records = score_split(
        llm_config, "system", ["a", "b", "c"], KEYS, cache_file, "sha", progress=False
    )
    assert sorted(requested) == [1, 2]
    assert [r["idx"] for r in records] == [0, 1, 2]
    assert sorted(load_cache(cache_file)) == [0, 1, 2]


def test_score_split_does_not_cache_failures(monkeypatch, cache_file, llm_config):
    """Failed sentences must stay out of the cache so the next run retries them."""
    monkeypatch.setattr(llm, "make_client", lambda cfg: object())
    monkeypatch.setattr(
        llm,
        "score_one",
        lambda client, cfg, sp, up, idx, key, sha, split: {
            "idx": idx,
            "key": key,
            "error": "boom",
        },
    )

    records = score_split(
        llm_config, "system", ["a"], ["doc-1::3"], cache_file, "sha", progress=False
    )
    assert failed_indices(records) == [0]
    assert load_cache(cache_file) == {}


def test_score_split_rejects_mismatched_prompts_and_keys(cache_file):
    with pytest.raises(ValueError):
        score_split(LLMConfig(), "system", ["a", "b"], ["only-one"], cache_file, "sha")


# --- diagnostics --------------------------------------------------------------


def test_failed_indices(sample_records):
    assert failed_indices(sample_records) == [2]


def test_truncated_count():
    records = [
        {"idx": 0, "choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]},
        {"idx": 1, "error": "boom"},
    ]
    assert truncated_count(records) == 1


def test_verify_cache_alignment_perfect(sample_records, sample_examples):
    assert verify_cache_alignment(sample_records, sample_examples) == 1.0


def test_verify_cache_alignment_detects_a_shift(sample_records, sample_examples):
    shifted = [{**rec, "idx": rec["idx"] + 1} for rec in sample_records]
    # doc-1's 3 tags land on doc-2::4 (2 tokens); doc-2's 3 tags land on doc-3::0 (1 token)
    assert verify_cache_alignment(shifted, sample_examples) == 0.0


def test_verify_cache_alignment_empty_is_zero():
    assert verify_cache_alignment([], [{"tokens": ["a"]}]) == 0.0
