import asyncio
import json

import pytest

import uq_pet.config
import uq_pet.llm_scoring
from uq_pet.config import LLMScoreConfig
from uq_pet.llm_scoring import (
    build_ner_prompt,
    derive_sample_seed,
    load_cache,
    parse_ner_output,
    prompt_fingerprint,
    score_pool,
)
from uq_pet.uncertainty import WhiteboxDataMissingError, compute_metric

TOKENS = ["The", "clerk", "checks", "the", "form"]


def _output(tags: list[str]) -> str:
    return json.dumps([{"token": t, "tag": tag} for t, tag in zip(TOKENS, tags, strict=True)])


def test_parse_ner_output_valid_json():
    tags = ["B-Actor", "I-Actor", "B-Activity", "B-Activity Data", "I-Activity Data"]
    assert parse_ner_output(_output(tags), TOKENS) == tags


def test_parse_ner_output_strips_code_fence():
    tags = ["O", "O", "B-Activity", "O", "O"]
    fenced = f"```json\n{_output(tags)}\n```"
    assert parse_ner_output(fenced, TOKENS) == tags


def test_parse_ner_output_pads_short_with_o():
    short = json.dumps([{"token": "The", "tag": "B-Actor"}])
    assert parse_ner_output(short, TOKENS) == ["B-Actor", "O", "O", "O", "O"]


def test_parse_ner_output_truncates_long():
    long = json.dumps([{"token": "x", "tag": "O"}] * 10)
    assert len(parse_ner_output(long, TOKENS)) == len(TOKENS)


def test_parse_ner_output_invalid_tag_maps_to_o():
    tags = ["B-Actor", "NOT-A-TAG", "O", "O", "O"]
    assert parse_ner_output(_output(tags), TOKENS) == ["B-Actor", "O", "O", "O", "O"]


def test_parse_ner_output_garbage_returns_all_o():
    assert parse_ner_output("sorry, I cannot help", TOKENS) == ["O"] * len(TOKENS)


def test_build_ner_prompt_contains_tokens_and_example():
    example_tokens = ["Alice", "approves"]
    example_tags = ["B-Actor", "B-Activity"]
    prompt = build_ner_prompt(TOKENS, example_tokens, example_tags)
    assert str(TOKENS) in prompt
    assert str(example_tokens) in prompt
    assert "B-Activity" in prompt


def test_prompt_fingerprint_stable_and_sensitive():
    fp = prompt_fingerprint(["Alice"], ["B-Actor"])
    assert fp == prompt_fingerprint(["Alice"], ["B-Actor"])
    assert fp != prompt_fingerprint(["Bob"], ["B-Actor"])


def test_default_prompt_renders_to_historical_fingerprint():
    # Golden value from before the template moved to prompts/ner_v1.txt.
    # If this changes, existing score caches stop validating against their
    # headers — edit a *new* prompt file (llm.prompt: ner_v2) instead.
    assert prompt_fingerprint(["Alice"], ["B-Actor"]) == "a92de2863f0cbb3c"


def test_unknown_prompt_raises_with_available_list():
    with pytest.raises(FileNotFoundError, match="ner_v1"):
        build_ner_prompt(TOKENS, ["Alice"], ["B-Actor"], prompt="does_not_exist")


def test_derive_sample_seed_stable_and_distinct():
    seed = derive_sample_seed(3407, "doc-0", 0)
    assert seed == derive_sample_seed(3407, "doc-0", 0)
    assert seed != derive_sample_seed(3407, "doc-0", 1)
    assert seed != derive_sample_seed(3407, "doc-1", 0)
    assert 0 <= seed < 2**31


def test_old_cache_without_entropies_still_loads(tmp_path):
    # Pre-mlx cache format: no `backend` in the header, no `token_entropies`
    # in records. Must keep loading, and black-box metrics must keep working.
    header = {"model": "meta-llama/llama-3-8b-instruct", "num_samples": 2}
    record = {
        "key": "doc-0",
        "tokens": TOKENS,
        "gt_tags": ["O"] * len(TOKENS),
        "raw_responses": ["", ""],
        "parsed_samples": [["O"] * len(TOKENS), ["B-Actor"] + ["O"] * 4],
    }
    path = tmp_path / "old_cache.jsonl"
    path.write_text(json.dumps({"header": header}) + "\n" + json.dumps(record) + "\n")

    cache = load_cache(path, expected_header=header)
    assert compute_metric("mean_token_entropy", cache["doc-0"]) == pytest.approx(0.2)
    with pytest.raises(WhiteboxDataMissingError):
        compute_metric("predictive_entropy", cache["doc-0"])


def test_score_pool_test_split_writes_own_cache(monkeypatch, tmp_path):
    from datasets import Dataset

    monkeypatch.setattr(uq_pet.config, "LLM_SCORES_DIR", tmp_path)
    monkeypatch.setattr(uq_pet.llm_scoring, "make_client", lambda: None)

    async def fake_sample(client, prompt, cfg, semaphore):
        return _output(["B-Actor", "O", "O", "O", "O"])

    monkeypatch.setattr(uq_pet.llm_scoring, "get_single_sample", fake_sample)

    few_shot = {"document name": "doc-fs", "sentence-ID": 0,
                "tokens": ["Alice", "approves"], "ner-tags": [1, 3]}
    test_split = Dataset.from_list([{
        "document name": "doc-9", "sentence-ID": 1,
        "tokens": TOKENS, "ner-tags": [1, 2, 3, 0, 0],
    }])
    cfg = LLMScoreConfig(num_samples=2)

    cache = asyncio.run(score_pool(
        cfg, test_split, few_shot_example=few_shot, split="test",
    ))

    assert set(cache) == {"doc-9::1"}
    assert cache["doc-9::1"]["parsed_samples"] == [["B-Actor", "O", "O", "O", "O"]] * 2
    assert cache["doc-9::1"]["gt_tags"] == ["B-Actor", "I-Actor", "B-Activity", "O", "O"]

    cache_path = cfg.cache_path("test")
    assert cache_path.name.endswith("_test.jsonl")
    assert not cfg.cache_path().exists()  # the pool cache is untouched
    with open(cache_path) as f:
        header = json.loads(f.readline())["header"]
    assert header["split"] == "test"
    # The prompt (and so its fingerprint) is built from the pool's few-shot
    # example, matching the pool cache.
    assert header["prompt_fingerprint"] == prompt_fingerprint(
        few_shot["tokens"], ["B-Actor", "B-Activity"]
    )
