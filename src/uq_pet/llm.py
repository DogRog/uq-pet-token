"""Repeated LLM sampling over a split, cached to resumable JSONL.

The cache is append-only and line-buffered, so an interrupted run loses at most the
record in flight; a re-run picks up from the records already on disk and, if the cache
is complete, never constructs a client or touches the network.

Records are keyed by `idx` = position in the split. Records written by this module also
carry `key` (a stable sentence identifier) and `prompt_sha`, which lets `load_cache`
detect a cache that no longer matches the data or the prompt in hand.
"""

import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from uq_pet.config import PROJECT_ROOT, LLMConfig
from uq_pet.prompt import parse_tag_ids

logger = logging.getLogger(__name__)

RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def make_client(cfg: LLMConfig) -> OpenAI:
    """Build the gateway client, resolving the API key named by cfg.api_key_env."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env} is not set — put it in {PROJECT_ROOT / '.env'} or the "
            f"environment. (Scoring is only needed when the cache is incomplete.)"
        )
    return OpenAI(
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=0,  # retries are handled in score_one, with jittered backoff
    )


CHANNEL_TOKEN = "<|channel|>"
MESSAGE_TOKEN = "<|message|>"
ANSWER_CHANNEL = "final"
SPECIAL_TOKEN = re.compile(r"^<\|.*\|>$")


def answer_tokens(content: list[Any]) -> list[Any]:
    """The tokens of the answer itself, dropping a reasoning model's scratchpad.

    A harmony-format model (gpt-oss) returns its chain of thought in the same token
    stream as the answer — `<|channel|>analysis<|message|>` ... `<|channel|>final
    <|message|>[1, 2, 0]<|return|>` — while `message.content` holds the final channel
    alone. Every metric in `uncertainty` assumes the token stream *is* the answer;
    `is_tag_token` counts the digits in "so 'MSP' is I-Actor (2)" exactly like a real
    tag, so keeping the reasoning would score sentences by their scratchpad.

    Only the last channel counts, and only if it is the answer channel: a sample cut
    off mid-reasoning never opened one, and returns no tokens rather than its
    reasoning. That is the honest answer — `sentence_scores` then omits it, which is
    what a sample carrying no answer deserves.

    A stream with no channel markers is passed through untouched, so this changes
    nothing for the non-reasoning models.
    """
    channels = [i for i, t in enumerate(content) if t.token == CHANNEL_TOKEN]
    if not channels:
        return list(content)

    start = channels[-1]
    body = next((i for i, t in enumerate(content[start:], start) if t.token == MESSAGE_TOKEN), None)
    if body is None:  # truncated before the channel's message began
        return []
    name = "".join(t.token for t in content[start + 1 : body]).strip()
    if name != ANSWER_CHANNEL:
        return []
    return [t for t in content[body + 1 :] if not SPECIAL_TOKEN.match(t.token)]


def pack_choice(choice: Any) -> dict:
    """Flatten one API choice into a JSON-serializable record fragment."""
    packed: dict[str, Any] = {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
    }
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    if content:
        packed["logprobs"] = [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top": {tp.token: tp.logprob for tp in (t.top_logprobs or [])},
            }
            for t in answer_tokens(content)
        ]
    return packed


def load_cache(
    path: Path,
    *,
    model: str | None = None,
    params: dict | None = None,
    prompt_sha: str | None = None,
    keys: list[str] | None = None,
) -> dict[int, dict]:
    """Read the JSONL cache into {idx: record}, dropping records we cannot trust.

    A record is dropped when its `model`/`params`/`prompt_sha` disagree with the
    arguments, or when its `key` contradicts `keys[idx]`. Records predating those
    fields are grandfathered — they can only be checked by verify_cache_alignment.
    """
    if not path.exists():
        return {}

    records: dict[int, dict] = {}
    dropped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1  # truncated tail from a killed run
            continue

        idx = rec.get("idx")
        if not isinstance(idx, int):
            dropped += 1
            continue
        if model is not None and rec.get("model", model) != model:
            dropped += 1
            continue
        if params is not None and rec.get("params", params) != params:
            dropped += 1
            continue
        if prompt_sha is not None and rec.get("prompt_sha", prompt_sha) != prompt_sha:
            dropped += 1
            continue
        if keys is not None and "key" in rec and 0 <= idx < len(keys) and rec["key"] != keys[idx]:
            dropped += 1
            continue
        records[idx] = rec  # later lines win: a retry supersedes an earlier attempt

    if dropped:
        logger.warning("Dropped %d unusable record(s) from %s", dropped, path.name)
    return records


def score_one(
    client: OpenAI,
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    idx: int,
    key: str,
    prompt_sha: str,
    split: str,
) -> dict:
    """Sample the model n_samples times for one sentence. Pure: writes nothing."""
    params = cfg.sampling_params()
    response = None
    for attempt in range(cfg.max_retries):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **params,
            )
            break
        except RETRYABLE as e:
            if attempt == cfg.max_retries - 1:
                return {"idx": idx, "key": key, "error": repr(e)}
            time.sleep(min(2**attempt, 30) * (0.5 + random.random()))
        except openai.APIStatusError as e:
            return {"idx": idx, "key": key, "error": repr(e)}  # 400/401/404: fatal
    else:
        return {"idx": idx, "key": key, "error": "exhausted retries"}

    return {
        "idx": idx,
        "key": key,
        "split": split,
        "model": response.model,
        "params": params,
        "prompt_sha": prompt_sha,
        "choices": [pack_choice(c) for c in response.choices],
    }


def score_split(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompts: list[str],
    keys: list[str],
    cache_path: Path,
    prompt_sha: str,
    *,
    split: str = "pool",
    limit: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """Score every uncached sentence in the split, appending to cache_path.

    Returns all usable records (cached + new), sorted by idx. Constructs the client
    only if there is work to do, so a complete cache needs no API key.
    """
    if len(user_prompts) != len(keys):
        raise ValueError(f"{len(user_prompts)} prompts but {len(keys)} keys")

    n = len(user_prompts) if limit is None else min(limit, len(user_prompts))
    cached = load_cache(
        cache_path,
        model=cfg.model,
        params=cfg.sampling_params(),
        prompt_sha=prompt_sha,
        keys=keys,
    )
    todo = [i for i in range(n) if i not in cached]
    records = [rec for idx, rec in cached.items() if idx < n]

    if not todo:
        logger.info("Cache complete for %s (%d records) — no API calls", split, len(records))
        return sorted(records, key=lambda r: r["idx"])

    logger.info("Scoring %d/%d %s sentences (%d cached)", len(todo), n, split, len(cached))
    client = make_client(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    # Line-buffered append is what makes the cache crash-resumable: each completed
    # record hits disk immediately, so a kill loses only work that was in flight.
    with (
        cache_path.open("a", buffering=1) as out,
        ThreadPoolExecutor(max_workers=cfg.workers) as pool,
    ):
        futures = [
            pool.submit(
                score_one,
                client,
                cfg,
                system_prompt,
                user_prompts[i],
                i,
                keys[i],
                prompt_sha,
                split,
            )
            for i in todo
        ]
        for fut in tqdm(as_completed(futures), total=len(todo), disable=not progress):
            rec = fut.result()
            records.append(rec)
            if "error" in rec:
                continue  # deliberately not cached, so the next run retries it
            with lock:
                out.write(json.dumps(rec) + "\n")

    return sorted(records, key=lambda r: r["idx"])


def failed_indices(records: Iterable[dict]) -> list[int]:
    return sorted(rec["idx"] for rec in records if "error" in rec)


def truncated_count(records: Iterable[dict]) -> int:
    """Choices that hit the max_tokens ceiling — their logprobs cover a cut-off array."""
    return sum(
        1
        for rec in records
        if "error" not in rec
        for choice in rec.get("choices", [])
        if choice.get("finish_reason") == "length"
    )


def verify_cache_alignment(records: Iterable[dict], examples: list[dict]) -> float:
    """Fraction of records whose response length matches the example they claim to be.

    The cache is keyed by position, so a reordered split would silently pair every
    sentence with another sentence's scores. The model returns one tag per token, so
    comparing array lengths detects that: measured ~0.96 when aligned and <0.05 under
    any shift.
    """
    checked = matched = 0
    for rec in records:
        if "error" in rec:
            continue
        idx = rec["idx"]
        if not 0 <= idx < len(examples):
            continue
        choices = rec.get("choices") or []
        if not choices:
            continue
        tags = parse_tag_ids(choices[0].get("text"))
        if tags is None:
            continue
        checked += 1
        matched += len(tags) == len(examples[idx]["tokens"])
    return matched / checked if checked else 0.0
