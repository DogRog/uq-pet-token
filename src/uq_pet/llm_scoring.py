"""LLM repeated-sampling pass over the experiment pool, with JSONL caching.

Cache format (`data/processed/llm_scores/<model>_k<K>_t<temp>_seed<seed>.jsonl`):
line 0 is a header record describing the config; every other line is one
sentence record with the raw responses and parsed tag sequences. Records from
the "mlx" backend additionally carry `token_entropies` (per-sample lists of
per-token predictive entropies in bits) for white-box metrics, and the header
carries `backend: mlx`. Metrics are never stored — they are recomputed from
the cached fields, so budgets, metrics and repeats can be swept without new
generation. Existing keys are skipped on rerun, so an interrupted pass resumes
where it left off. Scoring the held-out test split (for the LLM-alone baseline
in reports) writes a separate cache with a `_test` filename suffix and
`split: test` in the header.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from string import Template

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from .config import (
    DEFAULT_PROMPT,
    FEW_SHOT_EXAMPLE_INDEX,
    NER_DATASET_URL,
    NER_TAGS,
    PROJECT_ROOT,
    PROMPTS_DIR,
    LLMScoreConfig,
)
from .data import sentence_key, tag_ids_to_labels

load_dotenv(PROJECT_ROOT / ".env")


def load_prompt_template(prompt: str = DEFAULT_PROMPT) -> Template:
    """prompts/<prompt>.txt as a string.Template with $tags, $example_tokens,
    $example_output and $tokens placeholders (re-read on every call so template
    edits show up immediately, e.g. in a notebook)."""
    path = PROMPTS_DIR / f"{prompt}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"No prompt template {path}. Available: {available}")
    return Template(path.read_text().removesuffix("\n"))


def build_ner_prompt(tokens: list, example_tokens: list, example_tags: list,
                     prompt: str = DEFAULT_PROMPT) -> str:
    example_pairs = [{"token": tok, "tag": tag} for tok, tag in zip(example_tokens, example_tags)]
    return load_prompt_template(prompt).substitute(
        tags=", ".join(f"'{tag}'" for tag in NER_TAGS),
        example_tokens=example_tokens,
        example_output=json.dumps(example_pairs, indent=2),
        tokens=tokens,
    )


def derive_sample_seed(base_seed: int, key: str, sample_idx: int) -> int:
    """Deterministic per-(sentence, sample) seed, independent of scoring order
    so an interrupted+resumed pass reproduces the same draws."""
    digest = hashlib.sha256(f"{base_seed}:{key}:{sample_idx}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def parse_ner_output(output_str: str, tokens: list) -> list[str]:
    """Parse the model's JSON output into a fixed-length list of tags matching `tokens`."""
    expected_length = len(tokens)
    cleaned_str = output_str.strip()

    if cleaned_str.startswith("```"):
        cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()

    extracted_tags = []
    try:
        parsed = json.loads(cleaned_str)
        if isinstance(parsed, list):
            extracted_tags = [item.get("tag", "O") for item in parsed if isinstance(item, dict)]
    except Exception:
        pass

    if len(extracted_tags) < expected_length:
        extracted_tags.extend(["O"] * (expected_length - len(extracted_tags)))
    else:
        extracted_tags = extracted_tags[:expected_length]

    return [tag if tag in NER_TAGS else "O" for tag in extracted_tags]


def prompt_fingerprint(few_shot_tokens: list, few_shot_tags: list,
                       prompt: str = DEFAULT_PROMPT) -> str:
    """Hash of the prompt template + few-shot example, to detect stale caches."""
    template = build_ner_prompt(["<TOKENS>"], few_shot_tokens, few_shot_tags, prompt)
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


async def get_single_sample(client: AsyncOpenAI, prompt: str, cfg: LLMScoreConfig,
                            semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        for attempt in range(cfg.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    print(f"API error after {cfg.max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


def _sentence_record(example: dict, raw_responses: list[str],
                     token_entropies: list[list[float]] | None = None) -> dict:
    """One cache line: the raw LLM responses + parsed tags for one sentence."""
    tokens = example["tokens"]
    record = {
        "key": sentence_key(example),
        "document": example["document name"],
        "sentence_id": example["sentence-ID"],
        "tokens": tokens,
        "gt_tags": tag_ids_to_labels(example["ner-tags"]),
        "raw_responses": raw_responses,
        "parsed_samples": [parse_ner_output(r, tokens) for r in raw_responses],
    }
    if token_entropies is not None:
        record["token_entropies"] = token_entropies
    return record


def load_cache(cache_path: Path, expected_header: dict | None = None) -> dict[str, dict]:
    """Load cached sentence records keyed by sentence key; validate the header."""
    cache = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line_no, line in enumerate(f):
            record = json.loads(line)
            if line_no == 0 and "header" in record:
                if expected_header is not None:
                    mismatched = {
                        k: (record["header"].get(k), v)
                        for k, v in expected_header.items()
                        if record["header"].get(k) != v
                    }
                    if mismatched:
                        raise ValueError(
                            f"Cache {cache_path} header mismatch (cached, expected): {mismatched}"
                        )
                continue
            cache[record["key"]] = record
    return cache


async def score_pool(cfg: LLMScoreConfig, dataset: Dataset, limit: int | None = None,
                     *, few_shot_example: dict | None = None,
                     split: str = "pool") -> dict[str, dict]:
    """Sample the LLM K times for every sentence, appending results to the cache.

    `dataset` is normally the pool; pass the test split with `split="test"`
    (own cache file) and the pool's `few_shot_example` so the prompt is
    identical and the test set never appears in its own prompts.
    Returns the full cache (existing + newly scored records).
    """
    if few_shot_example is None:
        few_shot_example = dataset[FEW_SHOT_EXAMPLE_INDEX]
    few_shot_tokens = few_shot_example["tokens"]
    few_shot_tags = tag_ids_to_labels(few_shot_example["ner-tags"])

    header = {
        "model": cfg.model,
        "num_samples": cfg.num_samples,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "prompt_fingerprint": prompt_fingerprint(few_shot_tokens, few_shot_tags, cfg.prompt),
        "dataset_url": NER_DATASET_URL,
    }
    # Non-default values only, so headers of pre-existing caches keep validating.
    if cfg.backend != "openrouter":
        header["backend"] = cfg.backend
    if cfg.prompt != DEFAULT_PROMPT:
        header["prompt"] = cfg.prompt
    if split != "pool":
        header["split"] = split

    cache_path = cfg.cache_path(split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path, expected_header=header)
    if not cache_path.exists():
        with open(cache_path, "w") as f:
            f.write(json.dumps({"header": header}) + "\n")

    examples = list(dataset)
    if limit is not None:
        examples = examples[:limit]
    pending = [ex for ex in examples if sentence_key(ex) not in cache]
    print(f"Scoring {split}: {len(pending)} to score, {len(cache)} cached ({cache_path})")

    if cfg.backend in ("mlx", "hf"):
        if cfg.backend == "mlx":
            from .mlx_scoring import MLXGenerator
            generator = MLXGenerator(cfg.model)
        else:
            from .hf_scoring import HFGenerator
            generator = HFGenerator(cfg.model, batch_size=cfg.batch_size)
        with open(cache_path, "a") as f:
            _score_pending_local(generator, cfg, pending, few_shot_tokens, few_shot_tags,
                                 cache, f, desc=f"Scoring {split}")
        return cache

    client = make_client()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    write_lock = asyncio.Lock()
    progress = tqdm(total=len(pending), desc=f"Scoring {split}", unit="sent")

    # Sentences run concurrently; the semaphore caps total in-flight API calls.
    async def score_sentence(example, f):
        prompt = build_ner_prompt(example["tokens"], few_shot_tokens, few_shot_tags, cfg.prompt)
        raw = await asyncio.gather(*[
            get_single_sample(client, prompt, cfg, semaphore)
            for _ in range(cfg.num_samples)
        ])
        record = _sentence_record(example, list(raw))
        async with write_lock:
            f.write(json.dumps(record) + "\n")
            f.flush()
            cache[record["key"]] = record
            progress.set_postfix_str(record["key"])
            progress.update(1)

    with open(cache_path, "a") as f:
        await asyncio.gather(*[score_sentence(ex, f) for ex in pending])
    progress.close()

    return cache


def _score_pending_local(generator, cfg: LLMScoreConfig, pending: list, few_shot_tokens: list,
                         few_shot_tags: list, cache: dict, f, desc: str = "Scoring pool") -> None:
    """In-process scoring, `generator.batch_size` sentences at a time.

    `generator` is any backend with a `.sample_batch(prompts, temperature,
    max_tokens, seeds) -> per-prompt (texts, entropies)` method; the hf backend
    decodes a whole chunk's samples as one GPU batch. Sentences are ordered by
    length so chunk mates finish at similar times (a batch decodes until its
    longest row stops). The order is deterministic and records are written a
    whole chunk at a time, so a resumed pass re-forms the same chunks and
    reproduces the same draws.
    """
    pending = sorted(pending, key=lambda ex: len(ex["tokens"]))
    progress = tqdm(total=len(pending), desc=desc, unit="sent")
    for start in range(0, len(pending), generator.batch_size):
        chunk = pending[start:start + generator.batch_size]
        keys = [sentence_key(ex) for ex in chunk]
        prompts = [build_ner_prompt(ex["tokens"], few_shot_tokens, few_shot_tags, cfg.prompt)
                   for ex in chunk]
        seeds = [[derive_sample_seed(cfg.seed, key, i) for i in range(cfg.num_samples)]
                 for key in keys]
        results = generator.sample_batch(prompts, cfg.temperature, cfg.max_tokens, seeds)
        for example, (raw, entropies) in zip(chunk, results):
            record = _sentence_record(example, raw, entropies)
            f.write(json.dumps(record) + "\n")
            cache[record["key"]] = record
        f.flush()
        progress.update(len(chunk))
    progress.close()
