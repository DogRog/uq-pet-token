"""How hard each dataset's tagging task is, read from its pinned splits; nothing is trained.

Every function takes datasets as `{name: (sentences, tag names)}`, where each sentence
carries its `split` ("bootstrap", "pool", or "test") and integer `ner_tags`. Entities are
chunked like seqeval's default mode, the same chunking that exact-match F1 scores.
"""

import math
import random
from collections import Counter

import polars as pl
from seqeval.metrics.sequence_labeling import get_entities

from uq_pet.token_model import evaluate_predictions

SPLITS = ("bootstrap", "pool", "test")
MEASURES = {
    "entity_words_pct": "Words inside an entity (%)",
    "type_entropy": "Type entropy (normalised)",
    "rare_type_share": "Types with fewer than {rare_below} mentions",
    "multi_word_share": "Multi-word entities",
    "mean_span_words": "Mean entity length (words)",
    "capitalised_share": "Entities starting with a capital",
    "unseen_entity_rate": "Test entities unseen in the pool",
    "ambiguous_word_share": "Words whose form has several types",
    "type_entropy_given_word": "Type entropy given the word (bits)",
    "lookup_f1": "Lookup-baseline test F1",
}


def _names(sentence: dict, tags: list[str]) -> list[str]:
    return [tags[tag] for tag in sentence["ner_tags"]]


def _type(tag: str) -> str:
    return "O" if tag == "O" else tag[2:]


def entity_types(tags: list[str]) -> list[str]:
    """The entity types of a tag set, in tag order."""
    return list(dict.fromkeys(_type(tag) for tag in tags if tag != "O"))


def mentions(sentences: list[dict], tags: list[str]) -> list[dict]:
    """Every gold entity: its split, type, length in words, and lowercased surface."""
    rows = []
    for sentence in sentences:
        for entity_type, start, end in get_entities(_names(sentence, tags)):
            words = sentence["tokens"][start : end + 1]
            rows.append(
                {
                    "split": sentence["split"],
                    "type": entity_type,
                    "words": end + 1 - start,
                    "surface": " ".join(words).lower(),
                    "sentence_initial": start == 0,
                    "capitalised": words[0][:1].isupper(),
                }
            )
    return rows


def overview(datasets: dict[str, tuple[list[dict], list[str]]]) -> pl.DataFrame:
    """Documents, sentences per split, words, entities, and types of every dataset."""
    rows = []
    for name, (sentences, tags) in datasets.items():
        per_split = Counter(sentence["split"] for sentence in sentences)
        words = sum(len(sentence["tokens"]) for sentence in sentences)
        tagged = sum(tag != "O" for sentence in sentences for tag in _names(sentence, tags))
        rows.append(
            {
                "dataset": name,
                "documents": len({sentence["document_name"] for sentence in sentences}),
                **{f"{split}_sentences": per_split[split] for split in SPLITS},
                "words": words,
                "entities": len(mentions(sentences, tags)),
                "types": len(entity_types(tags)),
                "entity_words_pct": 100 * tagged / words,
            }
        )
    return pl.DataFrame(rows)


def type_frequencies(datasets: dict[str, tuple[list[dict], list[str]]]) -> pl.DataFrame:
    """Each type's share of its dataset's pool entities, ranked; unseen types rank last."""
    frames = []
    for name, (sentences, tags) in datasets.items():
        counts = Counter(m["type"] for m in mentions(sentences, tags) if m["split"] == "pool")
        ranked = sorted(entity_types(tags), key=lambda t: -counts[t])
        total = counts.total()
        frames.append(
            pl.DataFrame(
                {
                    "dataset": name,
                    "rank": range(1, len(ranked) + 1),
                    "type": ranked,
                    "mentions": [counts[t] for t in ranked],
                    "share": [counts[t] / total for t in ranked],
                }
            )
        )
    return pl.concat(frames)


def lookup_predictions(train: list[dict], test: list[dict], tags: list[str]) -> list[list[str]]:
    """Tag each test word with its lowercased form's most frequent type in `train`.

    Unknown words get O. A word continues the entity of the word before it when both get
    the same type, so a lookup can still predict multi-word entities.
    """
    seen: dict[str, Counter] = {}
    for sentence in train:
        for word, tag in zip(sentence["tokens"], _names(sentence, tags), strict=True):
            seen.setdefault(word.lower(), Counter())[_type(tag)] += 1
    predictions = []
    for sentence in test:
        types = [
            seen[word.lower()].most_common(1)[0][0] if word.lower() in seen else "O"
            for word in sentence["tokens"]
        ]
        predictions.append(
            [
                "O"
                if entity_type == "O"
                else ("I-" if index and types[index - 1] == entity_type else "B-") + entity_type
                for index, entity_type in enumerate(types)
            ]
        )
    return predictions


def pool_profile(
    pool: list[dict], test: list[dict], tags: list[str], rare_below: int = 10
) -> dict[str, float]:
    """The difficulty measures of `MEASURES` for one pool, scored against the test split.

    The capital-letter share leaves out entities that open a sentence, which are
    capitalised whatever they are. Word forms are lowercased.
    """
    pool_mentions = mentions(pool, tags)
    types = entity_types(tags)
    type_counts = Counter(m["type"] for m in pool_mentions)
    shares = [count / len(pool_mentions) for count in type_counts.values()]
    inside = [m for m in pool_mentions if not m["sentence_initial"]]
    pool_surfaces = {m["surface"] for m in pool_mentions}
    test_mentions = mentions(test, tags)
    word_types: dict[str, Counter] = {}
    for sentence in pool:
        for word, tag in zip(sentence["tokens"], _names(sentence, tags), strict=True):
            word_types.setdefault(word.lower(), Counter())[_type(tag)] += 1
    words = sum(counts.total() for counts in word_types.values())
    conditional_entropy = -sum(
        count * math.log2(count / counts.total())
        for counts in word_types.values()
        for count in counts.values()
    )
    gold = [_names(sentence, tags) for sentence in test]
    return {
        "entity_words_pct": 100 * sum(m["words"] for m in pool_mentions) / words,
        "type_entropy": -sum(p * math.log(p) for p in shares) / math.log(len(types))
        if len(types) > 1
        else 0.0,
        "rare_type_share": sum(type_counts[t] < rare_below for t in types) / len(types),
        "multi_word_share": sum(m["words"] > 1 for m in pool_mentions) / len(pool_mentions),
        "mean_span_words": sum(m["words"] for m in pool_mentions) / len(pool_mentions),
        "capitalised_share": sum(m["capitalised"] for m in inside) / len(inside)
        if inside
        else math.nan,
        "unseen_entity_rate": sum(m["surface"] not in pool_surfaces for m in test_mentions)
        / len(test_mentions),
        "ambiguous_word_share": sum(
            counts.total() for counts in word_types.values() if len(counts) > 1
        )
        / words,
        "type_entropy_given_word": conditional_entropy / words,
        "lookup_f1": evaluate_predictions(gold, lookup_predictions(pool, test, tags))["entity_f1"],
    }


def size_matched_profile(
    datasets: dict[str, tuple[list[dict], list[str]]],
    pool_sentences: int,
    draws: int = 200,
    seed: int = 0,
    rare_below: int = 10,
) -> pl.DataFrame:
    """Mean and 95% interval of every measure over pools of `pool_sentences` sentences.

    Each draw samples pool sentences without replacement with a local seeded RNG, so a
    larger pool is compared at the size of the smallest. A pool no larger than
    `pool_sentences` is used whole, and its interval collapses to one value.
    """
    rows = []
    for name, (sentences, tags) in datasets.items():
        pool = [sentence for sentence in sentences if sentence["split"] == "pool"]
        test = [sentence for sentence in sentences if sentence["split"] == "test"]
        rng = random.Random(seed)
        whole = len(pool) <= pool_sentences
        samples = [
            pool if whole else rng.sample(pool, pool_sentences)
            for _ in range(1 if whole else draws)
        ]
        values = pl.DataFrame([pool_profile(sample, test, tags, rare_below) for sample in samples])
        for measure in MEASURES:
            column = values[measure].drop_nans()
            rows.append(
                {
                    "dataset": name,
                    "measure": measure,
                    "mean": column.mean(),
                    "low": column.quantile(0.025, "linear"),
                    "high": column.quantile(0.975, "linear"),
                    "draws": len(samples),
                }
            )
    return pl.DataFrame(rows)


def subword_fertility(sentences: list[dict], tokenizer) -> float:
    """Subwords per word when the tokenizer splits these pre-split words, untruncated."""
    encoding = tokenizer(
        [sentence["tokens"] for sentence in sentences],
        is_split_into_words=True,
        add_special_tokens=False,
    )
    subwords = sum(
        sum(word is not None for word in encoding.word_ids(index))
        for index in range(len(sentences))
    )
    return subwords / sum(len(sentence["tokens"]) for sentence in sentences)
