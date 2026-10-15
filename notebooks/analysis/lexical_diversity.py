"""Lexical diversity of the datasets: type-token ratios per sentence and per word window."""

from itertools import combinations

import polars as pl
from scipy import stats


def dataset_words(datasets: dict[str, tuple[list[dict], list[str]]]) -> pl.DataFrame:
    """Every word of each dataset in document order, with its gold tag and entity flag.

    `datasets` maps a name to its sentences (every split, with integer `ner_tags`) and its
    tag names. Words are lowercased; tokens without a letter or digit (punctuation) are
    dropped.
    """
    rows = []
    for name, (sentences, tags) in datasets.items():
        for sentence in sorted(sentences, key=lambda s: (s["document_name"], s["sentence_id"])):
            for word_idx, (word, tag) in enumerate(
                zip(sentence["tokens"], sentence["ner_tags"], strict=True)
            ):
                if not any(char.isalnum() for char in word):
                    continue
                rows.append(
                    {
                        "dataset": name,
                        "document": sentence["document_name"],
                        "sentence_id": sentence["sentence_id"],
                        "word_idx": word_idx,
                        "word": word.lower(),
                        "tag": tags[tag],
                    }
                )
    return pl.DataFrame(rows).with_columns(entity=pl.col("tag") != "O")


def sentence_ttr(words: pl.DataFrame) -> pl.DataFrame:
    """Type-token ratio and length of every sentence (punctuation dropped)."""
    return (
        words.group_by("dataset", "document", "sentence_id")
        .agg(pl.col("word").n_unique().alias("types"), pl.len().alias("words"))
        .with_columns(ttr=pl.col("types") / pl.col("words"))
    )


def window_ttr(words: pl.DataFrame, size: int, entity_only: bool = False) -> pl.DataFrame:
    """Type-token ratio of consecutive, non-overlapping windows of `size` words.

    Windows run over each dataset's text in document order; a final window shorter than
    `size` is dropped, so every value is comparable across datasets (mean segmental TTR).
    With `entity_only`, only words inside gold entities are kept before windowing.
    """
    kept = words.filter(pl.col("entity")) if entity_only else words
    return (
        kept.with_columns(window=(pl.int_range(pl.len()) // size).over("dataset"))
        .group_by("dataset", "window", maintain_order=True)
        .agg(pl.col("word").n_unique().alias("types"), pl.len().alias("words"))
        .filter(pl.col("words") == size)
        .with_columns(ttr=pl.col("types") / size)
    )


def entity_contexts(words: pl.DataFrame) -> pl.DataFrame:
    """Entity words only, each replaced by its (previous, word, next) trigram.

    The Unique-Contextualized-Entity-Ratio of Schmidt (2022): windows of these trigrams
    count distinct entity words in distinct phrasings. Neighbours are taken after
    punctuation is dropped; sentence edges read `<s>` and `</s>`.
    """
    sentence = ["dataset", "document", "sentence_id"]
    return words.with_columns(
        word=pl.concat_str(
            [
                pl.col("word").shift(1).over(sentence).fill_null("<s>"),
                pl.col("word"),
                pl.col("word").shift(-1).over(sentence).fill_null("</s>"),
            ],
            separator=" ",
        )
    ).filter(pl.col("entity"))


def pairwise_window_tests(windows: pl.DataFrame, order: list[str]) -> pl.DataFrame:
    """Mann–Whitney tests of window TTR between every pair of datasets, per word set.

    Each row compares `dataset` against an earlier `against` in `order`. Holm adjusts
    across the pairs within each `words_kept` set; pairs with fewer than two windows on
    either side are left out.
    """
    rows = []
    for (kept,), group in windows.group_by("words_kept", maintain_order=True):
        values = {name: group.filter(pl.col("dataset") == name)["ttr"].to_list() for name in order}
        pairs = [
            (later, earlier)
            for earlier, later in combinations(order, 2)
            if len(values[earlier]) > 1 and len(values[later]) > 1
        ]
        p_values = [stats.mannwhitneyu(values[a], values[b]).pvalue for a, b in pairs]
        ranked = sorted(range(len(pairs)), key=p_values.__getitem__)
        holm, running = [0.0] * len(pairs), 0.0
        for rank, index in enumerate(ranked):
            running = max(running, min(1.0, (len(pairs) - rank) * p_values[index]))
            holm[index] = running
        rows += [
            {
                "words_kept": kept,
                "dataset": a,
                "against": b,
                "ttr": sum(values[a]) / len(values[a]),
                "against_ttr": sum(values[b]) / len(values[b]),
                "windows": len(values[a]),
                "against_windows": len(values[b]),
                "p_value": float(p),
                "holm_p": h,
            }
            for (a, b), p, h in zip(pairs, p_values, holm, strict=True)
        ]
    return pl.DataFrame(rows)
