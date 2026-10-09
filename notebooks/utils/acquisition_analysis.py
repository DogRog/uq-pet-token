"""Descriptive diagnostics from saved acquisition logs; no model or dataset loading."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl


def acquisition_diagnostics(selections: list[dict], results: pl.DataFrame) -> dict:
    """Validate logs and produce seed-level composition, coverage, and timing tables."""
    evaluations = {(row["seed"], row["arm"], row["round"]): row for row in results.to_dicts()}
    if len(evaluations) != results.height:
        raise ValueError("Duplicate evaluation rows for seed/arm/round.")
    groups = defaultdict(list)
    identities = {}
    for row in selections:
        if row["arm"] not in {"random", "uncertainty"}:
            raise ValueError("Unexpected acquisition arm.")
        key = (row["pool_idx"], row["word_idx"])
        identity = (row["document_name"], row["sentence_id"], row["token"], row["label"])
        if key in identities and identities[key] != identity:
            raise ValueError("Token identities disagree across seeds or arms.")
        identities[key] = identity
        groups[row["seed"], row["arm"]].append(row)
    seeds = sorted({row["seed"] for row in selections})
    if not seeds or set(groups) != {
        (seed, arm) for seed in seeds for arm in ("random", "uncertainty")
    }:
        raise ValueError("Selections must contain both arms for every seed.")
    if set(results["seed"].unique().to_list()) != set(seeds):
        raise ValueError("Selection and evaluation seeds disagree.")
    categories = {
        "Entity type": sorted(
            {value[3][2:] if value[3] != "O" else "O" for value in identities.values()}
        ),
        "BIO": sorted({value[3][0] for value in identities.values()}),
        "Entity / O": ["Entity", "O"],
    }
    composition, coverage, timing, validation = [], [], [], []
    positions = {}
    uncertainty_scores = {}
    for (seed, arm), rows in sorted(groups.items()):
        rounds = defaultdict(list)
        for row in rows:
            rounds[row["round"]].append(row)
        seen, words, sentences, documents, word_labels = set(), set(), set(), set(), set()
        counts = {grouping: Counter() for grouping in categories}
        pool_sizes = {
            row["scoreable_pool_tokens"]
            for key, row in evaluations.items()
            if key[:2] == (seed, arm)
        }
        if len(pool_sizes) != 1 or next(iter(pool_sizes)) <= 0:
            raise ValueError("Missing or inconsistent scoreable pool size.")
        pool_size = next(iter(pool_sizes))
        expected_rounds = {key[2] for key in evaluations if key[:2] == (seed, arm) and key[2] > 0}
        if set(rounds) != expected_rounds:
            raise ValueError("Selection rounds do not match evaluation rounds.")
        for round_idx, batch in sorted(rounds.items()):
            previous_count = len(seen)
            midpoint = previous_count + (len(batch) + 1) / 2
            for row in batch:
                key = (row["pool_idx"], row["word_idx"])
                if key in seen:
                    raise ValueError("A token was acquired more than once within a seed/arm.")
                seen.add(key)
                positions[seed, arm, key] = 100 * midpoint / pool_size
                if arm == "uncertainty":
                    uncertainty_scores[seed, key] = row.get("uq_score")
                words.add(row["token"].lower())
                sentences.add(row["pool_idx"])
                documents.add(row["document_name"])
                word_labels.add((row["token"].lower(), row["label"]))
                counts["Entity type"][row["label"][2:] if row["label"] != "O" else "O"] += 1
                counts["BIO"][row["label"][0]] += 1
                counts["Entity / O"]["Entity" if row["label"] != "O" else "O"] += 1
            evaluation = evaluations[seed, arm, round_idx]
            if evaluation["n_acquired"] != len(seen) or evaluation["n_new"] != len(batch):
                raise ValueError("Selection counts disagree with evaluation budgets.")
            common = {
                "seed": seed,
                "arm": arm,
                "round": round_idx,
                "n_acquired": len(seen),
                "percent_acquired": 100 * len(seen) / pool_size,
            }
            coverage.append(
                {
                    **common,
                    "distinct_words": len(words),
                    "distinct_sentences": len(sentences),
                    "distinct_documents": len(documents),
                    "distinct_word_label_pairs": len(word_labels),
                    "repeated_word_label_percent": 100 * (1 - len(word_labels) / len(seen)),
                }
            )
            for grouping, labels in categories.items():
                for label in labels:
                    composition.append(
                        {
                            **common,
                            "grouping": grouping,
                            "category": label,
                            "share_percent": 100 * counts[grouping][label] / len(seen),
                        }
                    )
        validation.append(
            {
                "seed": seed,
                "arm": arm,
                "selected_tokens": len(seen),
                "scoreable_pool_tokens": pool_size,
                "full_pool": len(seen) == pool_size,
            }
        )
    for seed in seeds:
        random_keys = {key for s, arm, key in positions if s == seed and arm == "random"}
        uq_keys = {key for s, arm, key in positions if s == seed and arm == "uncertainty"}
        for key in sorted(random_keys & uq_keys):
            document, sentence, token, label = identities[key]
            timing.append(
                {
                    "seed": seed,
                    "pool_idx": key[0],
                    "word_idx": key[1],
                    "document_name": document,
                    "sentence_id": sentence,
                    "token": token,
                    "label": label,
                    "entity_type": label[2:] if label != "O" else "O",
                    "BIO": label[0],
                    "uq_score": uncertainty_scores[seed, key],
                    "random_position_percent": positions[seed, "random", key],
                    "uq_position_percent": positions[seed, "uncertainty", key],
                    "advance_pp": positions[seed, "random", key]
                    - positions[seed, "uncertainty", key],
                }
            )
        for row in validation:
            if row["seed"] == seed:
                row["same_final_set"] = random_keys == uq_keys
                row["matched_tokens"] = len(random_keys & uq_keys)
    timing_schema = {
        "seed": pl.Int64,
        "pool_idx": pl.Int64,
        "word_idx": pl.Int64,
        "document_name": pl.String,
        "sentence_id": pl.Int64,
        "token": pl.String,
        "label": pl.String,
        "entity_type": pl.String,
        "BIO": pl.String,
        "uq_score": pl.Float64,
        "random_position_percent": pl.Float64,
        "uq_position_percent": pl.Float64,
        "advance_pp": pl.Float64,
    }
    return {
        "composition": pl.DataFrame(composition),
        "coverage": pl.DataFrame(coverage),
        "timing": pl.DataFrame(timing, schema=timing_schema),
        "validation": pl.DataFrame(validation),
    }


def paired_performance(results: pl.DataFrame) -> pl.DataFrame:
    """Pair evaluations within seed and exact acquisition budget, before averaging."""
    metrics = [
        name
        for name in (
            "entity_f1",
            "entity_precision",
            "entity_recall",
            "entity_macro_f1",
            "token_accuracy",
        )
        if name in results.columns
    ]
    keys = ["seed", "n_acquired"]
    arms = {}
    for arm in ("random", "uncertainty"):
        arms[arm] = results.filter(pl.col("arm") == arm).select(*keys, *metrics)
    paired = arms["uncertainty"].join(arms["random"], on=keys, suffix="_random", validate="1:1")
    if paired.height != arms["random"].height or paired.height != arms["uncertainty"].height:
        raise ValueError("Both arms must have evaluations at the same budgets for each seed.")
    return paired.select(
        *keys,
        *[
            ((pl.col(metric) - pl.col(f"{metric}_random")) * 100).alias(metric)
            for metric in metrics
        ],
    ).unpivot(index=keys, variable_name="metric", value_name="gap_pp")


def threshold_efficiency(
    results: pl.DataFrame, threshold: float, field: str = "entity_f1"
) -> pl.DataFrame:
    """First observed crossing of ``threshold`` by ``field`` per seed and arm.

    Unreached thresholds stay null; crossings are not interpolated.
    """
    rows = []
    for seed in sorted(results["seed"].unique().to_list()):
        row = {"seed": seed, "target_f1": threshold}
        for arm in ("random", "uncertainty"):
            hits = results.filter(
                (pl.col("seed") == seed) & (pl.col("arm") == arm) & (pl.col(field) >= threshold)
            )
            row[f"{arm}_labels"] = hits["n_acquired"].min() if hits.height else None
        reached = row["random_labels"] is not None and row["uncertainty_labels"] is not None
        row["labels_saved"] = row["random_labels"] - row["uncertainty_labels"] if reached else None
        row["status"] = (
            "Both reached"
            if reached
            else ", ".join(
                f"{arm} not reached"
                for arm in ("random", "uncertainty")
                if row[f"{arm}_labels"] is None
            )
        )
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema_overrides={
            "random_labels": pl.Int64,
            "uncertainty_labels": pl.Int64,
            "labels_saved": pl.Int64,
        },
    )


def tag_selections(
    selections: list[dict], results: pl.DataFrame, run_id: str
) -> tuple[pl.DataFrame, list[str]]:
    """Join selections to their round's progress; order labels by distinct pool tokens."""
    frame = (
        pl.DataFrame(selections)
        .with_columns(pl.lit(run_id).alias("run_id"))
        .join(
            results.select("seed", "arm", "round", "percent_acquired", "scoreable_pool_tokens"),
            on=["seed", "arm", "round"],
            how="left",
            validate="m:1",
        )
    )
    label_order = (
        frame.select("pool_idx", "word_idx", "label")
        .unique()
        .group_by("label")
        .len()
        .sort(["len", "label"], descending=[True, False])["label"]
        .to_list()
    )
    return frame, label_order


def enrichment_at_budget(composition: pl.DataFrame, n_acquired: int) -> pl.DataFrame:
    """UQ-minus-random share of each category at one budget, paired within seed."""
    at_budget = composition.filter(pl.col("n_acquired") == n_acquired)
    pairs = (
        at_budget.filter(pl.col("arm") == "uncertainty")
        .join(
            at_budget.filter(pl.col("arm") == "random"),
            on=["seed", "category"],
            suffix="_random",
            validate="1:1",
        )
        .with_columns(
            (pl.col("share_percent") - pl.col("share_percent_random")).alias("enrichment_pp")
        )
    )
    return (
        pairs.group_by("n_acquired", "category")
        .agg(
            pl.col("share_percent_random").mean().alias("random_share_percent"),
            pl.col("share_percent").mean().alias("uq_share_percent"),
            pl.col("enrichment_pp").mean(),
            (pl.col("enrichment_pp") > 0).sum().alias("seeds_enriched"),
            pl.len().alias("seeds"),
        )
        .sort("enrichment_pp", descending=True)
    )


def timing_by_group(timing: pl.DataFrame, group: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Seed-mean timing advantage per group, then a summary that weights seeds equally.

    ``group`` is ``entity_type``, ``BIO``, or ``entity_status`` (Entity or O).
    """
    seed_means = (
        timing.with_columns(
            pl.when(pl.col("label") == "O")
            .then(pl.lit("O"))
            .otherwise(pl.lit("Entity"))
            .alias("entity_status")
        )
        .group_by("seed", group)
        .agg(pl.col("advance_pp").mean(), pl.len().alias("matched_tokens"))
    )
    summary = (
        seed_means.group_by(group)
        .agg(
            pl.col("advance_pp").mean().alias("mean_advance_pp"),
            pl.col("advance_pp").min().alias("min_seed_advance_pp"),
            pl.col("advance_pp").max().alias("max_seed_advance_pp"),
            (pl.col("advance_pp") > 0).sum().alias("seeds_earlier"),
            pl.len().alias("seeds"),
        )
        .sort("mean_advance_pp", descending=True)
    )
    return seed_means, summary


def token_timing(timing: pl.DataFrame, sentences: dict[tuple[str, int], list[str]]) -> pl.DataFrame:
    """Per-token timing across paired seeds, with the token bolded in its sentence.

    ``sentences`` maps (document name, sentence ID) to the sentence's words; a missing
    sentence or a word that does not match the logged token gets a placeholder context.
    """
    tokens = (
        timing.group_by("pool_idx", "word_idx", "document_name", "sentence_id", "token", "label")
        .agg(
            pl.col("uq_score").mean().alias("mean_uncertainty_score"),
            pl.col("advance_pp").mean().alias("mean_advance_pp"),
            pl.col("advance_pp").min().alias("min_advance_pp"),
            (pl.col("advance_pp") > 0).sum().alias("seeds_earlier"),
            pl.len().alias("paired_seeds"),
        )
        .sort("mean_advance_pp", descending=True)
    )
    rows = []
    for token in tokens.to_dicts():
        words = sentences.get((token["document_name"], token["sentence_id"]))
        matches = (
            words is not None
            and token["word_idx"] < len(words)
            and words[token["word_idx"]] == token["token"]
        )
        rows.append(
            {
                **token,
                "sentence_context": " ".join(
                    f"**{word}**" if index == token["word_idx"] else word
                    for index, word in enumerate(words)
                )
                if matches
                else "Local sentence context unavailable or token mismatch",
            }
        )
    return pl.DataFrame(rows)


def pet_sentences(path: Path) -> dict[tuple[str, int], list[str]]:
    """Map (document name, sentence ID) to words from a local PET JSONL file, if present."""
    if not path.is_file():
        return {}
    sentences = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        sentences[record["document name"], record["sentence-ID"]] = record["tokens"]
    return sentences
