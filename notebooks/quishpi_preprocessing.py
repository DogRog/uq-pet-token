import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Quishpi preprocessing checks

    Does `data_prep` turn Quishpi's brat files into the right sentences, BIO tags, and
    splits? The notebook runs the real pipeline (`brat_sentences` and
    `load_quishpi_splits`), then checks its output against the raw files and the protocol in
    the README and ADR 0008.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Checks** lists every test with its expected and observed value. `info` rows
      describe the data and cannot fail.
    - Spans are read from the `.ann` files separately from `brat_sentences`. Each unique
      span must come back as a tagged entity over the same words, or be dropped under a
      longer span that overlaps it. Every tagged entity must come from a span.
    - **Document viewer** shows every sentence of one document with its entities and
      split.
    - Raw files are downloaded only when missing. Nothing is trained, and the validation
      holdout is not covered.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import html
    from collections import Counter

    import altair as alt
    import marimo as mo
    import polars as pl

    from uq_pet import data_prep

    return Counter, alt, html, mo, data_prep, pl


@app.cell
def _(data_prep):
    quishpi_dir = data_prep.download_quishpi()
    documents = {
        name: {
            "text": (quishpi_dir / "texts" / f"{name}.txt").read_text(encoding="utf-8"),
            "annotations": (quishpi_dir / "judgeannotations" / f"{name}.ann").read_text(
                encoding="utf-8"
            ),
        }
        for name in data_prep.QUISHPI_DOCUMENTS
    }
    sentences = [
        sentence
        for name, document in documents.items()
        for sentence in data_prep.brat_sentences(
            name, document["text"], document["annotations"], data_prep.QUISHPI_TAGS
        )
    ]
    seed_examples, pool_inputs, pool_gold, test_examples = data_prep.load_quishpi_splits(
        quishpi_dir
    )
    return (
        documents,
        pool_gold,
        pool_inputs,
        quishpi_dir,
        seed_examples,
        sentences,
        test_examples,
    )


@app.cell
def _(data_prep):
    def tag_names(sentence: dict) -> list[str]:
        return [data_prep.QUISHPI_TAGS[tag] for tag in sentence["ner_tags"]]

    def decode_entities(tags: list[str]) -> list[tuple[int, int, str]]:
        """Return (first word, end word, type) runs; a stray I- tag starts a new run."""
        entities = []
        for index, tag in enumerate(tags):
            if tag == "O":
                continue
            if (
                tag.startswith("I-")
                and entities
                and entities[-1][1] == index
                and entities[-1][2] == tag[2:]
            ):
                entities[-1][1] = index + 1
            else:
                entities.append([index, index + 1, tag[2:]])
        return [tuple(entity) for entity in entities]

    return decode_entities, tag_names


@app.cell
def _(Counter, decode_entities, documents, data_prep, sentences, tag_names):
    span_rows = []
    document_rows = []
    word_problems = []
    tag_problems = []
    unexplained_entities = []
    for name, document in documents.items():
        text = document["text"]
        encoded = text.encode()
        words = [(m.start(), m.end(), m.group()) for m in data_prep.WORD_PATTERN.finditer(text)]
        doc_sentences = [s for s in sentences if s["document_name"] == name]
        doc_tokens = [token for s in doc_sentences for token in s["tokens"]]
        if doc_tokens != [w[2] for w in words] or "".join(doc_tokens) != "".join(text.split()):
            word_problems.append(name)
        for sentence in doc_sentences:
            tags = tag_names(sentence)
            for index, tag in enumerate(tags):
                if tag.startswith("I-") and (index == 0 or tags[index - 1][2:] != tag[2:]):
                    tag_problems.append((name, sentence["sentence_id"], index, tag))

        # Word indices run over the whole document, matching `words`.
        entities = decode_entities([tag for s in doc_sentences for tag in tag_names(s)])
        first_span_ids = {}
        kept_ranges = set()
        for line in document["annotations"].splitlines():
            if not line.startswith("T"):
                continue
            span_id, info, surface = line.split("\t")
            entity_type, byte_start, byte_end = info.split(" ")
            byte_start, byte_end = int(byte_start), int(byte_end)
            start = len(encoded[:byte_start].decode())
            end = len(encoded[:byte_end].decode())
            covered = [i for i, w in enumerate(words) if w[0] < end and w[1] > start]
            word_range = (covered[0], covered[-1] + 1, entity_type)
            overlapping = [e for e in entities if e[0] < word_range[1] and e[1] > word_range[0]]
            if (start, end, entity_type) in first_span_ids:
                outcome = "duplicate"
            elif word_range in entities:
                outcome = "kept"
                kept_ranges.add(word_range)
            elif overlapping:
                outcome = "dropped"
            else:
                outcome = "missing"
            first_span_ids.setdefault((start, end, entity_type), span_id)
            winner = overlapping[0] if outcome == "dropped" else None
            span_rows.append(
                {
                    "document": name,
                    "span_id": span_id,
                    "type": entity_type,
                    "surface": surface,
                    "outcome": outcome,
                    "same_as": first_span_ids[(start, end, entity_type)]
                    if outcome == "duplicate"
                    else None,
                    "dropped_under": f"{winner[2]}: "
                    + " ".join(w[2] for w in words[winner[0] : winner[1]])
                    if winner
                    else None,
                    "winner_is_longer": words[winner[1] - 1][1] - words[winner[0]][0] >= end - start
                    if winner
                    else None,
                    "matches_at_bytes": text[start:end] == surface,
                    "matches_as_characters": text[byte_start:byte_end] == surface,
                    "splits_word": words[covered[0]][0] < start or words[covered[-1]][1] > end,
                }
            )
        unexplained_entities += [
            (name, " ".join(w[2] for w in words[e[0] : e[1]]), e[2])
            for e in entities
            if e not in kept_ranges
        ]

        annotation_lines = document["annotations"].splitlines()
        type_counts = Counter(
            line.split("\t")[1].split(" ")[0] for line in annotation_lines if line.startswith("T")
        )
        document_rows.append(
            {
                "document": name,
                "characters": len(text),
                "non_ascii_characters": sum(not char.isascii() for char in text),
                "sentences": len(doc_sentences),
                "words": len(doc_tokens),
                "Action": type_counts["Action"],
                "Entity": type_counts["Entity"],
                "Condition": type_counts["Condition"],
                "other_span_types": sum(type_counts.values())
                - type_counts["Action"]
                - type_counts["Entity"]
                - type_counts["Condition"],
                "ignored_lines": sum(not line.startswith("T") for line in annotation_lines),
            }
        )
    return document_rows, span_rows, tag_problems, unexplained_entities, word_problems


@app.cell
def _(
    data_prep,
    pool_gold,
    pool_inputs,
    quishpi_dir,
    seed_examples,
    sentences,
    test_examples,
):
    def sentence_key(sentence: dict) -> tuple[str, int]:
        return sentence["document_name"], sentence["sentence_id"]

    sentence_by_key = {sentence_key(s): s for s in sentences}
    split_members = {
        "bootstrap": [sentence_key(s) for s in seed_examples],
        "pool": [sentence_key(s) for s in pool_inputs],
        "test": [sentence_key(s) for s in test_examples],
    }
    split_of = {key: split for split, keys in split_members.items() for key in keys}
    splits_partition = len(split_of) == sum(map(len, split_members.values())) and set(
        split_of
    ) == set(sentence_by_key)
    labels_intact = (
        all(s == sentence_by_key[sentence_key(s)] for s in [*seed_examples, *test_examples])
        and all(p["tokens"] == sentence_by_key[sentence_key(p)]["tokens"] for p in pool_inputs)
        and pool_gold
        == {
            (p["pool_idx"], word_idx): tag
            for p in pool_inputs
            for word_idx, tag in enumerate(sentence_by_key[sentence_key(p)]["ner_tags"])
        }
    )
    pool_is_label_free = all("ner_tags" not in p for p in pool_inputs)
    split_is_reproducible = data_prep.load_quishpi_splits(quishpi_dir) == (
        seed_examples,
        pool_inputs,
        pool_gold,
        test_examples,
    )
    return (
        labels_intact,
        pool_is_label_free,
        split_is_reproducible,
        split_members,
        split_of,
        splits_partition,
    )


@app.cell
def _(
    document_rows,
    labels_intact,
    data_prep,
    pool_is_label_free,
    span_rows,
    split_is_reproducible,
    split_members,
    splits_partition,
    tag_problems,
    unexplained_entities,
    word_problems,
):
    def check(name: str, expected, observed, status: str | None = None) -> dict:
        return {
            "check": name,
            "expected": str(expected),
            "observed": str(observed),
            "status": status or ("pass" if expected == observed else "FAIL"),
        }

    outcomes = [row["outcome"] for row in span_rows]
    dropped = [row for row in span_rows if row["outcome"] == "dropped"]
    checks = [
        check("Documents", 18, len(document_rows)),
        check(
            "Span types other than Action, Entity, Condition",
            0,
            sum(row["other_span_types"] for row in document_rows),
        ),
        check(
            "Spans that do not match their text at byte offsets",
            0,
            sum(not row["matches_at_bytes"] for row in span_rows),
        ),
        check(
            "Spans that would break if offsets were read as characters",
            "",
            sum(not row["matches_as_characters"] for row in span_rows),
            "info",
        ),
        check("Documents whose words do not cover the text in order", 0, len(word_problems)),
        check("Spans that cut a word", 0, sum(row["splits_word"] for row in span_rows)),
        check("Span lines", "", len(outcomes), "info"),
        check("Exact duplicate span lines", "", outcomes.count("duplicate"), "info"),
        check("Spans kept as an entity over the same words", "", outcomes.count("kept"), "info"),
        check("Spans dropped under an overlapping span", "", len(dropped), "info"),
        check(
            "Dropped spans whose winner is shorter",
            0,
            sum(not row["winner_is_longer"] for row in dropped),
        ),
        check("Spans missing from the tags", 0, outcomes.count("missing")),
        check("Tagged entities with no span", 0, len(unexplained_entities)),
        check("I- tags that start a sentence or follow another type", 0, len(tag_problems)),
        check("Sentences", "", sum(row["sentences"] for row in document_rows), "info"),
        check(
            "Bootstrap, pool, and test sentences",
            (data_prep.N_SEED_SENTENCES, 155, 41),
            tuple(len(split_members[split]) for split in ("bootstrap", "pool", "test")),
        ),
        check("Splits are disjoint and cover every sentence", True, splits_partition),
        check("Pool inputs carry no tags", True, pool_is_label_free),
        check("Split words and tags equal the sentences'", True, labels_intact),
        check("Loading again gives the same split", True, split_is_reproducible),
    ]
    return (checks,)


@app.cell(hide_code=True)
def _(checks, mo, pl):
    failures = [row["check"] for row in checks if row["status"] == "FAIL"]
    mo.vstack(
        [
            mo.md("## Checks"),
            mo.callout(
                mo.md(f"**{len(failures)} failed:** " + "; ".join(failures))
                if failures
                else mo.md("**Every check passed.**"),
                kind="danger" if failures else "success",
            ),
            mo.ui.table(pl.DataFrame(checks), selection=None, page_size=30),
        ]
    )
    return


@app.cell(hide_code=True)
def _(document_rows, mo, pl, span_rows, tag_problems, unexplained_entities):
    spans = pl.DataFrame(span_rows)
    mo.vstack(
        [
            mo.md("## Documents and spans"),
            mo.ui.tabs(
                {
                    "Documents": mo.ui.table(pl.DataFrame(document_rows), selection=None),
                    "Spans not kept": mo.ui.table(
                        spans.filter(pl.col("outcome") != "kept").select(
                            "document",
                            "span_id",
                            "type",
                            "surface",
                            "outcome",
                            "same_as",
                            "dropped_under",
                        ),
                        selection=None,
                    ),
                    "Spans read at byte offsets": mo.ui.table(
                        spans.filter(~pl.col("matches_as_characters")).select(
                            "document", "span_id", "type", "surface"
                        ),
                        selection=None,
                    ),
                    "All spans": mo.ui.table(spans, selection=None),
                    "Tag problems": mo.ui.table(
                        pl.DataFrame(
                            [
                                *[
                                    {"document": d, "problem": f"sentence {s}, word {w}: {t}"}
                                    for d, s, w, t in tag_problems
                                ],
                                *[
                                    {"document": d, "problem": f"untraced {t}: {words}"}
                                    for d, words, t in unexplained_entities
                                ],
                            ],
                            schema={"document": pl.String, "problem": pl.String},
                        ),
                        selection=None,
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(decode_entities, pl, sentences, split_of, tag_names):
    sentence_frame = pl.DataFrame(
        [
            {
                "document": s["document_name"],
                "sentence_id": s["sentence_id"],
                "split": split_of[(s["document_name"], s["sentence_id"])],
                "words": len(s["tokens"]),
                "tagged_words": sum(tag != 0 for tag in s["ner_tags"]),
            }
            for s in sentences
        ]
    )
    entity_frame = pl.DataFrame(
        [
            {
                "split": split_of[(s["document_name"], s["sentence_id"])],
                "type": entity_type,
                "words": end - first,
            }
            for s in sentences
            for first, end, entity_type in decode_entities(tag_names(s))
        ]
    )
    return entity_frame, sentence_frame


@app.cell(hide_code=True)
def _(alt, entity_frame, mo, pl, sentence_frame):
    split_summary = (
        sentence_frame.group_by("split")
        .agg(
            pl.len().alias("sentences"),
            pl.col("words").sum(),
            (pl.col("tagged_words").sum() / pl.col("words").sum()).round(3).alias("tagged_share"),
            pl.col("document").n_unique().alias("documents"),
        )
        .join(
            entity_frame.pivot(on="type", index="split", values="words", aggregate_function="len"),
            on="split",
        )
        .sort("split")
    )
    type_shares = (
        entity_frame.group_by("split", "type")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("split")).alias("share"))
    )
    share_chart = (
        alt.Chart(type_shares)
        .mark_bar()
        .encode(
            x=alt.X("share:Q", title="Share of the split's entities", axis=alt.Axis(format="%")),
            y=alt.Y("split:N", title=None),
            color=alt.Color("type:N", title="Type"),
            tooltip=["split", "type", alt.Tooltip("len:Q", title="entities")],
        )
        .properties(width=420, height=100, title="Entity types by split")
    )
    length_chart = (
        alt.Chart(sentence_frame)
        .mark_bar()
        .encode(
            x=alt.X("words:Q", bin=alt.Bin(step=4), title="Words per sentence"),
            y=alt.Y("count():Q", title="Sentences"),
            color=alt.Color("split:N", title="Split"),
        )
        .properties(width=420, height=180, title="Sentence lengths")
    )
    entity_lengths = (
        entity_frame.group_by("type")
        .agg(
            pl.len().alias("entities"),
            pl.col("words").mean().round(2).alias("mean_words"),
            pl.col("words").max().alias("max_words"),
        )
        .sort("type")
    )
    shared_documents = (
        sentence_frame.group_by("document")
        .agg(pl.col("split").unique().sort().str.join(", ").alias("splits"))
        .sort("document")
    )
    mo.vstack(
        [
            mo.md("""
    ## Splits and statistics

    ADR 0008 splits sentences, not documents, so one text can sit in both the pool and
    the test split. The last table shows which documents do.
    """),
            mo.ui.table(split_summary, selection=None),
            mo.hstack([share_chart, length_chart], justify="start", wrap=True),
            mo.hstack(
                [
                    mo.ui.table(entity_lengths, selection=None, label="Entity lengths in words"),
                    mo.ui.table(shared_documents, selection=None, label="Splits per document"),
                ],
                justify="start",
                widths=[1, 2],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, data_prep):
    document_selector = mo.ui.dropdown(
        options=list(data_prep.QUISHPI_DOCUMENTS),
        value=data_prep.QUISHPI_DOCUMENTS[0],
        label="Document",
    )
    return (document_selector,)


@app.cell(hide_code=True)
def _(
    decode_entities,
    document_selector,
    documents,
    html,
    mo,
    pl,
    sentences,
    span_rows,
    split_of,
    tag_names,
):
    entity_colors = {
        "Action": "rgba(59, 130, 246, 0.25)",
        "Entity": "rgba(16, 185, 129, 0.25)",
        "Condition": "rgba(245, 158, 11, 0.3)",
    }

    def render(sentence: dict) -> str:
        tokens = sentence["tokens"]
        pieces = [html.escape(token) for token in tokens]
        for first, end, entity_type in reversed(decode_entities(tag_names(sentence))):
            pieces[first:end] = [
                f'<span style="background:{entity_colors[entity_type]};border-radius:3px;'
                f'padding:1px 3px" title="{entity_type}">{" ".join(pieces[first:end])}'
                f"<sub style='opacity:.7'> {entity_type}</sub></span>"
            ]
        return " ".join(pieces)

    selected = document_selector.value
    legend = " ".join(
        f'<span style="background:{color};border-radius:3px;padding:1px 6px">{name}</span>'
        for name, color in entity_colors.items()
    )
    rows = "".join(
        f"<tr><td style='opacity:.6;padding-right:12px;vertical-align:top'>"
        f"{s['sentence_id']}&nbsp;·&nbsp;{split_of[(selected, s['sentence_id'])]}</td>"
        f"<td style='padding-bottom:6px;line-height:1.9'>{render(s)}</td></tr>"
        for s in sentences
        if s["document_name"] == selected
    )
    not_kept = pl.DataFrame(span_rows).filter(
        (pl.col("document") == selected) & (pl.col("outcome") != "kept")
    )
    mo.vstack(
        [
            mo.md("## Document viewer"),
            document_selector,
            mo.Html(f"<div>{legend}</div><table>{rows}</table>"),
            mo.accordion(
                {
                    "Raw text": mo.plain_text(documents[selected]["text"]),
                    "Raw annotations": mo.plain_text(documents[selected]["annotations"]),
                    f"Spans not kept ({not_kept.height})": mo.ui.table(
                        not_kept.select("span_id", "type", "surface", "outcome", "dropped_under"),
                        selection=None,
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
