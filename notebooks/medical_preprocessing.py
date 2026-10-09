import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # MedicalProcessInstruks preprocessing checks

    Does `data_prep` turn the released MedicalProcessInstruks files into the right BIO
    tags and splits? The notebook runs the real pipeline (`download_medical` and
    `load_medical_splits`), then checks its output against the raw files and the
    protocol in the README and ADR 0008.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Checks** lists every test with its expected and observed value. `info` rows
      describe the data and cannot fail.
    - The release gives integer tags without names. `data_prep` reads an odd id as the
      begin tag of a type and the next even id as its inside tag, then names the type
      `T` plus its begin id. The pairing check counts inside tags that do not follow
      their own type under that reading and under the opposite one. The few strays left
      open a sentence: the release cut an entity at a sentence break such as "e.g.".
    - Every word's split tag is compared with the raw id it came from, so a wrong
      conversion or a reordered sentence fails.
    - **Types** shows the most common words of each unnamed type, to judge whether each
      id pair forms one coherent entity type.
    - **Document viewer** shows every sentence of one document with its entities and
      split.
    - The dataset is gated: accept its conditions on the Hub and run
      `uv run hf auth login` once. Nothing is trained, and the validation holdout is not
      covered.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import html
    import json
    from collections import Counter

    import altair as alt
    import marimo as mo
    import polars as pl

    from uq_pet import data_prep

    return Counter, alt, data_prep, html, json, mo, pl


@app.cell
def _(data_prep, json, mo):
    medical_paths, access_message = None, None
    try:
        medical_paths = data_prep.download_medical()
    except RuntimeError as access_failure:
        access_message = str(access_failure)
    mo.stop(
        access_message is not None,
        mo.callout(mo.md(f"**Cannot load MedicalProcessInstruks.** {access_message}"), "danger"),
    )
    raw_rows = [
        {"file": path.name, **json.loads(line)}
        for path in medical_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    seed_examples, pool_inputs, pool_gold, test_examples = data_prep.load_medical_splits(
        medical_paths
    )
    return (
        medical_paths,
        pool_gold,
        pool_inputs,
        raw_rows,
        seed_examples,
        test_examples,
    )


@app.cell
def _(data_prep):
    def raw_type(raw_tag: int) -> str:
        """Name a raw id's type by its begin id, the odd id of its pair."""
        return f"T{raw_tag if raw_tag % 2 else raw_tag - 1:02d}"

    def tag_names(sentence: dict) -> list[str]:
        return [data_prep.MEDICAL_TAGS[tag] for tag in sentence["ner_tags"]]

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

    return decode_entities, raw_type, tag_names


@app.cell
def _(pool_gold, pool_inputs, seed_examples, test_examples):
    def sentence_key(sentence: dict) -> tuple[str, int]:
        return sentence["document_name"], sentence["sentence_id"]

    split_sentences = {
        "bootstrap": seed_examples,
        "pool": [
            {
                **{field: p[field] for field in ("document_name", "sentence_id", "tokens")},
                "ner_tags": [pool_gold[(p["pool_idx"], i)] for i in range(len(p["tokens"]))],
            }
            for p in pool_inputs
        ],
        "test": test_examples,
    }
    split_of = {
        sentence_key(s): split for split, members in split_sentences.items() for s in members
    }
    sentences = [s for members in split_sentences.values() for s in members]
    sentence_by_key = {sentence_key(s): s for s in sentences}
    splits_partition = len(split_of) == len(sentences) == len(sentence_by_key)
    pool_is_label_free = all("ner_tags" not in p for p in pool_inputs)
    pool_gold_is_complete = len(pool_gold) == sum(len(p["tokens"]) for p in pool_inputs)
    return (
        pool_gold_is_complete,
        pool_is_label_free,
        sentence_by_key,
        sentences,
        split_of,
        split_sentences,
        splits_partition,
    )


@app.cell
def _(Counter, data_prep, raw_rows, raw_type, sentence_by_key):
    raw_keys = Counter((row["document_unique_id"], row["sentence_id"]) for row in raw_rows)
    ids_by_document = {}
    for row in raw_rows:
        ids_by_document.setdefault(row["document_unique_id"], []).append(row["sentence_id"])

    length_mismatches = sum(len(row["tokens"]) != len(row["ner_tags"]) for row in raw_rows)
    blank_words = sum(not token.strip() for row in raw_rows for token in row["tokens"])
    raw_tag_counts = Counter(tag for row in raw_rows for tag in row["ner_tags"])
    used_types = sorted({raw_type(tag) for tag in raw_tag_counts if tag})

    conversion_mismatches = []
    stray_rows = []
    opposite_strays = 0
    raw_by_key = {(row["document_unique_id"], row["sentence_id"]): row for row in raw_rows}
    for row in raw_rows:
        key = (row["document_unique_id"], row["sentence_id"])
        converted = sentence_by_key.get(key)
        expected = [
            "O" if tag == 0 else f"{'B' if tag % 2 else 'I'}-{raw_type(tag)}"
            for tag in row["ner_tags"]
        ]
        if (
            converted is None
            or converted["tokens"] != row["tokens"]
            or expected != [data_prep.MEDICAL_TAGS[tag] for tag in converted["ner_tags"]]
        ):
            conversion_mismatches.append(key)

        tags = row["ner_tags"]
        for index, tag in enumerate(tags):
            previous = tags[index - 1] if index else None
            if tag and tag % 2 == 0 and previous not in (tag, tag - 1):
                earlier = raw_by_key.get((key[0], key[1] - 1), {"tokens": []})
                stray_rows.append(
                    {
                        "file": row["file"],
                        "document": key[0],
                        "sentence_id": key[1],
                        "word": index,
                        "tag": f"I-{raw_type(tag)}",
                        "after": "sentence start" if previous is None else str(previous),
                        "previous_sentence_ends": " ".join(earlier["tokens"][-6:])
                        if index == 0
                        else None,
                        "context": " ".join(row["tokens"][max(index - 3, 0) : index + 4]),
                    }
                )
            # Under the opposite reading, even ids begin and the next odd id continues.
            if tag % 2 == 1 and previous not in (tag, tag + 1):
                opposite_strays += 1
    return (
        blank_words,
        conversion_mismatches,
        ids_by_document,
        length_mismatches,
        opposite_strays,
        raw_keys,
        stray_rows,
        used_types,
    )


@app.cell
def _(Counter, decode_entities, raw_rows, raw_type, split_of, tag_names, sentence_by_key):
    document_rows = []
    for document_name in dict.fromkeys(row["document_unique_id"] for row in raw_rows):
        document_sentences = [row for row in raw_rows if row["document_unique_id"] == document_name]
        document_rows.append(
            {
                "document": document_name,
                "document_id": sorted({row["document_id"] for row in document_sentences}),
                "file": sorted({row["file"] for row in document_sentences}),
                "sentences": len(document_sentences),
                "words": sum(len(row["tokens"]) for row in document_sentences),
                "tagged_share": round(
                    sum(tag != 0 for row in document_sentences for tag in row["ner_tags"])
                    / sum(len(row["ner_tags"]) for row in document_sentences),
                    3,
                ),
                "splits": ", ".join(
                    sorted(
                        {
                            split_of[(document_name, row["sentence_id"])]
                            for row in document_sentences
                        }
                    )
                ),
            }
        )

    entity_counts = Counter()
    surface_forms = {}
    raw_tag_counts_by_type = Counter()
    for release_row in raw_rows:
        split_sentence = sentence_by_key[
            (release_row["document_unique_id"], release_row["sentence_id"])
        ]
        for first, end, entity_type in decode_entities(tag_names(split_sentence)):
            entity_counts[(entity_type, release_row["file"])] += 1
            surface_forms.setdefault(entity_type, Counter())[
                " ".join(split_sentence["tokens"][first:end]).lower()
            ] += 1
        raw_tag_counts_by_type.update(
            (raw_type(tag), "B" if tag % 2 else "I") for tag in release_row["ner_tags"] if tag
        )
    files = sorted({row["file"] for row in raw_rows})
    type_rows = [
        {
            "type": entity_type,
            "begin_id": int(entity_type[1:]),
            "B_words": raw_tag_counts_by_type[(entity_type, "B")],
            "I_words": raw_tag_counts_by_type[(entity_type, "I")],
            **{
                f"entities_{name.split('_')[0]}": entity_counts[(entity_type, name)]
                for name in files
            },
            "common_words": " | ".join(
                f"{form} ({count})" for form, count in surface_forms[entity_type].most_common(6)
            ),
        }
        for entity_type in sorted(surface_forms)
    ]
    return document_rows, type_rows


@app.cell
def _(
    blank_words,
    conversion_mismatches,
    data_prep,
    document_rows,
    ids_by_document,
    length_mismatches,
    medical_paths,
    opposite_strays,
    pool_gold,
    pool_gold_is_complete,
    pool_inputs,
    pool_is_label_free,
    raw_keys,
    raw_rows,
    seed_examples,
    split_of,
    split_sentences,
    splits_partition,
    stray_rows,
    test_examples,
    type_rows,
    used_types,
):
    def check(name: str, expected, observed, status: str | None = None) -> dict:
        return {
            "check": name,
            "expected": str(expected),
            "observed": str(observed),
            "status": status or ("pass" if expected == observed else "FAIL"),
        }

    files_by_document = {}
    for raw_row in raw_rows:
        files_by_document.setdefault(raw_row["document_unique_id"], set()).add(raw_row["file"])
    test_file_types = {row["type"] for row in type_rows if row["entities_test"]}
    checks = [
        check("Files", list(data_prep.MEDICAL_FILES), [path.name for path in medical_paths]),
        check("Sentences", 325, len(raw_rows)),
        check("Documents", 13, len(document_rows)),
        check(
            "Documents in both released files",
            0,
            sum(len(found) > 1 for found in files_by_document.values()),
        ),
        check("Repeated (document, sentence) keys", 0, sum(n > 1 for n in raw_keys.values())),
        check(
            "Documents whose sentence ids are not 0, 1, 2, …",
            0,
            sum(sorted(ids) != list(range(len(ids))) for ids in ids_by_document.values()),
        ),
        check("Sentences with unequal word and tag counts", 0, length_mismatches),
        check("Blank words", 0, blank_words),
        check("Types used", list(data_prep.MEDICAL_TAGS[1::2]), [f"B-{t}" for t in used_types]),
        check("I tags off their own type (odd id begins)", "", len(stray_rows), "info"),
        check("I tags off their own type (even id begins)", "", opposite_strays, "info"),
        check(
            "Odd-begin reading fits better than the opposite",
            True,
            len(stray_rows) < opposite_strays,
        ),
        check(
            "Stray I tags inside a sentence rather than at its start",
            0,
            sum(row["word"] != 0 for row in stray_rows),
        ),
        check("Sentences whose split tags differ from the raw ids", 0, len(conversion_mismatches)),
        check(
            "Types missing from the released test file",
            "",
            ", ".join(t for t in used_types if t not in test_file_types),
            "info",
        ),
        check(
            "Bootstrap, pool, and test sentences",
            (data_prep.N_SEED_SENTENCES, 255, 65),
            tuple(len(split_sentences[split]) for split in ("bootstrap", "pool", "test")),
        ),
        check(
            "Splits are disjoint and cover every sentence",
            True,
            splits_partition and set(split_of) == set(raw_keys),
        ),
        check("Pool inputs carry no tags", True, pool_is_label_free),
        check("Pool labels cover every pool word", True, pool_gold_is_complete),
        check(
            "Loading again gives the same split",
            True,
            data_prep.load_medical_splits(medical_paths)
            == (seed_examples, pool_inputs, pool_gold, test_examples),
        ),
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
def _(Counter, document_rows, mo, pl, raw_rows, split_of, stray_rows, type_rows):
    sentence_counts = Counter(tuple(row["tokens"]) for row in raw_rows)
    repeated = [
        {
            "sentence": " ".join(tokens),
            "copies": count,
            "splits": ", ".join(
                sorted(
                    {
                        split_of[(row["document_unique_id"], row["sentence_id"])]
                        for row in raw_rows
                        if tuple(row["tokens"]) == tokens
                    }
                )
            ),
        }
        for tokens, count in sentence_counts.most_common()
        if count > 1
    ]
    no_letters = [
        {
            "document": row["document_unique_id"],
            "sentence_id": row["sentence_id"],
            "sentence": " ".join(row["tokens"]),
        }
        for row in raw_rows
        if not any(char.isalpha() for token in row["tokens"] for char in token)
    ]
    mo.vstack(
        [
            mo.md("""
    ## Documents, types, and odd sentences

    The released test file is one guideline, which is why `data_prep` joins both files
    and re-splits them. Repeated sentences and sentences without letters are markdown
    leftovers such as headings and lone full stops; they stay in the data.
    """),
            mo.ui.tabs(
                {
                    "Types": mo.ui.table(pl.DataFrame(type_rows), selection=None, page_size=20),
                    "Documents": mo.ui.table(pl.DataFrame(document_rows), selection=None),
                    f"Stray I tags ({len(stray_rows)})": mo.ui.table(
                        pl.DataFrame(stray_rows), selection=None
                    ),
                    f"Repeated sentences ({len(repeated)})": mo.ui.table(
                        pl.DataFrame(repeated), selection=None
                    ),
                    f"Sentences without letters ({len(no_letters)})": mo.ui.table(
                        pl.DataFrame(no_letters), selection=None
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
        .join(entity_frame.group_by("split").len().rename({"len": "entities"}), on="split")
        .sort("split")
    )
    type_counts = (
        entity_frame.group_by("type", "split")
        .len()
        .pivot(on="split", index="type", values="len")
        .fill_null(0)
        .select("type", "bootstrap", "pool", "test")
        .sort("type")
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
            x=alt.X("type:N", title=None),
            xOffset="split:N",
            y=alt.Y("share:Q", title="Share of the split's entities", axis=alt.Axis(format="%")),
            color=alt.Color("split:N", title="Split"),
            tooltip=["split", "type", alt.Tooltip("len:Q", title="entities")],
        )
        .properties(width=720, height=200, title="Entity types by split")
    )
    length_chart = (
        alt.Chart(sentence_frame)
        .mark_bar()
        .encode(
            x=alt.X("words:Q", bin=alt.Bin(step=4), title="Words per sentence"),
            y=alt.Y("count():Q", title="Sentences"),
            color=alt.Color("split:N", title="Split"),
        )
        .properties(width=420, height=200, title="Sentence lengths")
    )
    mo.vstack(
        [
            mo.md("""
    ## Splits and statistics

    ADR 0008 splits sentences, not documents, so one guideline can sit in both the pool
    and the test split. Rare types can miss the test split entirely.
    """),
            mo.ui.table(split_summary, selection=None),
            mo.hstack([share_chart, length_chart], justify="start", wrap=True),
            mo.ui.table(type_counts, selection=None, page_size=20, label="Entities per split"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(document_rows, mo):
    document_options = {
        f"{row['document_id'][0]} · {row['document'][:8]} ({row['sentences']} sentences)": row[
            "document"
        ]
        for row in sorted(document_rows, key=lambda row: row["document_id"])
    }
    document_selector = mo.ui.dropdown(
        options=document_options, value=next(iter(document_options)), label="Document"
    )
    return (document_selector,)


@app.cell(hide_code=True)
def _(
    data_prep,
    decode_entities,
    document_selector,
    html,
    mo,
    sentences,
    split_of,
    tag_names,
):
    type_names = [tag[2:] for tag in data_prep.MEDICAL_TAGS[1::2]]
    # Step hues by 7/19 of the wheel so neighbouring ids get distant colors.
    entity_colors = {
        name: f"hsla({index * 7 * 360 / len(type_names) % 360:.0f}, 70%, 50%, 0.3)"
        for index, name in enumerate(type_names)
    }

    def render(sentence: dict) -> str:
        pieces = [html.escape(token) for token in sentence["tokens"]]
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
        for s in sorted(sentences, key=lambda s: s["sentence_id"])
        if s["document_name"] == selected
    )
    mo.vstack(
        [
            mo.md("## Document viewer"),
            document_selector,
            mo.Html(f"<div style='line-height:2'>{legend}</div><table>{rows}</table>"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
