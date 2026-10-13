import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    # Preprocessing checks

    Does `data_prep` turn each dataset's raw files into the right sentences, BIO tags, and
    splits? The notebook runs the real pipeline for PET, Quishpi, and
    MedicalProcessInstruks, then checks its output against the raw files and the protocol
    in the README, ADR 0005, and ADR 0008. CoNLL-2003 ships its own splits and is not
    covered.
    """),
            mo.accordion(
                {
                    "How to read this": mo.md("""
    - **Checks** lists every test of every dataset with its expected and observed value.
      `info` rows describe the data and cannot fail.
    - Every dataset gets the same split checks: split sizes, a partition of the raw
      sentences, a label-free pool, split words and tags equal to the raw ones, and a
      reproducible reload.
    - **PET** ships BIO tag strings per sentence, so its raw tags are compared directly.
    - **Quishpi** spans are read from the `.ann` files separately from
      `brat_sentences`. Each unique span must come back as a tagged entity over the same
      words, or be dropped under a longer span that overlaps it. Every tagged entity must
      come from a span.
    - **MedicalProcessInstruks** gives integer tags without names. `data_prep` reads an
      odd id as the begin tag of a type and the next even id as its inside tag, then names
      the type `T` plus its begin id. The pairing check counts inside tags that do not
      follow their own type under that reading and under the opposite one. The few strays
      left open a sentence: the release cut an entity at a sentence break such as "e.g.".
      The dataset is gated: accept its conditions on the Hub and run
      `uv run hf auth login` once.
    - **Splits and statistics** and the **Document viewer** switch between datasets;
      **Lexical diversity** compares all of them at once.
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
    import json
    from collections import Counter

    import altair as alt
    import marimo as mo
    import polars as pl

    from uq_pet import data_prep
    from utils import lexical_diversity as ld

    return Counter, alt, data_prep, html, json, ld, mo, pl


@app.cell
def _():
    def check(name: str, expected, observed, status: str | None = None) -> dict:
        return {
            "check": name,
            "expected": str(expected),
            "observed": str(observed),
            "status": status or ("pass" if expected == observed else "FAIL"),
        }

    def tag_names(sentence: dict, tags: list[str]) -> list[str]:
        return [tags[tag] for tag in sentence["ner_tags"]]

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

    def stray_inside_tags(tags: list[str]) -> list[int]:
        """Return the indices of I- tags that start a sentence or follow another type."""
        return [
            index
            for index, tag in enumerate(tags)
            if tag.startswith("I-") and (index == 0 or tags[index - 1][2:] != tag[2:])
        ]

    def split_checks(splits, tags, expected_sizes, raw_sentences, reload):
        """Check a dataset's splits against its raw sentences, keyed by (document, sentence id).

        `raw_sentences` maps each key to its words and tag names as read from the raw files.
        Returns the checks, every split sentence with its tags and split, and each key's split.
        """
        seed_examples, pool_inputs, pool_gold, test_examples = splits
        members = {
            "bootstrap": seed_examples,
            "pool": [
                {
                    **{field: p[field] for field in ("document_name", "sentence_id", "tokens")},
                    "ner_tags": [
                        pool_gold.get((p["pool_idx"], i)) for i in range(len(p["tokens"]))
                    ],
                }
                for p in pool_inputs
            ],
            "test": test_examples,
        }
        sentences = [{**s, "split": split} for split, group in members.items() for s in group]
        split_of = {(s["document_name"], s["sentence_id"]): s["split"] for s in sentences}
        differing = sum(
            raw_sentences.get(key)
            != (s["tokens"], [tags[t] if t is not None else None for t in s["ner_tags"]])
            for s in sentences
            for key in [(s["document_name"], s["sentence_id"])]
        )
        checks = [
            check(
                "Bootstrap, pool, and test sentences",
                expected_sizes,
                tuple(len(members[split]) for split in ("bootstrap", "pool", "test")),
            ),
            check(
                "Splits are disjoint and cover every sentence",
                True,
                len(split_of) == len(sentences) and set(split_of) == set(raw_sentences),
            ),
            check("Pool inputs carry no tags", True, all("ner_tags" not in p for p in pool_inputs)),
            check(
                "Pool labels cover every pool word",
                True,
                len(pool_gold) == sum(len(p["tokens"]) for p in pool_inputs),
            ),
            check("Split sentences whose words or tags differ from the raw files", 0, differing),
            check("Loading again gives the same split", True, reload() == splits),
        ]
        return checks, sentences, split_of

    return check, decode_entities, split_checks, stray_inside_tags, tag_names


@app.cell
def _(data_prep, json):
    pet_path = data_prep.download_pet_ner()
    pet_raw = [json.loads(line) for line in pet_path.read_text(encoding="utf-8").splitlines()]
    pet_splits = data_prep.load_pet_splits(pet_path)
    return pet_path, pet_raw, pet_splits


@app.cell
def _(
    Counter,
    check,
    data_prep,
    pet_path,
    pet_raw,
    pet_splits,
    split_checks,
    stray_inside_tags,
):
    _keys = Counter((row["document name"], row["sentence-ID"]) for row in pet_raw)
    _ids_by_document = {}
    for _row in pet_raw:
        _ids_by_document.setdefault(_row["document name"], []).append(_row["sentence-ID"])
    pet_tag_counts = Counter(tag for row in pet_raw for tag in row["ner-tags"])
    _split_checks, pet_sentences, pet_split_of = split_checks(
        pet_splits,
        data_prep.NER_TAGS,
        (data_prep.N_SEED_SENTENCES, 328, 84),
        {
            (row["document name"], row["sentence-ID"]): (row["tokens"], row["ner-tags"])
            for row in pet_raw
        },
        lambda: data_prep.load_pet_splits(pet_path),
    )
    pet_checks = [
        check("Sentences", 417, len(pet_raw)),
        check("Documents", 45, len(_ids_by_document)),
        check("Repeated (document, sentence) keys", 0, sum(n > 1 for n in _keys.values())),
        check(
            "Documents whose sentence ids are not 0, 1, 2, …",
            0,
            sum(sorted(ids) != list(range(len(ids))) for ids in _ids_by_document.values()),
        ),
        check(
            "Sentences with unequal word and tag counts",
            0,
            sum(len(row["tokens"]) != len(row["ner-tags"]) for row in pet_raw),
        ),
        check("Blank words", 0, sum(not w.strip() for row in pet_raw for w in row["tokens"])),
        check("Tags outside NER_TAGS", [], sorted(set(pet_tag_counts) - set(data_prep.NER_TAGS))),
        check(
            "NER_TAGS that never occur",
            "",
            ", ".join(t for t in data_prep.NER_TAGS if t not in pet_tag_counts),
            "info",
        ),
        check(
            "I- tags that start a sentence or follow another type",
            0,
            sum(len(stray_inside_tags(row["ner-tags"])) for row in pet_raw),
        ),
        *_split_checks,
    ]
    pet_document_rows = [
        {
            "document": name,
            "sentences": len(ids),
            "words": sum(len(r["tokens"]) for r in pet_raw if r["document name"] == name),
            "splits": ", ".join(sorted({pet_split_of[(name, i)] for i in ids})),
        }
        for name, ids in _ids_by_document.items()
    ]
    return pet_checks, pet_document_rows, pet_sentences, pet_tag_counts


@app.cell
def _(data_prep):
    quishpi_dir = data_prep.download_quishpi()
    quishpi_documents = {
        name: {
            "text": (quishpi_dir / "texts" / f"{name}.txt").read_text(encoding="utf-8"),
            "annotations": (quishpi_dir / "judgeannotations" / f"{name}.ann").read_text(
                encoding="utf-8"
            ),
        }
        for name in data_prep.QUISHPI_DOCUMENTS
    }
    quishpi_sentences = [
        sentence
        for name, document in quishpi_documents.items()
        for sentence in data_prep.brat_sentences(
            name, document["text"], document["annotations"], data_prep.QUISHPI_TAGS
        )
    ]
    quishpi_splits = data_prep.load_quishpi_splits(quishpi_dir)
    return quishpi_dir, quishpi_documents, quishpi_sentences, quishpi_splits


@app.cell
def _(
    Counter,
    data_prep,
    decode_entities,
    quishpi_documents,
    quishpi_sentences,
    stray_inside_tags,
    tag_names,
):
    span_rows = []
    quishpi_document_rows = []
    word_problems = []
    quishpi_tag_problems = []
    unexplained_entities = []
    for _name, _document in quishpi_documents.items():
        _text = _document["text"]
        _encoded = _text.encode()
        _words = [(m.start(), m.end(), m.group()) for m in data_prep.WORD_PATTERN.finditer(_text)]
        _doc_sentences = [s for s in quishpi_sentences if s["document_name"] == _name]
        _doc_tokens = [token for s in _doc_sentences for token in s["tokens"]]
        if _doc_tokens != [w[2] for w in _words] or "".join(_doc_tokens) != "".join(_text.split()):
            word_problems.append(_name)
        for _sentence in _doc_sentences:
            _tags = tag_names(_sentence, data_prep.QUISHPI_TAGS)
            quishpi_tag_problems += [
                (_name, _sentence["sentence_id"], index, _tags[index])
                for index in stray_inside_tags(_tags)
            ]

        # Word indices run over the whole document, matching `_words`.
        _entities = decode_entities(
            [tag for s in _doc_sentences for tag in tag_names(s, data_prep.QUISHPI_TAGS)]
        )
        _first_span_ids = {}
        _kept_ranges = set()
        for _line in _document["annotations"].splitlines():
            if not _line.startswith("T"):
                continue
            _span_id, _info, _surface = _line.split("\t")
            _entity_type, _byte_start, _byte_end = _info.split(" ")
            _byte_start, _byte_end = int(_byte_start), int(_byte_end)
            _start = len(_encoded[:_byte_start].decode())
            _end = len(_encoded[:_byte_end].decode())
            _covered = [i for i, w in enumerate(_words) if w[0] < _end and w[1] > _start]
            _word_range = (_covered[0], _covered[-1] + 1, _entity_type)
            _overlapping = [e for e in _entities if e[0] < _word_range[1] and e[1] > _word_range[0]]
            if (_start, _end, _entity_type) in _first_span_ids:
                _outcome = "duplicate"
            elif _word_range in _entities:
                _outcome = "kept"
                _kept_ranges.add(_word_range)
            elif _overlapping:
                _outcome = "dropped"
            else:
                _outcome = "missing"
            _first_span_ids.setdefault((_start, _end, _entity_type), _span_id)
            _winner = _overlapping[0] if _outcome == "dropped" else None
            span_rows.append(
                {
                    "document": _name,
                    "span_id": _span_id,
                    "type": _entity_type,
                    "surface": _surface,
                    "outcome": _outcome,
                    "same_as": _first_span_ids[(_start, _end, _entity_type)]
                    if _outcome == "duplicate"
                    else None,
                    "dropped_under": f"{_winner[2]}: "
                    + " ".join(w[2] for w in _words[_winner[0] : _winner[1]])
                    if _winner
                    else None,
                    "winner_is_longer": _words[_winner[1] - 1][1] - _words[_winner[0]][0]
                    >= _end - _start
                    if _winner
                    else None,
                    "matches_at_bytes": _text[_start:_end] == _surface,
                    "matches_as_characters": _text[_byte_start:_byte_end] == _surface,
                    "splits_word": _words[_covered[0]][0] < _start
                    or _words[_covered[-1]][1] > _end,
                }
            )
        unexplained_entities += [
            (_name, " ".join(w[2] for w in _words[e[0] : e[1]]), e[2])
            for e in _entities
            if e not in _kept_ranges
        ]

        _annotation_lines = _document["annotations"].splitlines()
        _type_counts = Counter(
            line.split("\t")[1].split(" ")[0] for line in _annotation_lines if line.startswith("T")
        )
        quishpi_document_rows.append(
            {
                "document": _name,
                "characters": len(_text),
                "non_ascii_characters": sum(not char.isascii() for char in _text),
                "sentences": len(_doc_sentences),
                "words": len(_doc_tokens),
                "Action": _type_counts["Action"],
                "Entity": _type_counts["Entity"],
                "Condition": _type_counts["Condition"],
                "other_span_types": sum(_type_counts.values())
                - _type_counts["Action"]
                - _type_counts["Entity"]
                - _type_counts["Condition"],
                "ignored_lines": sum(not line.startswith("T") for line in _annotation_lines),
            }
        )
    return (
        quishpi_document_rows,
        quishpi_tag_problems,
        span_rows,
        unexplained_entities,
        word_problems,
    )


@app.cell
def _(
    check,
    data_prep,
    quishpi_dir,
    quishpi_document_rows,
    quishpi_sentences,
    quishpi_splits,
    quishpi_tag_problems,
    span_rows,
    split_checks,
    tag_names,
    unexplained_entities,
    word_problems,
):
    _split_checks, quishpi_split_sentences, quishpi_split_of = split_checks(
        quishpi_splits,
        data_prep.QUISHPI_TAGS,
        (data_prep.N_SEED_SENTENCES, 155, 41),
        {
            (s["document_name"], s["sentence_id"]): (
                s["tokens"],
                tag_names(s, data_prep.QUISHPI_TAGS),
            )
            for s in quishpi_sentences
        },
        lambda: data_prep.load_quishpi_splits(quishpi_dir),
    )
    _outcomes = [row["outcome"] for row in span_rows]
    _dropped = [row for row in span_rows if row["outcome"] == "dropped"]
    quishpi_checks = [
        check("Documents", 18, len(quishpi_document_rows)),
        check(
            "Span types other than Action, Entity, Condition",
            0,
            sum(row["other_span_types"] for row in quishpi_document_rows),
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
        check("Span lines", "", len(_outcomes), "info"),
        check("Exact duplicate span lines", "", _outcomes.count("duplicate"), "info"),
        check("Spans kept as an entity over the same words", "", _outcomes.count("kept"), "info"),
        check("Spans dropped under an overlapping span", "", len(_dropped), "info"),
        check(
            "Dropped spans whose winner is shorter",
            0,
            sum(not row["winner_is_longer"] for row in _dropped),
        ),
        check("Spans missing from the tags", 0, _outcomes.count("missing")),
        check("Tagged entities with no span", 0, len(unexplained_entities)),
        check("I- tags that start a sentence or follow another type", 0, len(quishpi_tag_problems)),
        check("Sentences", "", sum(row["sentences"] for row in quishpi_document_rows), "info"),
        *_split_checks,
    ]
    return quishpi_checks, quishpi_split_of, quishpi_split_sentences


@app.cell
def _(data_prep, json):
    medical_paths, medical_error = None, None
    try:
        medical_paths = data_prep.download_medical()
    except RuntimeError as access_failure:
        medical_error = str(access_failure)
    medical_raw = [
        {"file": path.name, **json.loads(line)}
        for path in medical_paths or []
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    medical_splits = (
        data_prep.load_medical_splits(medical_paths) if medical_paths else ([], [], {}, [])
    )
    return medical_error, medical_paths, medical_raw, medical_splits


@app.cell
def _(Counter, medical_raw):
    def raw_type(raw_tag: int) -> str:
        """Name a raw id's type by its begin id, the odd id of its pair."""
        return f"T{raw_tag if raw_tag % 2 else raw_tag - 1:02d}"

    raw_keys = Counter((row["document_unique_id"], row["sentence_id"]) for row in medical_raw)
    ids_by_document = {}
    for _row in medical_raw:
        ids_by_document.setdefault(_row["document_unique_id"], []).append(_row["sentence_id"])

    length_mismatches = sum(len(row["tokens"]) != len(row["ner_tags"]) for row in medical_raw)
    blank_words = sum(not token.strip() for row in medical_raw for token in row["tokens"])
    used_types = sorted({raw_type(tag) for row in medical_raw for tag in row["ner_tags"] if tag})

    stray_rows = []
    opposite_strays = 0
    _raw_by_key = {(row["document_unique_id"], row["sentence_id"]): row for row in medical_raw}
    for _row in medical_raw:
        _key = (_row["document_unique_id"], _row["sentence_id"])
        _tags = _row["ner_tags"]
        for _index, _tag in enumerate(_tags):
            _previous = _tags[_index - 1] if _index else None
            if _tag and _tag % 2 == 0 and _previous not in (_tag, _tag - 1):
                _earlier = _raw_by_key.get((_key[0], _key[1] - 1), {"tokens": []})
                stray_rows.append(
                    {
                        "file": _row["file"],
                        "document": _key[0],
                        "sentence_id": _key[1],
                        "word": _index,
                        "tag": f"I-{raw_type(_tag)}",
                        "after": "sentence start" if _previous is None else str(_previous),
                        "previous_sentence_ends": " ".join(_earlier["tokens"][-6:])
                        if _index == 0
                        else None,
                        "context": " ".join(_row["tokens"][max(_index - 3, 0) : _index + 4]),
                    }
                )
            # Under the opposite reading, even ids begin and the next odd id continues.
            if _tag % 2 == 1 and _previous not in (_tag, _tag + 1):
                opposite_strays += 1
    return (
        blank_words,
        ids_by_document,
        length_mismatches,
        opposite_strays,
        raw_keys,
        raw_type,
        stray_rows,
        used_types,
    )


@app.cell
def _(
    check,
    data_prep,
    medical_error,
    medical_paths,
    medical_raw,
    medical_splits,
    raw_type,
    split_checks,
):
    _expected_tags = {
        (row["document_unique_id"], row["sentence_id"]): (
            row["tokens"],
            [
                "O" if tag == 0 else f"{'B' if tag % 2 else 'I'}-{raw_type(tag)}"
                for tag in row["ner_tags"]
            ],
        )
        for row in medical_raw
    }
    medical_split_checks, medical_sentences, medical_split_of = split_checks(
        medical_splits,
        data_prep.MEDICAL_TAGS,
        (data_prep.N_SEED_SENTENCES, 255, 65),
        _expected_tags,
        lambda: data_prep.load_medical_splits(medical_paths) if medical_paths else medical_splits,
    )
    medical_load_check = check("Files load", "loaded", medical_error or "loaded")
    return medical_load_check, medical_sentences, medical_split_checks, medical_split_of


@app.cell
def _(
    Counter,
    decode_entities,
    medical_raw,
    medical_sentences,
    medical_split_of,
    raw_type,
    tag_names,
    data_prep,
):
    _sentence_by_key = {(s["document_name"], s["sentence_id"]): s for s in medical_sentences}
    medical_document_rows = []
    for _document_name in dict.fromkeys(row["document_unique_id"] for row in medical_raw):
        _rows = [row for row in medical_raw if row["document_unique_id"] == _document_name]
        medical_document_rows.append(
            {
                "document": _document_name,
                "document_id": sorted({row["document_id"] for row in _rows}),
                "file": sorted({row["file"] for row in _rows}),
                "sentences": len(_rows),
                "words": sum(len(row["tokens"]) for row in _rows),
                "tagged_share": round(
                    sum(tag != 0 for row in _rows for tag in row["ner_tags"])
                    / sum(len(row["ner_tags"]) for row in _rows),
                    3,
                ),
                "splits": ", ".join(
                    sorted(
                        {medical_split_of[(_document_name, row["sentence_id"])] for row in _rows}
                    )
                ),
            }
        )

    _entity_counts = Counter()
    _surface_forms = {}
    _raw_tag_counts_by_type = Counter()
    for _release_row in medical_raw:
        _split_sentence = _sentence_by_key[
            (_release_row["document_unique_id"], _release_row["sentence_id"])
        ]
        for _first, _end, _entity_type in decode_entities(
            tag_names(_split_sentence, data_prep.MEDICAL_TAGS)
        ):
            _entity_counts[(_entity_type, _release_row["file"])] += 1
            _surface_forms.setdefault(_entity_type, Counter())[
                " ".join(_split_sentence["tokens"][_first:_end]).lower()
            ] += 1
        _raw_tag_counts_by_type.update(
            (raw_type(tag), "B" if tag % 2 else "I") for tag in _release_row["ner_tags"] if tag
        )
    _files = sorted({row["file"] for row in medical_raw})
    type_rows = [
        {
            "type": entity_type,
            "begin_id": int(entity_type[1:]),
            "B_words": _raw_tag_counts_by_type[(entity_type, "B")],
            "I_words": _raw_tag_counts_by_type[(entity_type, "I")],
            **{
                f"entities_{name.split('_')[0]}": _entity_counts[(entity_type, name)]
                for name in _files
            },
            "common_words": " | ".join(
                f"{form} ({count})" for form, count in _surface_forms[entity_type].most_common(6)
            ),
        }
        for entity_type in sorted(_surface_forms)
    ]
    return medical_document_rows, type_rows


@app.cell
def _(
    blank_words,
    check,
    data_prep,
    ids_by_document,
    length_mismatches,
    medical_document_rows,
    medical_error,
    medical_load_check,
    medical_paths,
    medical_raw,
    medical_split_checks,
    opposite_strays,
    raw_keys,
    stray_rows,
    type_rows,
    used_types,
):
    _files_by_document = {}
    for _raw_row in medical_raw:
        _files_by_document.setdefault(_raw_row["document_unique_id"], set()).add(_raw_row["file"])
    _test_file_types = {row["type"] for row in type_rows if row.get("entities_test")}
    medical_checks = (
        [medical_load_check]
        if medical_error
        else [
            check("Files", list(data_prep.MEDICAL_FILES), [path.name for path in medical_paths]),
            check("Sentences", 325, len(medical_raw)),
            check("Documents", 13, len(medical_document_rows)),
            check(
                "Documents in both released files",
                0,
                sum(len(found) > 1 for found in _files_by_document.values()),
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
            check(
                "Types missing from the released test file",
                "",
                ", ".join(t for t in used_types if t not in _test_file_types),
                "info",
            ),
            *medical_split_checks,
        ]
    )
    return (medical_checks,)


@app.cell(hide_code=True)
def _(medical_checks, mo, pet_checks, pl, quishpi_checks):
    _by_dataset = {
        "PET": pet_checks,
        "Quishpi": quishpi_checks,
        "MedicalProcessInstruks": medical_checks,
    }
    _rows = [{"dataset": dataset, **row} for dataset, rows in _by_dataset.items() for row in rows]
    _failures = [f"{row['dataset']}: {row['check']}" for row in _rows if row["status"] == "FAIL"]
    mo.vstack(
        [
            mo.md("## 1. Checks"),
            mo.callout(
                mo.md(f"**{len(_failures)} failed:** " + "; ".join(_failures))
                if _failures
                else mo.md("**Every check passed** for " + ", ".join(_by_dataset) + "."),
                kind="danger" if _failures else "success",
            ),
            mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=80),
        ]
    )
    return


@app.cell(hide_code=True)
def _(data_prep, mo, pet_document_rows, pet_tag_counts, pl):
    _tags = pl.DataFrame([{"tag": tag, "words": pet_tag_counts[tag]} for tag in data_prep.NER_TAGS])
    mo.vstack(
        [
            mo.md("""
    ## 2. PET

    PET ships one JSON line per sentence with its words and BIO tag strings; `data_prep`
    maps the strings onto `NER_TAGS`. The 80/20 sentence split is pinned by ADR 0005.
    """),
            mo.hstack(
                [
                    mo.ui.table(_tags, selection=None, label="Words per tag", page_size=15),
                    mo.ui.table(pl.DataFrame(pet_document_rows), selection=None, label="Documents"),
                ],
                justify="start",
                widths=[1, 2],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    mo,
    pl,
    quishpi_document_rows,
    quishpi_tag_problems,
    span_rows,
    unexplained_entities,
):
    _spans = pl.DataFrame(span_rows)
    mo.vstack(
        [
            mo.md("""
    ## 3. Quishpi spans

    Quishpi's brat offsets are UTF-8 byte offsets, and where spans overlap the longer one
    keeps the words (ADR 0008).
    """),
            mo.ui.tabs(
                {
                    "Documents": mo.ui.table(pl.DataFrame(quishpi_document_rows), selection=None),
                    "Spans not kept": mo.ui.table(
                        _spans.filter(pl.col("outcome") != "kept").select(
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
                        _spans.filter(~pl.col("matches_as_characters")).select(
                            "document", "span_id", "type", "surface"
                        ),
                        selection=None,
                    ),
                    "All spans": mo.ui.table(_spans, selection=None),
                    "Tag problems": mo.ui.table(
                        pl.DataFrame(
                            [
                                *[
                                    {"document": d, "problem": f"sentence {s}, word {w}: {t}"}
                                    for d, s, w, t in quishpi_tag_problems
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


@app.cell(hide_code=True)
def _(
    Counter,
    medical_document_rows,
    medical_error,
    medical_raw,
    medical_split_of,
    mo,
    pl,
    stray_rows,
    type_rows,
):
    _heading = mo.md("""
    ## 4. MedicalProcessInstruks types

    The released test file is one guideline, which is why `data_prep` joins both files
    and re-splits them (ADR 0008). Repeated sentences and sentences without letters are
    markdown leftovers such as headings and lone full stops; they stay in the data.
    """)
    mo.stop(
        medical_error is not None,
        mo.vstack(
            [
                _heading,
                mo.callout(
                    mo.md(f"**Cannot load MedicalProcessInstruks.** {medical_error}"), "danger"
                ),
            ]
        ),
    )
    _sentence_counts = Counter(tuple(row["tokens"]) for row in medical_raw)
    _repeated = [
        {
            "sentence": " ".join(tokens),
            "copies": count,
            "splits": ", ".join(
                sorted(
                    {
                        medical_split_of[(row["document_unique_id"], row["sentence_id"])]
                        for row in medical_raw
                        if tuple(row["tokens"]) == tokens
                    }
                )
            ),
        }
        for tokens, count in _sentence_counts.most_common()
        if count > 1
    ]
    _no_letters = [
        {
            "document": row["document_unique_id"],
            "sentence_id": row["sentence_id"],
            "sentence": " ".join(row["tokens"]),
        }
        for row in medical_raw
        if not any(char.isalpha() for token in row["tokens"] for char in token)
    ]
    mo.vstack(
        [
            _heading,
            mo.ui.tabs(
                {
                    "Types": mo.ui.table(pl.DataFrame(type_rows), selection=None, page_size=20),
                    "Documents": mo.ui.table(pl.DataFrame(medical_document_rows), selection=None),
                    f"Stray I tags ({len(stray_rows)})": mo.ui.table(
                        pl.DataFrame(stray_rows), selection=None
                    ),
                    f"Repeated sentences ({len(_repeated)})": mo.ui.table(
                        pl.DataFrame(_repeated), selection=None
                    ),
                    f"Sentences without letters ({len(_no_letters)})": mo.ui.table(
                        pl.DataFrame(_no_letters), selection=None
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(
    data_prep,
    medical_document_rows,
    medical_error,
    medical_sentences,
    pet_sentences,
    quishpi_split_sentences,
):
    datasets = {
        "PET": {"tags": data_prep.NER_TAGS, "sentences": pet_sentences, "labels": {}},
        "Quishpi": {
            "tags": data_prep.QUISHPI_TAGS,
            "sentences": quishpi_split_sentences,
            "labels": {},
        },
    }
    if medical_error is None:
        datasets["MedicalProcessInstruks"] = {
            "tags": data_prep.MEDICAL_TAGS,
            "sentences": medical_sentences,
            "labels": {
                row["document"]: f"{row['document_id'][0]} · {row['document'][:8]}"
                for row in medical_document_rows
            },
        }
    return (datasets,)


@app.cell(hide_code=True)
def _(datasets, mo):
    dataset_selector = mo.ui.dropdown(options=list(datasets), value="PET", label="Dataset")
    return (dataset_selector,)


@app.cell(hide_code=True)
def _(alt, dataset_selector, datasets, decode_entities, mo, pl, tag_names):
    _dataset = datasets[dataset_selector.value]
    _sentences = pl.DataFrame(
        [
            {
                "document": s["document_name"],
                "split": s["split"],
                "words": len(s["tokens"]),
                "tagged_words": sum(tag != 0 for tag in s["ner_tags"]),
            }
            for s in _dataset["sentences"]
        ]
    )
    _entities = pl.DataFrame(
        [
            {"split": s["split"], "type": entity_type, "words": end - first}
            for s in _dataset["sentences"]
            for first, end, entity_type in decode_entities(tag_names(s, _dataset["tags"]))
        ]
    )
    _split_summary = (
        _sentences.group_by("split")
        .agg(
            pl.len().alias("sentences"),
            pl.col("words").sum(),
            (pl.col("tagged_words").sum() / pl.col("words").sum()).round(3).alias("tagged_share"),
            pl.col("document").n_unique().alias("documents"),
        )
        .join(_entities.group_by("split").len().rename({"len": "entities"}), on="split")
        .sort("split")
    )
    _type_table = (
        _entities.group_by("type")
        .agg(
            pl.len().alias("entities"),
            pl.col("words").mean().round(2).alias("mean_words"),
            pl.col("words").max().alias("max_words"),
        )
        .join(
            _entities.group_by("type", "split")
            .len()
            .pivot(on="split", index="type", values="len")
            .fill_null(0),
            on="type",
        )
        .sort("type")
    )
    _type_shares = (
        _entities.group_by("split", "type")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("split")).alias("share"))
    )
    _share_chart = (
        alt.Chart(_type_shares)
        .mark_bar()
        .encode(
            x=alt.X("type:N", title=None),
            xOffset="split:N",
            y=alt.Y("share:Q", title="Share of the split's entities", axis=alt.Axis(format="%")),
            color=alt.Color("split:N", title="Split"),
            tooltip=["split", "type", alt.Tooltip("len:Q", title="entities")],
        )
        .properties(
            width=max(300, 40 * _type_table.height), height=200, title="Entity types by split"
        )
    )
    _length_chart = (
        alt.Chart(_sentences)
        .mark_bar()
        .encode(
            x=alt.X("words:Q", bin=alt.Bin(step=4), title="Words per sentence"),
            y=alt.Y("count():Q", title="Sentences"),
            color=alt.Color("split:N", title="Split"),
        )
        .properties(width=420, height=200, title="Sentence lengths")
    )
    _shared = (
        _sentences.group_by("document")
        .agg(pl.col("split").n_unique().alias("splits"))
        .filter(pl.col("splits") > 1)
        .height
    )
    mo.vstack(
        [
            mo.md("""
    ## 5. Splits and statistics

    ADR 0008 splits every dataset by sentence, like PET, not by document, so one text can
    sit in both the pool and the test split. Rare types can miss the test split entirely.
    """),
            dataset_selector,
            mo.ui.table(_split_summary, selection=None),
            mo.md(
                f"{_shared} of {_sentences['document'].n_unique()} documents have sentences "
                "in more than one split."
            ),
            mo.hstack([_share_chart, _length_chart], justify="start", wrap=True),
            mo.ui.table(_type_table, selection=None, page_size=20, label="Entities per type"),
        ]
    )
    return


@app.cell
def _(datasets, ld):
    lexicon = ld.dataset_words(
        {name: (dataset["sentences"], dataset["tags"]) for name, dataset in datasets.items()}
    )
    return (lexicon,)


@app.cell(hide_code=True)
def _(mo):
    ttr_window = mo.ui.slider(
        start=25, stop=250, step=25, value=50, show_value=True, label="Window size (words)"
    )
    return (ttr_window,)


@app.cell(hide_code=True)
def _(alt, datasets, ld, lexicon, mo, pl, ttr_window):
    _order = list(datasets)
    _sentences = ld.sentence_ttr(lexicon)
    _windows = pl.concat(
        [
            ld.window_ttr(lexicon, ttr_window.value).with_columns(words_kept=pl.lit("all words")),
            ld.window_ttr(lexicon, ttr_window.value, entity_only=True).with_columns(
                words_kept=pl.lit("entity words")
            ),
        ]
    )
    _y = alt.Y("dataset:N", title=None, sort=_order)
    _color = alt.Color("dataset:N", sort=_order, legend=None)

    def _box(data, column, title, domain=None):
        return (
            alt.Chart(data)
            .mark_boxplot(size=22, extent=1.5)
            .encode(
                x=alt.X(
                    f"{column}:Q",
                    title=title,
                    scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=False),
                ),
                y=_y,
                color=_color,
            )
            .properties(width=330, height=130)
        )

    _charts = alt.vconcat(
        alt.hconcat(
            _box(_sentences, "ttr", "Per-sentence TTR", [0, 1]),
            _box(_sentences, "words", "Sentence length (words)"),
        ),
        alt.hconcat(
            *[
                _box(
                    _windows.filter(pl.col("words_kept") == kept),
                    "ttr",
                    f"TTR per {ttr_window.value}-word window, {kept}",
                    [0, 1],
                )
                for kept in ("all words", "entity words")
            ]
        ),
    )

    def _window_means(kept, suffix):
        return (
            _windows.filter(pl.col("words_kept") == kept)
            .group_by("dataset")
            .agg(
                pl.col("ttr").mean().alias(f"window_ttr_{suffix}"),
                pl.len().alias(f"windows_{suffix}"),
            )
        )

    _summary = (
        lexicon.group_by("dataset")
        .agg(
            pl.len().alias("words"),
            pl.col("word").n_unique().alias("distinct_words"),
            (100 * pl.col("entity").mean()).alias("entity_words_pct"),
        )
        .join(
            _sentences.group_by("dataset").agg(
                pl.len().alias("sentences"),
                pl.col("words").median().alias("median_sentence_words"),
                pl.col("ttr").mean().alias("sentence_ttr"),
            ),
            on="dataset",
        )
        .join(_window_means("all words", "all"), on="dataset", how="left")
        .join(_window_means("entity words", "entity"), on="dataset", how="left")
        .sort(pl.col("dataset").replace_strict(_order, list(range(len(_order)))))
    )
    _tests = ld.pairwise_window_tests(_windows, _order)

    def _finding(row):
        verb = (
            "does not clearly differ from"
            if row["holm_p"] >= 0.05
            else "is more diverse than"
            if row["ttr"] > row["against_ttr"]
            else "is less diverse than"
        )
        return (
            f"- **{row['dataset']}** ({row['ttr']:.3f}) {verb} **{row['against']}** "
            f"({row['against_ttr']:.3f}); Holm p = {row['holm_p']:.2g}, "
            f"{row['windows']} vs {row['against_windows']} windows"
        )

    _findings = [
        f"**{kept.capitalize()}**, mean TTR per {ttr_window.value}-word window:\n\n"
        + "\n".join(
            _finding(row)
            for row in _tests.filter(pl.col("words_kept") == kept).iter_rows(named=True)
        )
        for kept in ("all words", "entity words")
        if _tests.height and kept in _tests["words_kept"].to_list()
    ]
    mo.vstack(
        [
            mo.md(
                "## 6. Lexical diversity: is any dataset simpler?\n"
                "The **type-token ratio** (TTR) is distinct words divided by all words: "
                "lower means more repetition, a simpler vocabulary to learn. Plain TTR "
                "falls as a text gets longer, so datasets of different sizes are compared "
                "on windows of the same number of words (mean segmental TTR). **Entity "
                "words** keeps only words inside gold entities, the vocabulary the model "
                "has to tag."
            ),
            ttr_window,
            mo.ui.altair_chart(_charts),
            mo.ui.table(
                _summary,
                selection=None,
                format_mapping={
                    "entity_words_pct": "{:.1f}",
                    "sentence_ttr": "{:.3f}",
                    "window_ttr_all": "{:.3f}",
                    "window_ttr_entity": "{:.3f}",
                },
            ),
            mo.callout(mo.md("\n\n".join(_findings)), kind="neutral")
            if _findings
            else mo.md("Too few windows to compare the datasets; shrink the window."),
            mo.accordion(
                {
                    "Pairwise tests": mo.ui.table(
                        _tests,
                        selection=None,
                        format_mapping={
                            "ttr": "{:.3f}",
                            "against_ttr": "{:.3f}",
                            "p_value": "{:.2g}",
                            "holm_p": "{:.2g}",
                        },
                    ),
                    "How to read this": mo.md("""
    - Every split is included (bootstrap sentences, pool, test split), lowercased, with
      punctuation dropped. Windows run over each dataset's text in document order and do
      not overlap; a shorter final window is dropped.
    - Per-sentence TTR is close to 1 for short sentences, so it mostly tracks sentence
      length. MedicalProcessInstruks has the shortest sentences; compare windows instead.
    - Each pair of datasets is compared with a Mann–Whitney test on window TTR; Holm
      adjusts across the three pairs within each word set.
    - Entity windows are few on Quishpi, where only about a quarter of words are inside
      an entity; shrink the window to get more of them.
    - MedicalProcessInstruks has about half PET's words, so about half its windows, and
      comes from clinical guidelines rather than business process descriptions.
    - 13 of Quishpi's 18 texts are also PET documents (ADR 0008), so their running text
      is largely the same; the datasets differ mainly in what is labelled.
    - TTR measures vocabulary repetition, not how hard the tags are to tell apart.
    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(dataset_selector, datasets, mo):
    _dataset = datasets[dataset_selector.value]
    _counts = {}
    for _s in _dataset["sentences"]:
        _counts[_s["document_name"]] = _counts.get(_s["document_name"], 0) + 1
    _options = {
        f"{_dataset['labels'].get(name, name)} ({count} sentences)": name
        for name, count in sorted(
            _counts.items(), key=lambda item: _dataset["labels"].get(item[0], item[0])
        )
    }
    document_selector = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), label="Document"
    )
    return (document_selector,)


@app.cell(hide_code=True)
def _(
    dataset_selector,
    datasets,
    decode_entities,
    document_selector,
    html,
    mo,
    pl,
    quishpi_documents,
    span_rows,
    tag_names,
):
    _dataset = datasets[dataset_selector.value]
    _type_names = list(dict.fromkeys(tag[2:] for tag in _dataset["tags"][1:]))
    # Step hues by the golden angle so neighbouring types get distant colors.
    _colors = {
        name: f"hsla({index * 137.5 % 360:.0f}, 70%, 50%, 0.3)"
        for index, name in enumerate(_type_names)
    }

    def _render(sentence: dict) -> str:
        pieces = [html.escape(token) for token in sentence["tokens"]]
        for first, end, entity_type in reversed(
            decode_entities(tag_names(sentence, _dataset["tags"]))
        ):
            pieces[first:end] = [
                f'<span style="background:{_colors[entity_type]};border-radius:3px;'
                f'padding:1px 3px" title="{entity_type}">{" ".join(pieces[first:end])}'
                f"<sub style='opacity:.7'> {entity_type}</sub></span>"
            ]
        return " ".join(pieces)

    _selected = document_selector.value
    _legend = " ".join(
        f'<span style="background:{color};border-radius:3px;padding:1px 6px">'
        f"{html.escape(name)}</span>"
        for name, color in _colors.items()
    )
    _rows = "".join(
        f"<tr><td style='opacity:.6;padding-right:12px;vertical-align:top'>"
        f"{s['sentence_id']}&nbsp;·&nbsp;{s['split']}</td>"
        f"<td style='padding-bottom:6px;line-height:1.9'>{_render(s)}</td></tr>"
        for s in sorted(_dataset["sentences"], key=lambda s: s["sentence_id"])
        if s["document_name"] == _selected
    )
    _extras = []
    if dataset_selector.value == "Quishpi":
        _not_kept = pl.DataFrame(span_rows).filter(
            (pl.col("document") == _selected) & (pl.col("outcome") != "kept")
        )
        _extras.append(
            mo.accordion(
                {
                    "Raw text": mo.plain_text(quishpi_documents[_selected]["text"]),
                    "Raw annotations": mo.plain_text(quishpi_documents[_selected]["annotations"]),
                    f"Spans not kept ({_not_kept.height})": mo.ui.table(
                        _not_kept.select("span_id", "type", "surface", "outcome", "dropped_under"),
                        selection=None,
                    ),
                }
            )
        )
    mo.vstack(
        [
            mo.md("## 7. Document viewer"),
            mo.hstack([dataset_selector, document_selector], justify="start"),
            mo.Html(f"<div style='line-height:2'>{_legend}</div><table>{_rows}</table>"),
            *_extras,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
