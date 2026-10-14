import typing

import spacy

from autoannotate import data

nlp = spacy.load("en_core_web_md")


def document_from_string(text: str, doc_id: str) -> data.Document:
    tokens = []
    spacy_doc = nlp(text)
    for i, s in enumerate(spacy_doc.sents):
        for t in s:
            tokens.append(data.Token(
                text=t.text.strip(),
                index_in_document=t.i,
                sentence_index=i,
                pos_tag=t.pos_,
            ))

    return data.Document(
        id=doc_id,
        name=doc_id,
        text=text,
        category="",
        tokens=tokens,
        mentions=[],
        entities=[],
        relations=[],
    )


def ner_to_tag(
        ner: str, attributes: typing.Dict[str, str]
) -> typing.Tuple[str, str]:
    """Converts NER tags to xml style opening and closing tags.

    Examples:
    ------------------
    > ner_to_tag("activity")
    > <activity>, </activity>

    > ner_to_tag("further specification")
    > <further_specification>, </further_specification>

    :param ner: named entity recognition tag
    :param attributes: dictionary of attributes to include in opening tag

    :return: tuple of opening and closing tags
    """
    ner = ner.replace(" ", "_")
    if len(attributes) > 0:
        formatted_attributes = " ".join([f"{k}={v}" for k, v in attributes.items()])
        return f"<{ner} {formatted_attributes}>", f"</{ner}>"
    return f"<{ner}>", f"</{ner}>"


def format_document_text_with_entity_mentions(document: data.Document,
                                              only_types: typing.Optional[typing.List[str]] = None) -> str:
    """Formats the document text with its mentions enclosed in XML-style tags.

    Args:
        document: the PET style document to format
        only_types: only print XML-style tags for the following mention types

    Returns:
        text of the document with entity mentions enclosed in XML-style tags, e.g.,
        ``then the <actor id=0>clerk</actor> <activity id=1>sends</activity> [...]``
    """
    if only_types is not None:
        only_types = [t.lower() for t in only_types]

    token_texts = [t.text for t in document.tokens]
    mentions: typing.List[typing.Tuple[int, data.Mention]] = list(
        enumerate(m.copy() for m in document.mentions)
    )

    # sort so that last mentions in text come first (not by id)
    mentions.sort(key=lambda m: -m[1].token_document_indices[0])

    # insert tags, starting from behind, so we dont have to
    # adjust token indices of mentions...
    for i, mention in mentions:
        if only_types is not None and mention.type.lower() not in only_types:
            continue
        attributes = {"id": str(i)}
        opening_tag, closing_tag = ner_to_tag(mention.type, attributes)
        token_texts.insert(mention.token_document_indices[-1] + 1, closing_tag)
        token_texts.insert(mention.token_document_indices[0], opening_tag)

    return " ".join(token_texts)
