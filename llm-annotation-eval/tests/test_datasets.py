import json

import pytest

from autoannotate import JsonExporter, data
from eval import datasets, test_split
from experiments import definitions


def row(document, sentence_id, words_and_tags):
    """A sentence from "word/tag" pairs, untagged words are O."""
    pairs = [w.split("/") if "/" in w else (w, "O") for w in words_and_tags.split()]
    return {
        "document name": document,
        "sentence-ID": sentence_id,
        "tokens": [w for w, _ in pairs],
        "ner-tags": [t for _, t in pairs],
    }


def write_sentences(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class TestMentionSpans:
    def test_bio_spans(self):
        tags = ["B-Entity", "I-Entity", "O", "B-Action", "B-Action", "I-Action"]
        assert datasets.mention_spans(tags) == [
            (0, 2, "Entity"),
            (3, 4, "Action"),
            (4, 6, "Action"),
        ]

    def test_stray_inside_tags_start_a_mention(self):
        # as seqeval reads them
        tags = ["O", "I-T03", "I-T43", "I-T43", "B-T03", "I-T43"]
        assert datasets.mention_spans(tags) == [
            (1, 2, "T03"),
            (2, 4, "T43"),
            (4, 5, "T03"),
            (5, 6, "T43"),
        ]


class TestDocumentsFromSentences:
    def test_sentences_become_documents_with_their_mentions(self):
        rows = [
            row("a", 0, "the clerk checks/B-Action the invoice/B-Entity ."),
            row("b", 0, "it ends/B-Action"),
            row("a", 1, "otherwise/B-Condition it/B-Entity is returned/B-Action"),
        ]

        a, b = datasets.documents_from_sentences(rows)

        assert (a.id, b.id) == ("a", "b")
        assert [[t.text for t in s] for s in a.sentences] == [
            ["the", "clerk", "checks", "the", "invoice", "."],
            ["otherwise", "it", "is", "returned"],
        ]
        assert [t.index_in_document for t in a.tokens] == list(range(10))
        assert a.mentions == [
            data.Mention("action", (2,)),
            data.Mention("entity", (4,)),
            data.Mention("condition", (6,)),
            data.Mention("entity", (7,)),
            data.Mention("action", (9,)),
        ]
        assert b.mentions == [data.Mention("action", (1,))]

    def test_sentences_out_of_order_are_refused(self):
        with pytest.raises(ValueError, match="not numbered"):
            datasets.documents_from_sentences([row("a", 1, "x"), row("a", 0, "y")])

    def test_read_documents_reads_both_formats(self, tmp_path, make_doc):
        sentences_file = write_sentences(tmp_path / "s.jsonl", [row("a", 0, "it ends/B-Action")])
        documents_file = tmp_path / "d.jsonl"
        JsonExporter(str(documents_file)).export(
            [make_doc(["it ends"], [data.Mention("activity", (1,))], doc_id="a")]
        )

        [from_sentences] = datasets.read_documents(sentences_file)
        [document] = datasets.read_documents(documents_file)

        assert from_sentences.mentions == [data.Mention("action", (1,))]
        assert document.mentions == [data.Mention("activity", (1,))]


def test_train_leaves_out_the_test_split(tmp_path):
    rows = [row("a", 0, "x"), row("a", 1, "y"), row("b", 0, "z")]
    dataset = datasets.Dataset(
        name="d",
        types=("Action",),
        sentences_file=write_sentences(tmp_path / "d.jsonl", rows),
        test_file=write_sentences(tmp_path / "d-test.jsonl", [rows[1]]),
    )
    assert dataset.train() == [rows[0], rows[2]]
    assert dataset.input_file("md") == dataset.sentences_file
    assert dataset.input_file("md-test-sentences") == dataset.test_file


@pytest.mark.parametrize(
    "name, sentences, test_sentences",
    [
        ("pet", 417, 84),
        ("quishpi", 201, 41),
        ("medical", 325, 65),
    ],
)
def test_dataset_files(name, sentences, test_sentences):
    dataset = datasets.DATASETS[name]
    if not dataset.sentences_file.exists():
        pytest.skip(
            f"{dataset.sentences_file.name} is not generated, see scripts/uq-pet-token-splits.py"
        )
    everything = test_split.load_test_split(dataset.sentences_file)
    test = test_split.load_test_split(dataset.test_file)
    assert (len(everything), len(test)) == (sentences, test_sentences)
    by_key = {(r["document name"], r["sentence-ID"]): r for r in everything}
    assert len(by_key) == sentences
    assert all(by_key[(r["document name"], r["sentence-ID"])] == r for r in test)
    # every tag is one of the dataset's types
    assert {t.split("-", 1)[1] for r in everything for t in r["ner-tags"] if t != "O"} <= set(
        dataset.types
    )
    # the gold mentions of whole documents are the tagged ones, in the same words
    documents = datasets.documents_from_sentences(everything)
    predicted, dropped = test_split.predict_test_split(everything, documents, dataset.types)
    assert dropped == 0
    assert test_split.evaluate(everything, predicted)["entity_f1"] == 1.0


def test_quishpi_prompt():
    mention_definitions, example = definitions.mention_prompt(datasets.DATASETS["quishpi"])
    assert [d.type for d in mention_definitions] == ["Action", "Entity", "Condition"]
    assert "otherwise\tCondition\t2" in example


class TestMedicalPrompt:
    TRAIN = [
        row("a", 0, "Observe/B-T03 the/B-T41 patient/I-T41"),
        row("a", 1, "If/B-T37 needed/B-T09 ,/O give/B-T03 insulin/B-T43 and/B-T23 water/B-T43"),
        row("b", 0, "observe/B-T03 closely/B-T17"),
    ]

    def test_types_are_described_by_their_training_texts(self):
        by_type = {
            d.type: d.description for d in definitions.medical_mention_definitions(self.TRAIN)
        }

        assert list(by_type) == list(datasets.DATASETS["medical"].types)
        assert by_type["T03"].endswith(
            'the 3 mentions in the training sentences: "Observe", "give", "observe"'
        )
        assert by_type["T41"].endswith('the 1 mention in the training sentences: "the patient"')
        assert "does not occur" in by_type["T05"]

    def test_example_shows_the_most_types(self):
        example = definitions.medical_mention_format_example(self.TRAIN)

        assert "Sentence 0: If needed , give insulin and water" in example
        assert "Sentence 1: Observe the patient" in example
        assert "Sentence 2: observe closely" in example
        assert "the patient\tT41\t1" in example and "closely\tT17\t2" in example
