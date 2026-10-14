import pathlib

import pytest

from autoannotate import LLMMentionAnnotator, MentionParser, RelationParser, prompts, util
from eval import datasets
from experiments import definitions

PROMPT_DIR = pathlib.Path(__file__).parent.parent / "prompts"


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "template.txt"
    path.write_text("Types:\n{{type_definitions}}\n\n{{example}}\nend {{example}}")
    return prompts.Prompt(path)


def test_finds_params(template):
    assert template.template_params == {"type_definitions", "example"}


def test_replaces_every_occurrence(template):
    assert template(type_definitions="A, B", example="ex") == "Types:\nA, B\n\nex\nend ex"


def test_missing_param_fails(template):
    with pytest.raises(AssertionError, match="Missing"):
        template(type_definitions="A")


def test_unknown_param_fails(template):
    with pytest.raises(AssertionError, match="Unknown"):
        template(type_definitions="A", example="", typo="x")


def test_template_from_text():
    assert prompts.Prompt(text="{{a}} and {{b}}")(a="x", b="y") == "x and y"


@pytest.mark.parametrize("name", ["mentions.txt", "relations.txt"])
def test_annotator_templates_take_definitions_and_the_datasets_values(name):
    template = prompts.Prompt(PROMPT_DIR / name)
    assert template.template_params == {"type_definitions", *definitions.TEMPLATE_VALUES}


def example_parts(example):
    """The input and output lines of a format example."""
    _, rest = example.split("### Input Example", 1)
    given, answer = rest.split("### Output Example", 1)
    given = [line for line in given.splitlines()[1:] if line.strip()]
    answer = [line for line in answer.splitlines()[1:] if line.strip()]
    return given, answer


def available(name):
    dataset = datasets.DATASETS[name]
    if not dataset.sentences_file.exists():
        pytest.skip(f"{dataset.sentences_file.name} is missing, see scripts/uq-pet-token-splits.py")
    return dataset


@pytest.mark.parametrize("name", list(datasets.DATASETS))
def test_mention_types_are_the_datasets_types(name):
    mention_definitions, _ = definitions.mention_prompt(available(name))
    types = [d.type for d in mention_definitions]
    # the predicted types must be spelled like the dataset's tags, in any order
    assert len(types) == len(set(types)) and set(types) == set(datasets.DATASETS[name].types)
    assert all(d.description and "  " not in d.description for d in mention_definitions)


@pytest.mark.parametrize("name", list(datasets.DATASETS))
def test_mention_example_parses(name, make_doc):
    """The example's answer is in the format the parser reads, e.g. tabs were not replaced."""
    mention_definitions, values = definitions.mention_prompt(available(name))
    given, answer = example_parts(values["example"])
    sentences = [line.split(": ", 1)[1] for line in given]
    parser = MentionParser()

    parsed = parser.parse(make_doc(sentences), "\n".join(answer))

    assert parser.issues == []
    assert len(parsed.mentions) == len(answer)
    types = {d.type.lower() for d in mention_definitions}
    assert {m.type for m in parsed.mentions} <= types


@pytest.mark.parametrize("name", list(datasets.DATASETS))
def test_mention_prompt_fills_the_template(name):
    mention_definitions, values = definitions.mention_prompt(available(name))
    template = prompts.Prompt(definitions.mention_template())

    instruction = template(type_definitions="", **values)

    assert "{{" not in instruction and all(values.values())


def test_example_input_is_formatted_like_the_documents(make_doc):
    """The format example shows the sentences as the annotator sends them."""
    _, values = definitions.mention_prompt(datasets.DATASETS["pet"])
    given, _ = example_parts(values["example"])
    sentences = [line.split(": ", 1)[1] for line in given]

    assert "\n".join(given) == LLMMentionAnnotator._format(make_doc(sentences))


@pytest.mark.parametrize("name", list(datasets.DATASETS))
def test_mention_example_shows_the_bootstrap_sentences(name):
    """The example is uq-pet-token's bootstrap sentences with all their mentions."""
    dataset = available(name)
    _, values = definitions.mention_prompt(dataset)
    given, answer = example_parts(values["example"])
    bootstrap = dataset.bootstrap()

    assert given == [f"{i}: {' '.join(r['tokens'])}" for i, r in enumerate(bootstrap)]
    assert len(answer) == sum(len(datasets.mention_spans(r["ner-tags"])) for r in bootstrap)


def test_mention_example_lists_mentions_in_text_order():
    sentences = [
        {"document name": "a", "sentence-ID": 3, "tokens": ["the", "clerk", "checks", "it"],
         "ner-tags": ["B-Actor", "I-Actor", "B-Activity", "O"]},
        {"document name": "b", "sentence-ID": 0, "tokens": ["nothing", "here"],
         "ner-tags": ["O", "O"]},
        {"document name": "c", "sentence-ID": 1, "tokens": ["then", "pay"],
         "ner-tags": ["O", "B-Activity"]},
    ]

    example = definitions.mention_example(sentences, "{{sentences}}\n--\n{{mentions}}")

    assert example == (
        "0: the clerk checks it\n1: nothing here\n2: then pay\n--\n"
        "the clerk\tActor\t0\nchecks\tActivity\t0\npay\tActivity\t2"
    )


def test_relation_example_parses():
    dataset = datasets.DATASETS["pet"]
    relation_definitions, values = definitions.relation_prompt(dataset)
    given, answer = example_parts(values["example"])
    document = definitions.relation_example_document(dataset)
    parser = RelationParser()

    parsed = parser.parse(document.copy(clear=["relations"]), "\n".join(answer))

    assert given == [util.format_document_text_with_entity_mentions(document)]
    assert parser.issues == []
    assert parsed.relations == document.relations
    assert {r.type for r in parsed.relations} <= {d.type for d in relation_definitions}


def test_relation_example_is_the_bootstrap_sentences_with_their_gold():
    """The mentions in the bootstrap sentences, in text order, and the relations between them."""
    dataset = datasets.DATASETS["pet"]
    document = definitions.relation_example_document(dataset)
    bootstrap = dataset.bootstrap()

    assert [[t.text for t in s] for s in document.sentences] == [r["tokens"] for r in bootstrap]
    starts = [m.token_document_indices[0] for m in document.mentions]
    assert starts == sorted(starts)
    # PET v1.1 tags the bootstrap sentences like res/gold/pet.jsonl
    first = 0
    spans = []
    for r in bootstrap:
        spans += [
            (first + b, first + e, t.lower()) for b, e, t in datasets.mention_spans(r["ner-tags"])
        ]
        first += len(r["tokens"])
    assert [
        (m.token_document_indices[0], m.token_document_indices[-1] + 1, m.type)
        for m in document.mentions
    ] == spans
    assert {r.type for r in document.relations} >= {"uses", "flow", "actor performer"}


def test_only_pet_has_relations():
    with pytest.raises(ValueError, match="no relations"):
        definitions.relation_prompt(datasets.DATASETS["quishpi"])


class TestMedicalPrompt:
    TRAIN = [
        {"document name": d, "sentence-ID": i, "tokens": [w.split("/")[0] for w in text.split()],
         "ner-tags": [w.split("/")[1] if "/" in w else "O" for w in text.split()]}
        for d, i, text in [
            ("a", 0, "Observe/B-T03 the/B-T41 patient/I-T41"),
            ("a", 1, "If/B-T37 needed/B-T09 , give/B-T03 insulin/B-T43 and/B-T23 water/B-T43"),
            ("b", 0, "observe/B-T03 closely/B-T17"),
        ]
    ]

    @pytest.fixture
    def settings(self):
        return definitions._read("medical")["mentions"]

    def test_types_are_described_by_their_training_texts(self, settings):
        by_type = {
            d.type: d.description
            for d in definitions.medical_mention_definitions(self.TRAIN, settings)
        }

        assert list(by_type) == list(datasets.DATASETS["medical"].types)
        assert by_type["T03"].endswith(
            'the 3 mentions in the training sentences: "Observe", "give", "observe"'
        )
        assert by_type["T41"].endswith('the 1 mention in the training sentences: "the patient"')
        assert "does not occur" in by_type["T05"]
