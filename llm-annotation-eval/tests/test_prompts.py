import pathlib

import pytest

from autoannotate import MentionParser, RelationParser, data, prompts
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
def test_annotator_templates_take_definitions_and_example(name):
    template = prompts.Prompt(PROMPT_DIR / name)
    assert template.template_params == {"type_definitions", "example"}


def example_parts(example):
    """The input and output lines of a format example."""
    _, rest = example.split("*Input*", 1)
    given, answer = rest.split("*Output*", 1)
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
    mention_definitions, example = definitions.mention_prompt(available(name))
    given, answer = example_parts(example)
    sentences = [line.split(": ", 1)[1] for line in given]
    parser = MentionParser()

    parsed = parser.parse(make_doc(sentences), "\n".join(answer))

    assert parser.issues == []
    assert len(parsed.mentions) == len(answer)
    types = {d.type.lower() for d in mention_definitions}
    assert {m.type for m in parsed.mentions} <= types


def test_relation_example_parses(make_doc):
    relation_definitions, example = definitions.relation_prompt(datasets.DATASETS["pet"])
    _, answer = example_parts(example)
    # the example's input, with the mentions in the order of their ids
    document = make_doc(
        ["the LLM extracts the text ."],
        mentions=[
            data.Mention("activity", (2,)), data.Mention("actor", (1,)),
            data.Mention("activity data", (3, 4)),
        ],
    )
    parser = RelationParser()

    parsed = parser.parse(document, "\n".join(answer))

    assert parser.issues == []
    assert len(parsed.relations) == len(answer)
    assert {r.type for r in parsed.relations} <= {d.type for d in relation_definitions}


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

    def test_example_shows_the_most_types(self, settings):
        example = definitions.medical_mention_format_example(self.TRAIN, settings)

        assert "Sentence 0: If needed , give insulin and water" in example
        assert "Sentence 1: Observe the patient" in example
        assert "Sentence 2: observe closely" in example
        assert "the patient\tT41\t1" in example and "closely\tT17\t2" in example
