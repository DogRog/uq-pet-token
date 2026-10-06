import pathlib

import pytest

from autoannotate import prompts

PROMPT_DIR = pathlib.Path(__file__).parent.parent / "src" / "autoannotate" / "resources" / "prompts"


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


@pytest.mark.parametrize("name", ["annotate-mentions.txt", "annotate-relations.txt"])
def test_annotator_templates_take_definitions_and_example(name):
    template = prompts.Prompt(PROMPT_DIR / name)
    assert template.template_params == {"type_definitions", "example"}
