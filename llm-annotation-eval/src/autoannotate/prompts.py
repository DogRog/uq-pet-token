import pathlib
import re
import typing


class Prompt:
    """A template with {{name}} placeholders, read from a file or given as text."""

    def __init__(self, template_path: typing.Optional[pathlib.Path] = None, text: typing.Optional[str] = None):
        assert (template_path is None) != (text is None), "Give either a template file or its text."
        if template_path is not None:
            with open(template_path, "r", encoding="utf8") as f:
                text = f.read()
        self.template = text
        self.template_params = self.find_params(self.template)

    @staticmethod
    def find_params(text: str) -> typing.Set[str]:
        pattern = re.compile(r"\{\{([\w\-]+)}}")
        return set(pattern.findall(text))

    def apply(self, **kwargs: str) -> str:
        missing_params = self.template_params - kwargs.keys()
        assert len(missing_params) == 0, f"Missing template params: {missing_params}"
        ret = self.template
        for name, value in kwargs.items():
            assert name in self.template_params, f"Unknown parameter: {name}"
            ret = ret.replace("{{" + name + "}}", value)
        return ret


    def __call__(self, **kwargs: str) -> str:
        return self.apply(**kwargs)
