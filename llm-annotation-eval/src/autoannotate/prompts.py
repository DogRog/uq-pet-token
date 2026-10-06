import pathlib
import re
import typing


class Prompt:
    def __init__(self, template_path: pathlib.Path):
        with open(template_path, "r", encoding="utf8") as f:
            self.template = f.read()
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
