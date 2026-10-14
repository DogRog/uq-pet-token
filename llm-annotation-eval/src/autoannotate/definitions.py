import dataclasses
import typing


@dataclasses.dataclass
class MentionDefinition:
    type: str
    description: str
    # not used in the prompts
    priority: int = 0

    def for_llm(self):
        return f"**{self.type}**: {self.description}"


@dataclasses.dataclass
class RelationDefinition:
    type: str
    description: str
    allowed_sources: typing.Optional[typing.List[MentionDefinition]] = None
    allowed_targets: typing.Optional[typing.List[MentionDefinition]] = None

    def for_llm(self):
        return f"**{self.type}**: {self.description}"
