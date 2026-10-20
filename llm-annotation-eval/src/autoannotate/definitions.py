import dataclasses


@dataclasses.dataclass
class MentionDefinition:
    type: str
    description: str
    # not used in the prompts
    priority: int = 0

    def for_llm(self):
        return f"**{self.type}**: {self.description}"
