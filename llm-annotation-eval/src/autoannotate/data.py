import collections
import dataclasses
import json
import os
import pathlib
import typing


@dataclasses.dataclass
class Document:
    id: str
    text: str
    category: str
    name: str
    tokens: typing.List["Token"]
    entities: typing.List["Entity"]
    mentions: typing.List["Mention"]

    @property
    def sentences(self) -> typing.List[typing.List["Token"]]:
        ret = []
        last_id = None
        for token in self.tokens:
            if token.sentence_index != last_id:
                last_id = token.sentence_index
                ret.append([])
            ret[-1].append(token)
        return ret

    def copy(self, clear: typing.List[str]) -> "Document":
        return Document(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=[t.copy() for t in self.tokens],
            mentions=[] if "mentions" in clear else [m.copy() for m in self.mentions],
            entities=[] if "entities" in clear else [e.copy() for e in self.entities],
        )

    def __add__(self, other: "Document"):
        assert self.id == other.id
        assert self.tokens == other.tokens

        new_mentions = self.mentions
        new_mention_ids = {}
        for i, mention in enumerate(other.mentions):
            if mention not in new_mentions:
                new_mention_ids[i] = len(new_mentions)
                new_mentions.append(mention)
            else:
                new_mention_ids[i] = new_mentions.index(mention)

        new_entities = self.entities
        for entity in other.entities:
            mention_indices = [new_mention_ids[i] for i in entity.mention_indices]
            new_entity = Entity(mention_indices=tuple(mention_indices))
            if new_entity not in new_entities:
                new_entities.append(new_entity)

        return Document(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=self.tokens,
            mentions=new_mentions,
            entities=new_entities,
        )

    def to_dict(self) -> dict[str, typing.Any]:
        """Get document attributes as a dictionary."""
        data = dict()
        # write all fields
        data["text"] = self.text
        data["id"] = self.id
        data["name"] = self.name
        data["category"] = self.category
        data["tokens"] = [token.to_dict() for token in self.tokens]  # recursive serialization
        data["mentions"] = [mention.to_dict() for mention in self.mentions]  # recursive serialization
        data["entities"] = [entity.to_dict() for entity in self.entities]  # recursive serialization

        return data


@dataclasses.dataclass(frozen=True)
class Mention:
    type: str
    token_document_indices: typing.Tuple[int, ...]

    def copy(self) -> "Mention":
        return Mention(
            type=self.type.strip().lower(),
            token_document_indices=tuple(i for i in self.token_document_indices),
        )

    def text(self, document: "Document") -> str:
        return " ".join([document.tokens[i].text for i in self.token_document_indices])

    def pretty_dump(self, document: "Document") -> str:
        return f"{self.type}, '{self.text(document)}', {self.token_document_indices}"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Mention):
            return False
        if self.type.lower() != o.type.lower():
            return False
        return sorted(self.token_document_indices) == sorted(o.token_document_indices)

    def __hash__(self) -> int:
        element_counts = collections.Counter(self.token_document_indices)
        cur = hash(frozenset(element_counts.items()))
        cur += hash(self.type.lower())
        return cur

    def match(self, o: object):
        if not isinstance(o, Mention):
            return False
        if self.type.lower() != o.type.lower():
            return False
        if any([i in o.token_document_indices for i in self.token_document_indices]):
            return True
        return False

    def to_dict(self) -> dict[str, typing.Any]:
        """Get document attributes as a dictionary."""
        data = dict()
        data["type"] = self.type
        data["tokenDocumentIndices"] = self.token_document_indices

        return data


@dataclasses.dataclass(frozen=True)
class Entity:
    mention_indices: typing.Tuple[int, ...]

    def copy(self) -> "Entity":
        return Entity(mention_indices=tuple(i for i in self.mention_indices))

    def get_tag(self, document: "Document") -> str:
        tags = set(document.mentions[i].type for i in self.mention_indices)
        if len(tags) > 1:
            print(f"Entity has mentions of mixed ner tags: {tags}")
        return list(tags)[0]

    def pretty_dump(self, document: Document) -> str:
        formatted_mentions = [
            f"{i}: '{m.text(document)}' ({m.token_document_indices})"
            for i, m in [(i, document.mentions[i]) for i in self.mention_indices]
        ]
        return ", ".join(formatted_mentions)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Entity):
            return False
        if len(self.mention_indices) != len(o.mention_indices):
            return False
        return sorted(self.mention_indices) == sorted(o.mention_indices)

    def __hash__(self):
        element_counts = collections.Counter(self.mention_indices)
        return hash(frozenset(element_counts.items()))

    def to_dict(self) -> dict[str, typing.Any]:
        """Get document attributes as a dictionary."""
        return {"mentionIndices": self.mention_indices}


@dataclasses.dataclass(frozen=True, eq=True)
class Token:
    text: str
    index_in_document: int
    pos_tag: str
    sentence_index: int

    def char_indices(self, document: Document) -> typing.Tuple[int, int]:
        start = 0
        for i, other in enumerate(document.tokens):
            if other == self:
                return start, start + len(self.text)
            start += len(other.text) + 1
        raise AssertionError("Token text not found in document")

    def copy(self) -> "Token":
        return Token(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            sentence_index=self.sentence_index,
        )

    def to_dict(self) -> dict[str, typing.Any]:
        """Get token attributes as a dictionary."""
        data: dict[str, typing.Any] = dict()
        # dave data in the dict
        data["text"] = self.text
        data["indexInDocument"] = self.index_in_document
        data["posTag"] = self.pos_tag
        data["sentenceIndex"] = self.sentence_index

        return data


class JsonExporter:
    def __init__(self, path: str):
        self._dict_exporter = DictExporter()
        self._path = path

    def export(self, documents: typing.List[Document]):
        json_lines = []
        for document in documents:
            document_as_json = json.dumps(self._dict_exporter.export_document(document))
            json_lines.append(document_as_json)
        with open(self._path, "w", encoding="utf8") as f:
            f.write("\n".join(json_lines))

    def append(self, document: Document):
        """Appends a single document, e.g., to save each prediction as soon as it is done."""
        path = pathlib.Path(self._path)
        separator = ""
        if path.exists() and path.stat().st_size > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                # files written by export() don't end with a newline
                if f.read(1) != b"\n":
                    separator = "\n"
        with open(path, "a", encoding="utf8") as f:
            f.write(separator + json.dumps(self._dict_exporter.export_document(document)) + "\n")


class DictExporter:
    def export_document(self, document: Document) -> typing.Dict:
        return {
            "text": document.text,
            "name": document.name,
            "id": document.id,
            "category": document.category,
            "tokens": list(map(self.export_token, document.tokens)),
            "mentions": list(map(self.export_mention, document.mentions)),
            "entities": list(map(self.export_entity, document.entities)),
        }

    def export_token(self, token: Token) -> typing.Dict:
        return {
            "text": token.text,
            "indexInDocument": token.index_in_document,
            "posTag": token.pos_tag,
            "sentenceIndex": token.sentence_index,
        }

    def export_mention(self, mention: Mention) -> typing.Dict:
        return {
            "type": mention.type,
            "tokenDocumentIndices": list(mention.token_document_indices),
        }

    def export_entity(self, entity: Entity) -> typing.Dict:
        return {"mentionIndices": entity.mention_indices}


class DictImporter:
    @staticmethod
    def read_tokens_from_dict(
            json_tokens: typing.List[typing.Dict],
    ) -> typing.List[Token]:
        tokens = []
        for i, json_token in enumerate(json_tokens):
            tokens.append(
                Token(
                    text=json_token["text"],
                    pos_tag=json_token["posTag"],
                    index_in_document=i,
                    sentence_index=json_token["sentenceIndex"],
                )
            )
        return tokens

    @staticmethod
    def read_mentions_from_dict(
            json_mentions: typing.List[typing.Dict],
    ) -> typing.List[Mention]:
        mentions = []
        for json_mention in json_mentions:
            mention = DictImporter.read_mention_from_dict(
                json_mention
            )
            mentions.append(mention)
        return mentions

    @staticmethod
    def read_entities_from_dict(
            json_entities: typing.List[typing.Dict],
    ) -> typing.List[Entity]:
        entities = []
        for json_entity in json_entities:
            entity = DictImporter.read_entity_from_dict(
                json_entity
            )
            entities.append(entity)
        return entities

    @staticmethod
    def read_mention_from_dict(json_mention: typing.Dict) -> Mention:
        return Mention(
            type=json_mention["type"].lower().strip(),
            token_document_indices=tuple(json_mention["tokenDocumentIndices"]),
        )

    @staticmethod
    def read_entity_from_dict(json_entity: typing.Dict) -> Entity:
        return Entity(json_entity["mentionIndices"])

    @staticmethod
    def read_document_from_dict(json_data: typing.Dict) -> Document:
        # files from before relation extraction was removed also have "relations"; ignored
        mentions = DictImporter.read_mentions_from_dict(
            json_data["mentions"]
        )
        entities = DictImporter.read_entities_from_dict(
            json_data["entities"]
        )
        tokens = DictImporter.read_tokens_from_dict(
            json_data["tokens"]
        )
        document = Document(
            name=json_data["name"],
            text=json_data["text"],
            id=json_data["id"],
            category=json_data["category"],
            tokens=tokens,
            mentions=mentions,
            entities=entities,
        )
        return document


class JsonImporter:
    def __init__(self, file_path: typing.Union[str, pathlib.Path]):
        self._file_path = file_path
        self._dict_importer = DictImporter()

    def do_import(self) -> typing.List[Document]:
        documents: typing.List[Document] = []
        with open(self._file_path, "r", encoding="utf8") as f:
            for json_line in f:
                if json_line.strip() == "":
                    continue
                json_data = json.loads(json_line)
                documents.append(self._dict_importer.read_document_from_dict(json_data))
        return documents
