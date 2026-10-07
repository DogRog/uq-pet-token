import collections
import dataclasses
import json
import pathlib
import typing


@dataclasses.dataclass
class DocumentBase:
    id: str
    text: str

    def __add__(self, other):
        raise NotImplementedError()

    def copy(self, clear: typing.List[str]):
        raise NotImplementedError()

    def get_hint(self) -> str:
        """Get hint of this document for LLM.

        Returns:
        -------
        str
            Hint as string.
        """
        raise NotImplementedError()


@dataclasses.dataclass
class PetDocument:
    id: str
    text: str
    category: str
    name: str
    tokens: typing.List["PetToken"]
    entities: typing.List["PetEntity"]
    mentions: typing.List["PetMention"]
    relations: typing.List["PetRelation"]

    @property
    def sentences(self) -> typing.List[typing.List["PetToken"]]:
        ret = []
        last_id = None
        for token in self.tokens:
            if token.sentence_index != last_id:
                last_id = token.sentence_index
                ret.append([])
            ret[-1].append(token)
        return ret

    def copy(self, clear: typing.List[str]) -> "PetDocument":
        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=[t.copy() for t in self.tokens],
            mentions=[] if "mentions" in clear else [m.copy() for m in self.mentions],
            relations=(
                [] if "relations" in clear else [r.copy() for r in self.relations]
            ),
            entities=[] if "entities" in clear else [e.copy() for e in self.entities],
        )

    def __add__(self, other: "PetDocument"):
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
            new_entity = PetEntity(mention_indices=tuple(mention_indices))
            if new_entity not in new_entities:
                new_entities.append(new_entity)

        new_relations = self.relations
        for relation in other.relations:
            if relation.head_mention_index not in new_mention_ids:
                continue
            if relation.tail_mention_index not in new_mention_ids:
                continue
            new_relation = PetRelation(
                type=relation.type,
                head_mention_index=new_mention_ids[relation.head_mention_index],
                tail_mention_index=new_mention_ids[relation.tail_mention_index],
            )
            if new_relation not in new_relations:
                new_relations.append(new_relation)

        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=self.tokens,
            mentions=new_mentions,
            entities=new_entities,
            relations=new_relations,
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
        data["relations"] = [relation.to_dict() for relation in self.relations]  # recursive serialization

        return data

    def remove_mention(self, mention_index: int) -> None:
        self.mentions.pop(mention_index)
        new_relations: typing.List[PetRelation] = []
        for i, relation in enumerate(self.relations):
            if relation.head_mention_index == mention_index:
                continue
            if relation.tail_mention_index == mention_index:
                continue

            r = relation
            if relation.head_mention_index > mention_index:
                r = PetRelation(
                    type=r.type,
                    head_mention_index=r.head_mention_index - 1,
                    tail_mention_index=r.tail_mention_index,
                )
            if relation.tail_mention_index > mention_index:
                r = PetRelation(
                    type=r.type,
                    head_mention_index=r.head_mention_index,
                    tail_mention_index=r.tail_mention_index - 1,
                )
            new_relations.append(r)
        self.relations = new_relations

@dataclasses.dataclass(frozen=True)
class PetMention:
    type: str
    token_document_indices: typing.Tuple[int, ...]

    def copy(self) -> "PetMention":
        return PetMention(
            type=self.type.strip().lower(),
            token_document_indices=tuple(i for i in self.token_document_indices),
        )

    def text(self, document: "PetDocument") -> str:
        return " ".join([document.tokens[i].text for i in self.token_document_indices])

    def pretty_dump(self, document: "PetDocument") -> str:
        return f"{self.type}, '{self.text(document)}', {self.token_document_indices}"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PetMention):
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
        if not isinstance(o, PetMention):
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
class PetEntity:
    mention_indices: typing.Tuple[int, ...]

    def copy(self) -> "PetEntity":
        return PetEntity(mention_indices=tuple(i for i in self.mention_indices))

    def get_tag(self, document: "PetDocument") -> str:
        tags = set(document.mentions[i].type for i in self.mention_indices)
        if len(tags) > 1:
            print(f"Entity has mentions of mixed ner tags: {tags}")
        return list(tags)[0]

    def pretty_dump(self, document: PetDocument) -> str:
        formatted_mentions = [
            f"{i}: '{m.text(document)}' ({m.token_document_indices})"
            for i, m in [(i, document.mentions[i]) for i in self.mention_indices]
        ]
        return ", ".join(formatted_mentions)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PetEntity):
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
class PetRelation:
    type: str
    head_mention_index: int
    tail_mention_index: int

    def copy(self) -> "PetRelation":
        return PetRelation(
            head_mention_index=self.head_mention_index,
            tail_mention_index=self.tail_mention_index,
            type=self.type.lower().strip(),
        )

    def pretty_dump(self, document: PetDocument) -> str:
        head = document.mentions[self.head_mention_index].pretty_dump(document)
        tail = document.mentions[self.tail_mention_index].pretty_dump(document)
        return f"{head} -{self.type}-> {tail}"

    def to_dict(self) -> dict[str, typing.Any]:
        """Get document attributes as a dictionary."""
        data = dict()
        # put data in the dict
        data["headMentionIndex"] = self.head_mention_index
        data["tailMentionIndex"] = self.tail_mention_index
        data["type"] = self.type

        return data


@dataclasses.dataclass(frozen=True, eq=True)
class PetToken:
    text: str
    index_in_document: int
    pos_tag: str
    sentence_index: int

    def char_indices(self, document: PetDocument) -> typing.Tuple[int, int]:
        start = 0
        for i, other in enumerate(document.tokens):
            if other == self:
                return start, start + len(self.text)
            start += len(other.text) + 1
        raise AssertionError("Token text not found in document")

    def copy(self) -> "PetToken":
        return PetToken(
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


class PetJsonExporter:
    def __init__(self, path: str):
        self._dict_exporter = PetDictExporter()
        self._path = path

    def export(self, documents: typing.List[PetDocument]):
        json_lines = []
        for document in documents:
            document_as_json = json.dumps(self._dict_exporter.export_document(document))
            json_lines.append(document_as_json)
        with open(self._path, "w", encoding="utf8") as f:
            f.write("\n".join(json_lines))


class PetDictExporter:
    def export_document(self, document: PetDocument) -> typing.Dict:
        return {
            "text": document.text,
            "name": document.name,
            "id": document.id,
            "category": document.category,
            "tokens": list(map(self.export_token, document.tokens)),
            "mentions": list(map(self.export_mention, document.mentions)),
            "entities": list(map(self.export_entity, document.entities)),
            "relations": list(map(self.export_relation, document.relations)),
        }

    def export_token(self, token: PetToken) -> typing.Dict:
        return {
            "text": token.text,
            "indexInDocument": token.index_in_document,
            "posTag": token.pos_tag,
            "sentenceIndex": token.sentence_index,
        }

    def export_mention(self, mention: PetMention) -> typing.Dict:
        return {
            "type": mention.type,
            "tokenDocumentIndices": list(mention.token_document_indices),
        }

    def export_relation(self, relation: PetRelation) -> typing.Dict:
        return {
            "headMentionIndex": relation.head_mention_index,
            "tailMentionIndex": relation.tail_mention_index,
            "type": relation.type,
        }

    def export_entity(self, entity: PetEntity) -> typing.Dict:
        return {"mentionIndices": entity.mention_indices}


class PetImporter:
    class DictImporter:
        @staticmethod
        def read_tokens_from_dict(
                json_tokens: typing.List[typing.Dict],
        ) -> typing.List[PetToken]:
            tokens = []
            for i, json_token in enumerate(json_tokens):
                tokens.append(
                    PetToken(
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
        ) -> typing.List[PetMention]:
            mentions = []
            for json_mention in json_mentions:
                mention = PetImporter.DictImporter.read_mention_from_dict(
                    json_mention
                )
                mentions.append(mention)
            return mentions

        @staticmethod
        def read_entities_from_dict(
                json_entities: typing.List[typing.Dict],
        ) -> typing.List[PetEntity]:
            entities = []
            for json_entity in json_entities:
                entity = PetImporter.DictImporter.read_entity_from_dict(
                    json_entity
                )
                entities.append(entity)
            return entities

        @staticmethod
        def read_mention_from_dict(json_mention: typing.Dict) -> PetMention:
            return PetMention(
                type=json_mention["type"].lower().strip(),
                token_document_indices=tuple(json_mention["tokenDocumentIndices"]),
            )

        @staticmethod
        def read_entity_from_dict(json_entity: typing.Dict) -> PetEntity:
            return PetEntity(json_entity["mentionIndices"])

        @staticmethod
        def read_relations_from_dict(
                json_relations: typing.List[typing.Dict],
        ) -> typing.List[PetRelation]:
            relations = []
            for json_relation in json_relations:
                relations.append(
                    PetImporter.DictImporter.read_relation_from_dict(
                        json_relation
                    )
                )
            return relations

        @staticmethod
        def read_relation_from_dict(relation_dict: typing.Dict) -> PetRelation:
            head_mention_index = relation_dict["headMentionIndex"]
            tail_mention_index = relation_dict["tailMentionIndex"]
            return PetRelation(
                head_mention_index=head_mention_index,
                tail_mention_index=tail_mention_index,
                type=relation_dict["type"].lower().strip(),
            )

    def __init__(self, file_path: typing.Union[str, pathlib.Path]):
        self._file_path = file_path

    def do_import(self) -> typing.List[PetDocument]:
        documents: typing.List[PetDocument] = []
        with open(self._file_path, "r", encoding="utf8") as f:
            for json_line in f:
                json_data = json.loads(json_line)
                documents.append(self.read_document_from_json(json_data))
        return documents

    @staticmethod
    def read_document_from_json(json_data: typing.Dict) -> PetDocument:
        mentions = PetImporter.DictImporter.read_mentions_from_dict(
            json_data["mentions"]
        )
        entities = PetImporter.DictImporter.read_entities_from_dict(
            json_data["entities"]
        )
        relations = PetImporter.DictImporter.read_relations_from_dict(
            json_data["relations"]
        )
        tokens = PetImporter.DictImporter.read_tokens_from_dict(
            json_data["tokens"]
        )
        document = PetDocument(
            name=json_data["name"],
            text=json_data["text"],
            id=json_data["id"],
            category=json_data["category"],
            tokens=tokens,
            mentions=mentions,
            relations=relations,
            entities=entities,
        )
        return document
