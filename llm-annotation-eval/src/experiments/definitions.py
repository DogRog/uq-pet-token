"""Mention and relation definitions and the few-shot format examples of each dataset.

PET's are written by hand, as are Quishpi's, after Quishpi et al.'s annotations: mostly single
verbs as actions, the objects they act on as entities (not who performs them), and condition
clauses. MedicalProcessInstruks names none of its types, so its definitions only list each
type's most frequent texts in the training sentences, and its example is training sentences too;
the test split never appears in a prompt.

A run's manifest (run.json) records the prompts rendered from these, and resuming a run
refuses to continue if they changed in between.
"""
import collections
import typing

import autoannotate
from eval import datasets

mention_definitions = [
    autoannotate.MentionDefinition(
        type="Activity",
        description="""an active task or action executed during the business process, e.g.,
            "examine", "write", "bake", "review". Do not extract, if it is information about
            the process itself, such as "the process ends", as this is not a task in the process!
            Can also be an event during the process, that is executed by an external, implicit actor,
            that is not mentioned in the text, e.g., "placed" in "when an order is placed".
            Never contains the Actor that executes it, nor the Activity Data that's used during this
            Activity, is just the verb as in "checked" and not "is checked"!""",
        priority=10
    ),
    autoannotate.MentionDefinition(
        type="Actor",
        description="""a person, department, or similar role that participates actively in the business
         process, e.g., "the student", "the professor", "the judge", "the clerk". It should
         only be extracted, if the current sentence describes the actor executing a task.
         Include the determiner if it is present, e.g. extract "the student" from "First the
         student studies for their exam". Can also be a pronoun, such as "he", "I", "she".""",
        priority=5
    ),
    autoannotate.MentionDefinition(
        type="Further Specification",
        description="""important information about an activity, such as the mean, the manner
                         of execution, or how an activity is executed. Follows an Activity in the
                         same sentence, and describes how an Activity (task) is being done.""",
        priority=1
    ),
    autoannotate.MentionDefinition(
        type="XOR Gateway",
        description="""textual representation of a decision in the process, usually triggered by adverbs or
               conjunctions, e.g., "if", "otherwise", "when" """,
        priority=6
    ),
    autoannotate.MentionDefinition(
        type="Activity Data",
        description="""a physical object, or digital data that is relevant to the process, because
                 an action produces or uses it, e.g., "the paper", "the report", "the machine part".
                 Always include the determiner! Can also be a pronoun, like "it", but is always part of
                 a task description. Is never information about the process itself, as in "the process
                 ends", here "process" is not Activity Data!""",
        priority=5
    ),
    autoannotate.MentionDefinition(
        type="AND Gateway",
        description="""textual description of parallel work streams in the process, i.e., simultaneous
               actions performed, marked by phrases like, e.g., "while", "meanwhile", "at the same time" """,
        priority=6
    ),
    autoannotate.MentionDefinition(
        type="Condition Specification",
        description="""defines the condition of an XOR Gateway path usually directly following the
                           gateway trigger word, e.g., "it is ready", "the claim is valid",
                           "temperature is above 180" """,
        priority=4
    ),
]

mention_format_example = """
**Examples for the format given the input text:**

*Input*:

Sentence 0: the professor grades all student papers .
Sentence 1: some time passes .
Sentence 2: then the professor returns the papers .

*Output*:

the professor	Actor	0
grades	Activity	0
all student papers	Activity Data	0
the professor	Actor	2
returns	Activity	2
the papers	Activity Data	2
"""

relation_definitions = [
    autoannotate.RelationDefinition(
        type="uses",
        description="""Focus on mentions of type "activity". Find the "activity_data" mention, that is being used or created during the execution of this activity. If you find one, extract a relation of type "uses". The source mention is the activity, the target mention is the "activity_data" mention."""
    ),
    autoannotate.RelationDefinition(
        type="flow",
        description="""Focus on mentions of types "Activity", "XOR_Gateway", "AND_Gateway", and "Condition_Specification". If a condition_specification follows an XOR_Gateway connect the XOR_Gateway (source) to the Condition_Specification (target). If an Activity, XOR_Gateway, or AND_Gateway follows an Activity, connect the one appearing earlier (head) with the latter one (target). If an Activity, XOR_Gateway, AND_Gateway, or Condition_Specification follows an Activity, XOR_Gateway, AND_Gateway, or Condition Specification, connect the earlier one (source) to the latter one (target)."""
    ),
    autoannotate.RelationDefinition(
        type="actor performer",
        description="""Focus on mentions of type "activity" and look for the "actor", that actively performs the activity (not is passively affected by it!). If you find one, extract a relation of type "actor performer". The source mention is the "activity", and the target mention is the "actor"."""
    ),
    autoannotate.RelationDefinition(
        type="actor recipient",
        description="""Focus on mentions of type "activity" and look for the "actor", that receives the result of some activity execution, i.e., they are passively affected by the activity. If you find one, extract a relation of type "actor recipient". The source mention is the "Activity", the target mention is the "Actor"."""
    ),
    autoannotate.RelationDefinition(
        type="same gateway",
        description="""Textual descriptions of gateways are often spread over several phrases, especially the different outgoing paths, this relation links these disjointed descriptions together. Look for elements of type "xor_gateway" and "and_gateway". If you find another mention of the same type, check if they describe the same decision point in the process execution. If they do, extract a relation of type "same gateway", with the source mention being the one that appears earlier in the text, and the target mention being the one that appears later."""
    ),
    autoannotate.RelationDefinition(
        type="further specification",
        description="""Focus on mentions of type "activity". Try and find mentions of type "further_specification", that describe how this activity has to be executed, or describe the activity further. If you find one, extract a relation of type "further_specification", where the source mention is the activity and the target mention is the mention of type "further_specification"."""
    ),
]
relation_format_example = """
**Examples for the format given the input text:**

*Input*:

the <actor id=1> LLM </actor> <activity id=0> extracts </activity> <activity_data id=2> the text </activity> .

*Output*

actor performer	0	1
uses	0	2
"""


quishpi_mention_definitions = [
    autoannotate.MentionDefinition(
        type="Action",
        description="""an activity or event of the process, usually only its verb, e.g. "checks",
            "sends", or "informed" in "the customer is informed", without auxiliaries, the actor
            or the object. Verbs joined by "and" that share their object can form one mention,
            e.g. "checks and configures".""",
        priority=10
    ),
    autoannotate.MentionDefinition(
        type="Entity",
        description="""the business object an Action is performed on, i.e. what is created,
            checked, sent or changed, e.g. "invoice", "detailed problem report", "copy of the
            invoice". Usually without the determiner. Who performs an Action, e.g. a role,
            person or department like "the clerk", is not an Entity.""",
        priority=5
    ),
    autoannotate.MentionDefinition(
        type="Condition",
        description="""the condition under which a part of the process is executed, usually
            the clause after "if", "when" or "in case", without that word, e.g. "the part is
            available in-house" in "if the part is available in-house"; also the word that
            starts the alternative path, e.g. "otherwise".""",
        priority=5
    ),
]

quishpi_mention_format_example = """
**Examples for the format given the input text:**

*Input*:

Sentence 0: the clerk checks the invoice .
Sentence 1: if the invoice is correct , the clerk pays it .
Sentence 2: otherwise the invoice is returned .

*Output*:

checks\tAction\t0
invoice\tEntity\t0
the invoice is correct\tCondition\t1
pays\tAction\t1
otherwise\tCondition\t2
invoice\tEntity\t2
returned\tAction\t2
"""

# texts listed per medical type, sentences in its example and the most tokens they may have
MEDICAL_TEXTS_PER_TYPE = 10
MEDICAL_EXAMPLE_SENTENCES = 3
MEDICAL_EXAMPLE_MAX_TOKENS = 30


def _mentions(row: datasets.test_split.Row) -> typing.List[typing.Tuple[str, str]]:
    """The (text, type) of the tagged mentions of a sentence."""
    return [
        (" ".join(row["tokens"][start:end]), tag_type)
        for start, end, tag_type in datasets.mention_spans(row["ner-tags"])
    ]


def medical_mention_definitions(
        train: typing.List[datasets.test_split.Row]
) -> typing.List[autoannotate.MentionDefinition]:
    """Describes each unnamed medical type by its most frequent texts in the training sentences."""
    texts: typing.Dict[str, typing.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in train:
        for text, tag_type in _mentions(row):
            texts[tag_type][text] += 1
    definitions = []
    for tag_type in datasets.DATASETS["medical"].types:
        counts = texts[tag_type]
        if counts:
            most_common = counts.most_common(MEDICAL_TEXTS_PER_TYPE)
            listed = ", ".join(f'"{text}"' for text, _ in most_common)
            total = sum(counts.values())
            description = (
                f"a type the dataset does not name. Its most frequent texts among the {total} "
                f"mention{'s' if total > 1 else ''} in the training sentences: {listed}"
            )
        else:
            description = (
                "a type the dataset does not name. It does not occur in the training sentences."
            )
        definitions.append(
            autoannotate.MentionDefinition(type=tag_type, description=description, priority=1)
        )
    return definitions


def medical_mention_format_example(train: typing.List[datasets.test_split.Row]) -> str:
    """Training sentences that together show the most types, as the format example."""
    candidates = [r for r in train if len(r["tokens"]) <= MEDICAL_EXAMPLE_MAX_TOKENS]
    chosen: typing.List[datasets.test_split.Row] = []
    shown: typing.Set[str] = set()
    for _ in range(MEDICAL_EXAMPLE_SENTENCES):
        # the first sentence with the most types not shown yet
        best = max(candidates, key=lambda r: len({t for _, t in _mentions(r)} - shown))
        chosen.append(best)
        candidates.remove(best)
        shown |= {t for _, t in _mentions(best)}
    sentences = "\n".join(f"Sentence {i}: {' '.join(r['tokens'])}" for i, r in enumerate(chosen))
    mentions = "\n".join(
        f"{text}\t{tag_type}\t{i}" for i, r in enumerate(chosen) for text, tag_type in _mentions(r)
    )
    return f"""
**Examples for the format given the input text:**

*Input*:

{sentences}

*Output*:

{mentions}
"""


def mention_prompt(
        dataset: datasets.Dataset
) -> typing.Tuple[typing.List[autoannotate.MentionDefinition], str]:
    """The mention definitions and format example of a dataset."""
    if dataset.name == "pet":
        return mention_definitions, mention_format_example
    if dataset.name == "quishpi":
        return quishpi_mention_definitions, quishpi_mention_format_example
    if dataset.name == "medical":
        train = dataset.train()
        return medical_mention_definitions(train), medical_mention_format_example(train)
    raise ValueError(f"No mention definitions for dataset {dataset.name!r}.")
