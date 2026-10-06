import pathlib

import dotenv
import tqdm

import autoannotate

RES_DIR = pathlib.Path(__file__).parent.parent / "res"

dotenv.load_dotenv()

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


# "openai" or "nhr" (OpenAI-compatible LLM gateway of NHR@FAU), see autoannotate.PROVIDERS
# provider_name = "openai"
# model_name = "gpt-6-sol"

# for "nhr" use a gateway chat model id (LightOnOCR and Qwen3-Embedding are not chat models;
# deepseek-ai/DeepSeek-V4-Flash-0731 is listed but did not respond as of 2026-10-06):
#   moonshotai/Kimi-K3, moonshotai/Kimi-K2.6, deepseek-ai/DeepSeek-V4.1-Flash,
#   zai-org/GLM-5.3, zai-org/GLM-5.3-Flash, zai-org/GLM-5.2-FP8,
#   mistralai/Mistral-Medium-3.5-128B, RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8,
#   GaleneAI/Magistral-Small-2509-FP8-Dynamic, Qwen/Qwen3.6-35B-A3B-FP8, Qwen/Qwen3.8-27B-FP8,
#   RedHatAI/gemma-4-31B-it-FP8-block, google/gemma-4-E4B-it, gpt-oss-120b, XiaomiMiMo/MiMo-V2.6-Flash-RL,
#   Aleph-Alpha/Kolibri-1, ibm-granite/granite-4.1-3b, Microsoft/Phi-4-mini-instruct
provider_name = "nhr"
model_name = "deepseek-ai/DeepSeek-V4.1-Flash"

# None omits the parameter, most open-weight models on the NHR@FAU gateway don't support it
reasoning_effort = "none" if provider_name == "openai" else None
provider = autoannotate.get_provider(provider_name)
client = provider.client()
# set to an int to only annotate the first n documents, e.g. for quick test runs
max_documents = None
documents = autoannotate.JsonImporter(RES_DIR / "gold" / "pet.jsonl").do_import()
documents = documents[:max_documents]
predicted_documents = []
for document in tqdm.tqdm(documents):
    print(f"Predicting mentions of {document.name}...")
    document = document.copy(clear=["mentions", "relations", "entities"])
    mention_annotator = autoannotate.LLMMentionAnnotator(
        client=client,
        language_model=model_name,
        example=mention_format_example,
        mention_definitions=mention_definitions,
        reasoning_effort=reasoning_effort,
        instruction_role=provider.instruction_role,
    )
    md_annotation = mention_annotator.annotate(document)
    document = md_annotation.doc

    print(f"Predicting relations of {document.name}...")
    relation_annotator = autoannotate.LLMRelationsAnnotator(
        client=client,
        language_model=model_name,
        example=relation_format_example,
        type_definitions=relation_definitions,
        reasoning_effort=reasoning_effort,
        instruction_role=provider.instruction_role,
    )
    re_annotation = relation_annotator.annotate(document)
    document = re_annotation.doc

    predicted_documents.append(document)

# gateway model ids contain slashes, e.g. "deepseek-ai/DeepSeek-V4-Flash-0731"
out_dir = RES_DIR / "pred" / model_name.replace("/", "_") / "end-to-end"
print(f'Done. Dumping result to {out_dir / "pet.jsonl"}.')
out_dir.mkdir(parents=True, exist_ok=True)
autoannotate.JsonExporter(out_dir / "pet.jsonl").export(predicted_documents)
