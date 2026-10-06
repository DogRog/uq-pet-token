import pathlib

import numpy as np

from autoannotate.data import JsonImporter
from eval.metrics import Stats

RES_DIR = pathlib.Path(__file__).parent.parent / "res"

def load_ner_metrics(result_dir: pathlib.Path, original_file: pathlib.Path):
    ground_truth_importer = JsonImporter(original_file)
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }

    # create importers and import documents
    predictions_files = [f for f in result_dir.iterdir() if f.suffix == '.jsonl']
    assert len(predictions_files) == 1, f"No prediction in {result_dir}"
    prediction_importer = JsonImporter(predictions_files[0])
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    ner_stats = []
    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        if doc_id not in prediction_documents_dict:
            print(f"Skipping document {doc_id}, as we did not predict that yet.")
            continue

        predicted_doc = prediction_documents_dict[doc_id]

        original_entities = ground_truth_doc.mentions
        original_entities_buffer = original_entities.copy()

        predicted_entities = prediction_documents_dict[doc_id].mentions

        # now collect relation hashes
        matches: int = 0
        for p_entity in predicted_entities:
            p_tokens = set(
                predicted_doc.tokens[i]
                for i
                in p_entity.token_document_indices
            )

            match_found: bool = False
            for idx, t_entity in enumerate(original_entities_buffer):
                t_tokens = set(
                    ground_truth_doc.tokens[i]
                    for i
                    in t_entity.token_document_indices
                )

                if t_entity.type != p_entity.type:
                    # no type match
                    continue

                if not t_tokens.intersection(p_tokens):
                    # head argument is different
                    continue

                # got a match
                match_found = True
                break

            # check match and remove relation from the buffer
            if match_found:
                matches += 1
                original_entities_buffer.remove(t_entity)

        # how we can check the intersection of these two sets and find the results
        ner_stats.append(Stats(num_pred=len(predicted_entities), num_gold=len(original_entities), num_ok=matches))

    return ner_stats


def load_re_metrics(result_dir: pathlib.Path, original_file: pathlib.Path):
    ground_truth_importer = JsonImporter(original_file)
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }

    # create importers and import documents
    prediction_importer = JsonImporter(result_dir / "pet.jsonl")
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    re_stats = []
    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        if doc_id not in prediction_documents_dict:
            # print(f"Skipping document {doc_id}, as we did not predict that yet.")
            continue

        predicted_doc = prediction_documents_dict[doc_id]

        # get original relations
        original_relations = ground_truth_doc.relations
        original_relations_buffer = original_relations.copy()

        # predicted releations
        predicted_relations = prediction_documents_dict[doc_id].relations

        # now collect relation hashes
        matches: int = 0
        for relation in predicted_relations:
            p_head_tokens = set(predicted_doc.tokens[i]
                                for i
                                in predicted_doc.mentions[relation.head_mention_index].token_document_indices)
            p_tail_tokens = set(predicted_doc.tokens[i]
                                for i
                                in predicted_doc.mentions[relation.tail_mention_index].token_document_indices)

            match_found: bool = False
            # look for relation in
            for idx, original_relation in enumerate(original_relations_buffer):
                t_head_tokens = set(ground_truth_doc.tokens[i]
                                    for i
                                    in ground_truth_doc.mentions[
                                        original_relation.head_mention_index].token_document_indices)
                t_tail_tokens = set(ground_truth_doc.tokens[i]
                                    for i
                                    in ground_truth_doc.mentions[
                                        original_relation.tail_mention_index].token_document_indices)

                if original_relation.type != relation.type:
                    # no type match
                    continue

                if not t_head_tokens.intersection(p_head_tokens):
                    # head argument is different
                    continue

                if not t_tail_tokens.intersection(p_tail_tokens):
                    # tail argument is different
                    continue

                # got a match
                match_found = True
                break

            # check match and remove relation from the buffer
            if match_found:
                matches += 1
                original_relations_buffer.remove(original_relation)

        # how we can check the intersection of these two sets and find the results
        re_stats.append(Stats(num_pred=len(predicted_relations), num_gold=len(original_relations), num_ok=matches))

    return re_stats


if __name__ == '__main__':
    def main():
        results_dir = RES_DIR / "pred"
        pet_documents_path = RES_DIR / "gold" / "pet.jsonl"

        models = [
            "gpt-6-luna",
            "gpt-6-sol",
        ]

        print("==" * 20)
        print("Final Scores:")

        for model in models:
            for targets in (results_dir / model).iterdir():
                if targets.name == "end-to-end":
                    # end-to-end runs predict mentions and relations, so score both
                    loaders = [("md", load_ner_metrics), ("re", load_re_metrics)]
                elif targets.name in ["re", "isolated-re"]:
                    loaders = [("re", load_re_metrics)]
                elif targets.name == "md":
                    loaders = [("md", load_ner_metrics)]
                else:
                    raise AssertionError(f"Unknown target {targets.name}")

                for task, loader in loaders:
                    stats = loader(targets, pet_documents_path)
                    ps = [s.precision for s in stats]
                    rs = [s.recall for s in stats]
                    f1s = [s.f1 for s in stats]

                    print("--" * 20)
                    print(f"-- {model} ({targets.name}, {task})")
                    print("--" * 20)
                    print(f"Precision   : {np.mean(ps):5.1%} +- {np.std(ps):5.1%}")
                    print(f"Recall      : {np.mean(rs):5.1%} +- {np.std(rs):5.1%}")
                    print(f"F1          : {np.mean(f1s):5.1%} +- {np.std(f1s):5.1%}")
                    print("--" * 20)
                    print()
        print("==" * 20)


    main()
