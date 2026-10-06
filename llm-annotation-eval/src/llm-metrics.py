import pathlib
import typing

import numpy as np

from autoannotate.data import JsonImporter
from eval import scoring
from eval.metrics import Stats

RES_DIR = pathlib.Path(__file__).parent.parent / "res"

def load_ner_metrics(result_dir: pathlib.Path, original_file: pathlib.Path) -> typing.List[Stats]:
    predictions_files = [f for f in result_dir.iterdir() if f.suffix == '.jsonl']
    assert len(predictions_files) == 1, f"No prediction in {result_dir}"
    return _load_stats(predictions_files[0], original_file, scoring.mention_stats)


def load_re_metrics(result_dir: pathlib.Path, original_file: pathlib.Path) -> typing.List[Stats]:
    return _load_stats(result_dir / "pet.jsonl", original_file, scoring.relation_stats)


def _load_stats(predictions_file: pathlib.Path, original_file: pathlib.Path, stats_by_type) -> typing.List[Stats]:
    """:return: overall stats of every gold document that was predicted, see eval.scoring for the matching"""
    ground_truth_documents_dict = {doc.id: doc for doc in JsonImporter(original_file).do_import()}
    prediction_documents_dict = {doc.id: doc for doc in JsonImporter(predictions_file).do_import()}

    stats = []
    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        if doc_id not in prediction_documents_dict:
            # not predicted yet
            continue
        stats.append(scoring.total(stats_by_type(ground_truth_doc, prediction_documents_dict[doc_id])))
    return stats


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
