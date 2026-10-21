"""Save model snapshots during one run, then score the pool with one for the suggestion heatmap.

`train` runs one model seed and UQ metric and saves the uncertainty arm's model after
the chosen rounds. `score` loads the snapshot of one round, scores and tags every pool word
as the next round's selection sees it, and writes the heatmap that notebooks/paper_results.py
shows: the words labelled so far, the next round's suggestions, and the gold entities.
"""

import argparse
import json
from pathlib import Path

from uq_pet.active_learning import snapshot_path
from uq_pet.config import ExperimentConfig
from uq_pet.data_prep import DATASET_TAGS, load_splits
from uq_pet.experiment import execute_experiment
from uq_pet.paths import PROJECT_ROOT, RESULTS_DIR
from uq_pet.token_model import (
    UQ_METRICS,
    get_device,
    load_token_classifier,
    predict_tags,
    score_token_uncertainty,
)

HEATMAP_DIR = RESULTS_DIR / "suggestion_heatmap"
# Saved and recomputed scores of the same snapshot differ only by float noise.
SCORE_TOLERANCE = 1e-4


def train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        settings = json.loads(args.config.read_text())
        settings.update(
            model_seeds=[args.model_seed],
            uq_metric=args.uq_metric,
            seed_workers=1,
            wandb_enabled=False,
        )
        config = ExperimentConfig.model_validate(settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    name = config.checkpoint.rsplit("/", maxsplit=1)[-1]
    if config.dataset != "pet":
        name += f"-{config.dataset}"
    root = args.out or HEATMAP_DIR / f"{name}-{config.uq_metric}-seed{args.model_seed}"
    if root.exists():
        parser.error(f"{root} already exists; move it aside or pass --out")
    _, _, _, _, run_dir = execute_experiment(
        config, results_dir=root, snapshot_rounds=sorted(set(args.rounds))
    )
    print(f"Run saved to {run_dir}; snapshots in {root / 'snapshots'}")


def score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    runs = sorted(args.root.glob("bert_token_uq_*"))
    if len(runs) != 1:
        parser.error(f"expected one run under {args.root}, found {len(runs)}")
    config = json.loads((runs[0] / "config.json").read_text())
    seed, metric = config["model_seeds"][0], config["uq_metric"]
    snapshot = snapshot_path(args.root / "snapshots", seed, metric, args.round)
    if not snapshot.exists():
        parser.error(f"no snapshot at {snapshot}; saved rounds: {config['snapshot_rounds']}")

    labels = DATASET_TAGS[config["dataset"]]
    (_, pool_inputs, pool_gold, _), _ = load_splits(config["dataset"], config["dataset_percent"])
    device = get_device()
    model, tokenizer = load_token_classifier(str(snapshot), device, labels=labels)
    # Every word is scored, labelled or not; the pool inputs carry no labels.
    scores = score_token_uncertainty(
        model,
        tokenizer,
        pool_inputs,
        metric=metric,
        excluded=set(),
        max_length=config["max_length"],
        batch_size=config["score_batch_size"],
        device=device,
        precision=config["effective_precision"],
    )
    # The model's own guess for each word, shown beside its suggestions.
    predicted = predict_tags(
        model,
        tokenizer,
        pool_inputs,
        labels=labels,
        max_length=config["max_length"],
        batch_size=config["score_batch_size"],
        device=device,
        precision=config["effective_precision"],
    )

    rows = [
        row
        for row in json.loads((runs[0] / "selections.json").read_text())
        if row["arm"] == "uncertainty" and row["seed"] == seed
    ]
    labelled = {(row["pool_idx"], row["word_idx"]) for row in rows if row["round"] <= args.round}
    suggested = {
        (row["pool_idx"], row["word_idx"]): row["uq_score"]
        for row in rows
        if row["round"] == args.round + 1
    }
    difference = max((abs(scores[key] - value) for key, value in suggested.items()), default=0.0)
    if difference > SCORE_TOLERANCE:
        print(
            f"warning: recomputed scores differ from the saved round {args.round + 1} "
            f"scores by up to {difference:.2e}"
        )

    heatmap = {
        "run_dir": runs[0].name,
        "checkpoint": config["checkpoint"],
        "dataset": config["dataset"],
        "uq_metric": metric,
        "model_seed": seed,
        "round": args.round,
        "total_rounds": config["rounds"],
        "k": config["k"],
        "max_score_difference": difference,
        "sentences": [
            {
                "pool_idx": example["pool_idx"],
                "document_name": example["document_name"],
                "sentence_id": example["sentence_id"],
                "tokens": example["tokens"],
                "tags": [
                    labels[pool_gold[(example["pool_idx"], word)]]
                    for word in range(len(example["tokens"]))
                ],
                # None marks words lost to subword truncation, which are never scored.
                "scores": [
                    scores.get((example["pool_idx"], word))
                    for word in range(len(example["tokens"]))
                ],
                "predicted": predicted[example["pool_idx"]],
                "labelled": [
                    (example["pool_idx"], word) in labelled
                    for word in range(len(example["tokens"]))
                ],
                "suggested": [
                    (example["pool_idx"], word) in suggested
                    for word in range(len(example["tokens"]))
                ],
            }
            for example in pool_inputs
        ],
    }
    out = args.root / f"heatmap_round_{args.round:03d}.json"
    out.write_text(json.dumps(heatmap))
    print(
        f"Heatmap saved to {out}: {len(labelled)} words labelled, {len(suggested)} suggested "
        f"(scores match the run to {difference:.1e})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Run one seed and save model snapshots.")
    train_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/pet/best_uq/distilbert.json"
    )
    train_parser.add_argument("--uq-metric", choices=UQ_METRICS, default="margin")
    train_parser.add_argument("--model-seed", type=int, default=0)
    train_parser.add_argument(
        "--rounds", type=int, nargs="+", default=[0, 10, 25, 50], help="Rounds to snapshot."
    )
    train_parser.add_argument(
        "--out", type=Path, help=f"Run folder (default: under {HEATMAP_DIR})."
    )
    score_parser = commands.add_parser("score", help="Score the pool with one round's snapshot.")
    score_parser.add_argument("root", type=Path, help="Run folder that train wrote.")
    score_parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args(argv)
    (train if args.command == "train" else score)(args, parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
