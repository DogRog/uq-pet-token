"""Fine-tune, predict with, and evaluate a token-classification transformer on PET.

Manual torch loop (transformers' Trainer pulls in accelerate; the manual loop gives
exact device and seed control). Every arm uses the same recipe from TrainConfig, so
*which sentences were selected* is the only experimental variable. Evaluation is
entity-level seqeval micro F1 (primary), per-type F1, and token accuracy.
"""

import gc
import logging
import random

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging

from uq_pet.config import NER_TAGS, TrainConfig
from uq_pet.dataset import tag_ids_to_labels

logger = logging.getLogger(__name__)


def _drop_load_report(record: logging.LogRecord) -> bool:
    """Reject transformers' per-checkpoint LOAD REPORT table, keep its other warnings."""
    return "LOAD REPORT" not in record.getMessage()


def configure_hf_logging(quiet: bool = True) -> None:
    """Silence per-load checkpoint reports and progress bars.

    Every arm re-loads the same checkpoint with a fresh classifier head, so the
    missing/unexpected-key report and the weight-loading bar are expected noise at any
    verbosity — `quiet` only controls transformers' *other* logging. Called from the
    pipeline entry point only — importing this module must not reconfigure a caller's
    logging.
    """
    hf_logging.set_verbosity_error() if quiet else hf_logging.set_verbosity_warning()
    hf_logging.disable_progress_bar()

    modeling_logger = logging.getLogger("transformers.modeling_utils")
    if _drop_load_report not in modeling_logger.filters:
        modeling_logger.addFilter(_drop_load_report)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training. The only global seeding in the package."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_batch(
    tokenizer,
    batch_tokens: list[list[str]],
    batch_tags: list[list[int]] | None,
    max_length: int,
):
    """Tokenize pre-split words; label the first subword per word, -100 elsewhere."""
    encoding = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    if batch_tags is None:
        return encoding, None

    labels = []
    for i, tags in enumerate(batch_tags):
        seq_labels = []
        previous_word = None
        for word_id in encoding.word_ids(batch_index=i):
            seq_labels.append(
                -100 if word_id is None or word_id == previous_word else tags[word_id]
            )
            previous_word = word_id
        labels.append(seq_labels)
    return encoding, torch.tensor(labels)


def split_off_validation(
    examples: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Carve a validation set out of the selected sentences.

    Seeded per training seed, and a local Random so it cannot perturb the global RNG
    that drives batch order and initialisation. Returns (fit, val); `val` is empty
    when the fraction rounds down to nothing, which is the caller's cue to train the
    full epoch budget instead.
    """
    n_val = int(len(examples) * val_fraction)
    if n_val < 1 or n_val >= len(examples):
        return list(examples), []
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    return (
        [ex for i, ex in enumerate(examples) if i not in val_idx],
        [examples[i] for i in sorted(val_idx)],
    )


@torch.no_grad()
def validation_loss(model, loader, device: torch.device) -> float:
    """Mean per-batch cross-entropy over the validation split, in eval mode."""
    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    for encoding, labels in loader:
        encoding = {k: v.to(device) for k, v in encoding.items()}
        total += float(model(**encoding, labels=labels.to(device)).loss)
        batches += 1
    if was_training:
        model.train()
    return total / batches if batches else float("nan")


def train_token_classifier(
    examples: list[dict],
    seed: int,
    cfg: TrainConfig,
    device: torch.device | None = None,
):
    """Fine-tune on `examples` (dicts with 'tokens' and integer 'ner-tags').

    Returns (model, tokenizer, info); the model is left on the training device in eval
    mode. `info` records how long training actually ran — with a fixed epoch budget
    that is just `cfg.epochs`, but under early stopping it is the evidence for whether
    one fixed budget was ever the right one.
    """
    set_seed(seed)
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.checkpoint,
        num_labels=len(NER_TAGS),
        id2label=dict(enumerate(NER_TAGS)),
        label2id={tag: i for i, tag in enumerate(NER_TAGS)},
    ).to(device)

    def collate(batch):
        return encode_batch(
            tokenizer,
            [ex["tokens"] for ex in batch],
            [ex["ner-tags"] for ex in batch],
            cfg.max_length,
        )

    fit_examples, val_examples = (
        split_off_validation(examples, cfg.val_fraction, seed)
        if cfg.early_stopping
        else (list(examples), [])
    )
    if cfg.early_stopping and not val_examples:
        # A 3-sentence budget cannot spare a validation sentence. Say so rather than
        # reporting an early-stopping run that quietly used the fixed schedule.
        logger.warning(
            "val_fraction %.2g of %d sentences rounds to nothing: training the full "
            "%d epochs without early stopping",
            cfg.val_fraction,
            len(examples),
            cfg.epochs,
        )

    loader = DataLoader(
        dataset=fit_examples,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            dataset=val_examples,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        if val_examples
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # The schedule spans the full budget even when early stopping cuts it short: the
    # decay a step sees must not depend on when training happens to end.
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_fraction * total_steps),
        num_training_steps=total_steps,
    )

    best_loss, best_epoch, best_state, since_best = float("inf"), None, None, 0
    epochs_run = 0

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        for encoding, labels in loader:
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding, labels=labels.to(device))
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        epochs_run = epoch

        if val_loader is None:
            continue

        loss = validation_loss(model, val_loader, device)
        if loss < best_loss - cfg.early_stopping_min_delta:
            best_loss, best_epoch, since_best = loss, epoch, 0
            # On CPU: the restore below has to survive whatever the next epochs do to
            # the live weights, and a device-side copy per improvement is wasteful.
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= cfg.early_stopping_patience:
                break

    # Early stopping without restoring the best weights would only be a slower way of
    # picking a different arbitrary epoch.
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    info = {
        "n_fit": len(fit_examples),
        "n_val": len(val_examples),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 4) if best_epoch is not None else None,
    }
    return model, tokenizer, info


@torch.no_grad()
def predict_tags(
    model,
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[list[str]]:
    """Predict one tag per word (from the first-subword logits) for each example."""
    device = device or get_device()
    model.eval()
    predictions = []

    examples = list(examples)
    for start in range(0, len(examples), batch_size):
        batch_tokens = [ex["tokens"] for ex in examples[start : start + batch_size]]
        encoding, _ = encode_batch(tokenizer, batch_tokens, None, max_length)
        inputs = {k: v.to(device) for k, v in encoding.items()}
        logits = model(**inputs).logits.argmax(dim=-1).cpu()

        for i, tokens in enumerate(batch_tokens):
            # word_ids come from the *same padded* encoding as the logits, so the
            # positions line up regardless of which side the tokenizer pads on.
            tags = ["O"] * len(tokens)  # words truncated away default to O
            previous_word = None
            for position, word_id in enumerate(encoding.word_ids(batch_index=i)):
                if word_id is not None and word_id != previous_word:
                    tags[word_id] = NER_TAGS[logits[i, position].item()]
                previous_word = word_id
            predictions.append(tags)
    return predictions


def evaluate(predictions: list[list[str]], gold: list[list[str]]) -> dict:
    """Entity-level seqeval micro F1 (primary) + per-type F1 + token accuracy."""
    report = classification_report(gold, predictions, output_dict=True, zero_division=0)
    per_type_f1 = {
        entity_type: float(metrics["f1-score"])
        for entity_type, metrics in report.items()
        if entity_type not in ("micro avg", "macro avg", "weighted avg")
    }

    total = sum(len(seq) for seq in gold)
    correct = sum(
        1
        for pred_seq, gold_seq in zip(predictions, gold, strict=True)
        for p, g in zip(pred_seq, gold_seq, strict=True)
        if p == g
    )

    return {
        "entity_f1": float(f1_score(gold, predictions, average="micro", zero_division=0)),
        "entity_precision": float(
            precision_score(gold, predictions, average="micro", zero_division=0)
        ),
        "entity_recall": float(recall_score(gold, predictions, average="micro", zero_division=0)),
        "per_type_f1": per_type_f1,
        "token_accuracy": correct / total if total else 0.0,
    }


def evaluate_model_on(
    model,
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict:
    gold = [tag_ids_to_labels(ex["ner-tags"]) for ex in examples]
    predictions = predict_tags(
        model, tokenizer, examples, max_length=max_length, batch_size=batch_size, device=device
    )
    return evaluate(predictions, gold)


def train_and_evaluate(
    train_examples: list[dict],
    test_examples: list[dict],
    seed: int,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> dict:
    """One (arm, seed) cell: fine-tune, score on the held-out split, free the model."""
    model, tokenizer, info = train_token_classifier(train_examples, seed, cfg, device=device)
    try:
        metrics = evaluate_model_on(
            model,
            tokenizer,
            test_examples,
            max_length=cfg.max_length,
            batch_size=cfg.eval_batch_size,
            device=device,
        )
        return {**metrics, **info}
    finally:
        # Drop the caller's references before asking the allocator to release cached
        # blocks. Passing them to a helper would keep them alive for the collection.
        del model, tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
