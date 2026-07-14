"""Fine-tune, predict with, and evaluate a token-classification transformer
on PET sentences.

Compact manual torch loop (no HF Trainer: transformers v5 Trainer requires
accelerate, and the manual loop gives exact device/seed control). Every grid
cell uses the same fixed recipe from TrainConfig so the selected data is the
only experimental variable. Evaluation is entity-level seqeval micro F1
(primary) plus per-type F1 and token accuracy.
"""

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

from .config import NER_TAGS, TrainConfig
from .data import tag_ids_to_labels

# Every grid cell re-loads the checkpoint with a fresh classifier head, so
# transformers' per-load report (missing/unexpected keys) and "Loading
# weights" bar are expected noise repeated 65+ times per run.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

logger = logging.getLogger("uq_pet")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def encode_batch(tokenizer, batch_tokens: list[list[str]], batch_tags: list[list[int]] | None,
                 max_length: int):
    """Tokenize pre-split words; label first subword per word, -100 elsewhere."""
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
        word_ids = encoding.word_ids(batch_index=i)
        seq_labels = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word:
                seq_labels.append(-100)
            else:
                seq_labels.append(tags[word_id])
            previous_word = word_id
        labels.append(seq_labels)
    return encoding, torch.tensor(labels)


def train_token_classifier(examples: list[dict], cfg: TrainConfig, seed: int):
    """Train on `examples` (dicts with 'tokens' and integer 'ner-tags').

    Returns (model, tokenizer); model is left on the training device in eval mode.
    """
    set_seed(seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.checkpoint,
        num_labels=len(NER_TAGS),
        id2label=dict(enumerate(NER_TAGS)),
        label2id={tag: i for i, tag in enumerate(NER_TAGS)},
    ).to(device)

    def collate(batch):
        encoding, labels = encode_batch(
            tokenizer,
            [ex["tokens"] for ex in batch],
            [ex["ner-tags"] for ex in batch],
            cfg.max_length,
        )
        return encoding, labels

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(list(examples), batch_size=cfg.batch_size, shuffle=True,
                        generator=generator, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_fraction * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for encoding, labels in loader:
            encoding = {k: v.to(device) for k, v in encoding.items()}
            labels = labels.to(device)
            outputs = model(**encoding, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += outputs.loss.item()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            logger.info(f"  epoch {epoch + 1}/{cfg.epochs} loss={epoch_loss / len(loader):.4f}")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_tags(model, tokenizer, examples: list[dict],
                 cfg: TrainConfig, batch_size: int = 32) -> list[list[str]]:
    """Predict one tag per word (first-subword logits) for each example."""
    device = get_device()
    model.eval()
    predictions = []

    examples = list(examples)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        batch_tokens = [ex["tokens"] for ex in batch]
        encoding, _ = encode_batch(tokenizer, batch_tokens, None, cfg.max_length)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        logits = model(**encoding).logits.argmax(dim=-1).cpu()

        for i, tokens in enumerate(batch_tokens):
            word_ids = tokenizer(
                [tokens], is_split_into_words=True, truncation=True,
                max_length=cfg.max_length,
            ).word_ids(batch_index=0)
            tags = ["O"] * len(tokens)  # words truncated away default to O
            previous_word = None
            for position, word_id in enumerate(word_ids):
                if word_id is not None and word_id != previous_word:
                    tags[word_id] = NER_TAGS[logits[i, position].item()]
                previous_word = word_id
            predictions.append(tags)
    return predictions


def evaluate(predictions: list[list[str]], gold: list[list[str]]) -> dict:
    """Entity-level seqeval micro F1 (primary) + per-type F1 + token accuracy."""
    f1 = float(f1_score(gold, predictions, average="micro", zero_division=0))
    precision = float(precision_score(gold, predictions, average="micro", zero_division=0))
    recall = float(recall_score(gold, predictions, average="micro", zero_division=0))
    report = classification_report(gold, predictions, output_dict=True, zero_division=0)
    per_type_f1 = {
        entity_type: float(metrics["f1-score"])
        for entity_type, metrics in report.items()
        if entity_type not in ("micro avg", "macro avg", "weighted avg")
    }

    total = sum(len(seq) for seq in gold)
    correct = sum(
        1 for pred_seq, gold_seq in zip(predictions, gold)
        for p, g in zip(pred_seq, gold_seq) if p == g
    )

    return {
        "entity_f1": f1,
        "entity_precision": precision,
        "entity_recall": recall,
        "per_type_f1": per_type_f1,
        "token_accuracy": correct / total if total else 0.0,
    }


def evaluate_model_on(model, tokenizer, examples: list[dict], cfg: TrainConfig) -> dict:
    predictions = predict_tags(model, tokenizer, examples, cfg)
    gold = [tag_ids_to_labels(ex["ner-tags"]) for ex in examples]
    return evaluate(predictions, gold)
