"""Token-classifier training, uncertainty scoring, and held-out evaluation."""

import math
import random

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from uq_pet.pet_data import NER_TAGS, TokenKey

UQ_METRICS = ("entropy", "least_confidence", "margin")


def set_seed(seed: int) -> None:
    """Seed every RNG used by model initialization or optimization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_targets(tokenizer, examples: list[dict], max_length: int):
    """Tokenize sentences and supervise the first subword of requested words only."""
    encoding = tokenizer(
        [example["tokens"] for example in examples],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    labels = torch.full(encoding["input_ids"].shape, -100, dtype=torch.long)

    for batch_idx, example in enumerate(examples):
        targets = example["targets"]
        found: set[int] = set()
        previous_word = None
        for position, word_idx in enumerate(encoding.word_ids(batch_index=batch_idx)):
            if word_idx is not None and word_idx != previous_word and word_idx in targets:
                labels[batch_idx, position] = targets[word_idx]
                found.add(word_idx)
            previous_word = word_idx
        missing = set(targets) - found
        if missing:
            raise ValueError(f"target words were truncated: {sorted(missing)}")
    return encoding, labels


def train_items(
    model,
    optimizer,
    tokenizer,
    items: list[dict],
    *,
    passes: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    seed: int,
) -> float:
    """Update an existing model and optimizer, returning mean batch loss."""
    if passes < 0:
        raise ValueError(f"passes must be non-negative, got {passes}")
    if not items or passes == 0:
        return float("nan")

    def collate(batch):
        return encode_targets(tokenizer, batch, max_length)

    loader = DataLoader(
        list(items),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
    )
    losses = []
    model.train()
    optimizer.zero_grad()
    for _ in range(passes):
        for encoding, labels in loader:
            inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
            loss = model(**inputs, labels=labels.to(device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
    model.eval()
    return sum(losses) / len(losses)


def load_token_classifier(checkpoint: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=len(NER_TAGS),
        id2label=dict(enumerate(NER_TAGS)),
        label2id={label: idx for idx, label in enumerate(NER_TAGS)},
    ).to(device)
    return model, tokenizer


def scoreable_token_keys(
    tokenizer, pool_inputs: list[dict], *, max_length: int, batch_size: int
) -> set[TokenKey]:
    """Return pool words that survive tokenizer truncation."""
    keys: set[TokenKey] = set()
    for start in range(0, len(pool_inputs), batch_size):
        batch = pool_inputs[start : start + batch_size]
        encoding = tokenizer(
            [example["tokens"] for example in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        for batch_idx, example in enumerate(batch):
            previous_word = None
            for word_idx in encoding.word_ids(batch_index=batch_idx):
                if word_idx is not None and word_idx != previous_word:
                    keys.add((example["pool_idx"], word_idx))
                previous_word = word_idx
    return keys


def token_uncertainty(probabilities: torch.Tensor, metric: str) -> float:
    """Reduce one word's class probabilities to a larger-is-more-uncertain score."""
    if metric == "entropy":
        value = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() / math.log(
            probabilities.numel()
        )
    elif metric == "least_confidence":
        value = 1 - probabilities.max()
    elif metric == "margin":
        top_two = probabilities.topk(2).values
        value = 1 - (top_two[0] - top_two[1])
    else:
        raise ValueError(f"unknown UQ metric {metric!r}; expected one of {UQ_METRICS}")
    return float(value)


@torch.no_grad()
def score_token_uncertainty(
    model,
    tokenizer,
    pool_inputs: list[dict],
    *,
    metric: str,
    excluded: set[TokenKey],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[TokenKey, float]:
    """Score remaining words from first-subword class probabilities."""
    if metric not in UQ_METRICS:
        raise ValueError(f"unknown UQ metric {metric!r}; expected one of {UQ_METRICS}")

    scores: dict[TokenKey, float] = {}
    model.eval()
    for start in range(0, len(pool_inputs), batch_size):
        batch = pool_inputs[start : start + batch_size]
        encoding = tokenizer(
            [example["tokens"] for example in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
        probabilities = model(**inputs).logits.softmax(dim=-1).cpu()

        for batch_idx, example in enumerate(batch):
            previous_word = None
            for position, word_idx in enumerate(encoding.word_ids(batch_index=batch_idx)):
                if word_idx is not None and word_idx != previous_word:
                    key = (example["pool_idx"], word_idx)
                    if key not in excluded:
                        scores[key] = token_uncertainty(probabilities[batch_idx, position], metric)
                previous_word = word_idx
    return scores


@torch.no_grad()
def predict_tags(
    model,
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[list[str]]:
    """Predict one tag per original word, rejecting truncated evaluation data."""
    predictions: list[list[str]] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoding = tokenizer(
            [example["tokens"] for example in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
        predicted_ids = model(**inputs).logits.argmax(dim=-1).cpu()

        for batch_idx, example in enumerate(batch):
            tags: list[str | None] = [None] * len(example["tokens"])
            previous_word = None
            for position, word_idx in enumerate(encoding.word_ids(batch_index=batch_idx)):
                if word_idx is not None and word_idx != previous_word:
                    tags[word_idx] = NER_TAGS[int(predicted_ids[batch_idx, position])]
                previous_word = word_idx
            if any(tag is None for tag in tags):
                raise ValueError("evaluation sentence was truncated; increase max_length")
            predictions.append([tag for tag in tags if tag is not None])
    return predictions


def evaluate_model(
    model,
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions = predict_tags(
        model,
        tokenizer,
        examples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    gold = [[NER_TAGS[tag] for tag in example["ner_tags"]] for example in examples]
    total = sum(len(tags) for tags in gold)
    correct = sum(
        predicted == expected
        for predicted_sequence, gold_sequence in zip(predictions, gold, strict=True)
        for predicted, expected in zip(predicted_sequence, gold_sequence, strict=True)
    )
    return {
        "entity_f1": float(f1_score(gold, predictions, average="micro", zero_division=0)),
        "entity_precision": float(
            precision_score(gold, predictions, average="micro", zero_division=0)
        ),
        "entity_recall": float(recall_score(gold, predictions, average="micro", zero_division=0)),
        "token_accuracy": correct / total if total else 0.0,
    }
