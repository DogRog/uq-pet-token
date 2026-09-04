"""Token-classifier training, uncertainty scoring, and held-out evaluation."""

import math
import random
from contextlib import contextmanager

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging

from uq_pet.pet_data import NER_TAGS, TokenKey
from uq_pet.utils.truncation import (
    complete_word_positions,
    evaluation_word_positions,
    training_word_positions,
)

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


@contextmanager
def _training_rng(seed: int):
    """Isolate each update's randomness, including dropout on accelerators."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            set_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


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
        positions = training_word_positions(encoding, batch_idx, targets)
        for word_idx, target in targets.items():
            labels[batch_idx, positions[word_idx]] = target
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
    """Update with isolated seeded randomness, returning mean batch loss."""
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
    with _training_rng(seed):
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
    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer_kwargs = {"use_fast": True}
    if config.model_type in {"roberta", "xlm-roberta"}:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)
    if not tokenizer.is_fast:
        raise ValueError(
            f"checkpoint {checkpoint!r} did not provide a fast tokenizer; "
            "word-to-subword alignment requires one"
        )
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(NER_TAGS),
            id2label=dict(enumerate(NER_TAGS)),
            label2id={label: idx for idx, label in enumerate(NER_TAGS)},
        ).to(device)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
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
            keys.update(
                (example["pool_idx"], word_idx)
                for word_idx in complete_word_positions(encoding, batch_idx)
            )
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
            for word_idx, position in complete_word_positions(encoding, batch_idx).items():
                key = (example["pool_idx"], word_idx)
                if key not in excluded:
                    scores[key] = token_uncertainty(probabilities[batch_idx, position], metric)
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
        word_positions = [
            evaluation_word_positions(encoding, idx, len(example["tokens"]))
            for idx, example in enumerate(batch)
        ]
        inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
        predicted_ids = model(**inputs).logits.argmax(dim=-1).cpu()

        for batch_idx, positions in enumerate(word_positions):
            predictions.append(
                [
                    NER_TAGS[int(predicted_ids[batch_idx, positions[idx]])]
                    for idx in range(len(positions))
                ]
            )
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
        "entity_macro_f1": float(f1_score(gold, predictions, average="macro", zero_division=0)),
        "entity_precision": float(
            precision_score(gold, predictions, average="micro", zero_division=0)
        ),
        "entity_recall": float(recall_score(gold, predictions, average="micro", zero_division=0)),
        "token_accuracy": correct / total if total else 0.0,
    }
