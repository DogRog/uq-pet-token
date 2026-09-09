"""Token-classifier training, uncertainty scoring, and held-out evaluation."""

import copy
import logging
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.utils import logging as transformers_logging

from uq_pet.pet_data import NER_TAGS, TokenKey
from uq_pet.utils.truncation import (
    complete_word_positions,
    evaluation_word_positions,
    training_word_positions,
)

UQ_METRICS = ("entropy", "least_confidence", "margin")

# Applied in each seed process before model downloads begin.
logging.getLogger("huggingface_hub.utils._http").addFilter(
    lambda record: (
        "You are sending unauthenticated requests to the HF Hub." not in record.getMessage()
    )
)


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


def resolve_precision(precision: str, device: torch.device) -> str:
    """Use native CUDA BF16 when available; reject unsupported explicit requests."""
    if precision not in {"auto", "fp32", "bf16"}:
        raise ValueError(f"unknown precision {precision!r}")
    device = torch.device(device)
    supported = False
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "bf16" and not supported:
        raise ValueError("BF16 requires a CUDA GPU with native BF16 support; use auto or fp32")
    return ("bf16" if supported else "fp32") if precision == "auto" else precision


def _autocast(device: torch.device, precision: str):
    """Autocast forward operations while retaining FP32 parameters and optimizer state."""
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )


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


def encode_targets(tokenizer, examples: list[dict], max_length: int, cache: dict | None = None):
    if cache is not None:
        return _cached_targets(tokenizer, examples, max_length, cache)
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


def _cached_targets(tokenizer, examples, max_length, cache):
    # Cache label-free sentence encodings; supervision is rebuilt for every batch.
    entries = []
    for example in examples:
        key = (max_length, tuple(example["tokens"]))
        if key not in cache:
            encoding = tokenizer(
                [example["tokens"]],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            cache[key] = (
                {name: values[0] for name, values in encoding.items()},
                complete_word_positions(encoding, 0),
            )
        entries.append(cache[key])
    inputs = tokenizer.pad([entry[0] for entry in entries], return_tensors="pt")
    labels = torch.full(inputs["input_ids"].shape, -100, dtype=torch.long)
    for idx, (example, (encoding, positions)) in enumerate(zip(examples, entries, strict=True)):
        missing = example["targets"].keys() - positions.keys()
        if missing:
            raise ValueError(f"target words were truncated: {sorted(missing)}")
        offset = (
            labels.shape[1] - len(encoding["input_ids"]) if tokenizer.padding_side == "left" else 0
        )
        for word, label in example["targets"].items():
            labels[idx, positions[word] + offset] = label
    return inputs, labels


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
    tokenization_cache: dict | None = None,
    precision: str = "fp32",
) -> float:
    """Update with isolated seeded randomness, returning mean batch loss."""
    if passes < 0:
        raise ValueError(f"passes must be non-negative, got {passes}")
    if not items or passes == 0:
        return float("nan")

    def collate(batch):
        return encode_targets(tokenizer, batch, max_length, tokenization_cache)

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
                with _autocast(device, precision):
                    loss = model(**inputs, labels=labels.to(device)).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.detach())
    model.eval()
    losses = torch.stack(losses).cpu().tolist()
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
            dtype=torch.float32,
            num_labels=len(NER_TAGS),
            id2label=dict(enumerate(NER_TAGS)),
            label2id={label: idx for idx, label in enumerate(NER_TAGS)},
        ).to(device)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    return model, tokenizer


@dataclass
class InferenceBatch:
    """Device inputs and first-subword alignment reused across acquisition rounds."""

    inputs: dict[str, torch.Tensor]
    word_positions: list[dict[int, int]]


def prepare_inference_batches(
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int,
    evaluation: bool = False,
    device: torch.device | None = None,
) -> list[InferenceBatch]:
    """Tokenize fixed batches once, without retaining or reading any labels."""
    batches = []
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
        positions = [
            evaluation_word_positions(encoding, idx, len(example["tokens"]))
            if evaluation
            else complete_word_positions(encoding, idx)
            for idx, example in enumerate(batch)
        ]
        batches.append(
            InferenceBatch({name: value.to(device) for name, value in encoding.items()}, positions)
        )
    return batches


def scoreable_token_keys(
    tokenizer,
    pool_inputs: list[dict],
    *,
    max_length: int,
    batch_size: int,
    prepared_batches: list[InferenceBatch] | None = None,
) -> set[TokenKey]:
    """Return pool words that survive tokenizer truncation."""
    if prepared_batches is None:
        prepared_batches = prepare_inference_batches(
            tokenizer, pool_inputs, max_length=max_length, batch_size=batch_size
        )
    return {
        (example["pool_idx"], word_idx)
        for example, positions in zip(
            pool_inputs,
            (positions for batch in prepared_batches for positions in batch.word_positions),
            strict=True,
        )
        for word_idx in positions
    }


def _uncertainty_values(probabilities: torch.Tensor, metric: str) -> torch.Tensor:
    """Reduce the class dimension for one word or a batch of words."""
    if metric == "entropy":
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1) / math.log(
            probabilities.shape[-1]
        )
    if metric == "least_confidence":
        return 1 - probabilities.amax(dim=-1)
    if metric == "margin":
        top_two = probabilities.topk(2, dim=-1).values
        return 1 - (top_two[..., 0] - top_two[..., 1])
    raise ValueError(f"unknown UQ metric {metric!r}; expected one of {UQ_METRICS}")


def token_uncertainty(probabilities: torch.Tensor, metric: str) -> float:
    """Reduce one word's class probabilities to a larger-is-more-uncertain score."""
    return float(_uncertainty_values(probabilities, metric))


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
    prepared_batches: list[InferenceBatch] | None = None,
    precision: str = "fp32",
) -> dict[TokenKey, float]:
    """Score remaining words from first-subword class probabilities."""
    if metric not in UQ_METRICS:
        raise ValueError(f"unknown UQ metric {metric!r}; expected one of {UQ_METRICS}")

    scores: dict[TokenKey, float] = {}
    model.eval()
    if prepared_batches is None:
        prepared_batches = prepare_inference_batches(
            tokenizer, pool_inputs, max_length=max_length, batch_size=batch_size
        )
    offset = 0
    for batch in prepared_batches:
        keys, rows, columns = [], [], []
        for batch_idx, positions in enumerate(batch.word_positions):
            pool_idx = pool_inputs[offset + batch_idx]["pool_idx"]
            for word_idx, position in positions.items():
                key = (pool_idx, word_idx)
                if key not in excluded:
                    keys.append(key)
                    rows.append(batch_idx)
                    columns.append(position)
        offset += len(batch.word_positions)
        if not keys:
            continue
        inputs = {name: tensor.to(device) for name, tensor in batch.inputs.items()}
        with _autocast(device, precision):
            logits = model(**inputs).logits
        probabilities = logits[rows, columns].float().softmax(dim=-1)
        values = _uncertainty_values(probabilities, metric).cpu().tolist()
        scores.update(zip(keys, values, strict=True))
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
    prepared_batches: list[InferenceBatch] | None = None,
    precision: str = "fp32",
) -> list[list[str]]:
    """Predict one tag per original word, rejecting truncated evaluation data."""
    predictions: list[list[str]] = []
    model.eval()
    if prepared_batches is None:
        prepared_batches = prepare_inference_batches(
            tokenizer, examples, max_length=max_length, batch_size=batch_size, evaluation=True
        )
    for example, positions in zip(
        examples,
        (positions for batch in prepared_batches for positions in batch.word_positions),
        strict=True,
    ):
        if positions.keys() != set(range(len(example["tokens"]))):
            raise ValueError("evaluation sentence was truncated; increase max_length")
    for batch in prepared_batches:
        inputs = {name: tensor.to(device) for name, tensor in batch.inputs.items()}
        with _autocast(device, precision):
            logits = model(**inputs).logits
        predicted_ids = logits.argmax(dim=-1).cpu().tolist()

        for batch_idx, positions in enumerate(batch.word_positions):
            predictions.append(
                [
                    NER_TAGS[predicted_ids[batch_idx][positions[idx]]]
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
    prepared_batches: list[InferenceBatch] | None = None,
    precision: str = "fp32",
) -> dict[str, float]:
    predictions = predict_tags(
        model,
        tokenizer,
        examples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        prepared_batches=prepared_batches,
        precision=precision,
    )
    return evaluate_predictions(examples, predictions)


def evaluate_predictions(examples, predictions):
    """Compute the same word/entity metrics for scalar and batched model predictions."""
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


class BatchedTokenModels:
    """Vectorize a small group of independent learners with stacked FP32 AdamW state."""

    def __init__(self, base_model, optimizer_state, count, *, device, precision):
        self.count = count
        self.device = device
        self.precision = precision
        self.inference_inputs = {}
        # Stack on CPU before transfer; retain no extra trained model copies on the GPU.
        self.params = {
            name: value.detach()
            .cpu()
            .unsqueeze(0)
            .repeat(count, *([1] * value.ndim))
            .to(device)
            .requires_grad_()
            for name, value in base_model.named_parameters()
        }
        self.buffers = {
            name: value.detach().cpu().unsqueeze(0).repeat(count, *([1] * value.ndim)).to(device)
            for name, value in base_model.named_buffers()
        }
        self.template = copy.deepcopy(base_model).to("meta")
        # Eager attention supports vmap backward and independent dropout across architectures.
        self.template.set_attn_implementation("eager")
        groups = optimizer_state["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(self.params):
            raise ValueError(
                "model batching requires the experiment's single AdamW parameter group"
            )
        options = {key: value for key, value in groups[0].items() if key != "params"}
        self.optimizer = torch.optim.AdamW([{"params": list(self.params.values()), **options}])
        for parameter, original_id in zip(self.params.values(), groups[0]["params"], strict=True):
            original = optimizer_state["state"].get(original_id, {})
            self.optimizer.state[parameter] = {
                name: value.clone()
                if name == "step"
                else value.unsqueeze(0).repeat(count, *([1] * value.ndim)).to(device)
                for name, value in original.items()
            }

    def prepare_inputs(self, inputs):
        """Build masks outside vmap to avoid Transformers' tensor-dependent mask branches."""
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        config = self.template.config
        if config.model_type == "deberta-v2":
            return inputs  # DeBERTa builds its boolean mask without data-dependent branches.
        dummy = torch.empty((*inputs["input_ids"].shape, 1), device=self.device)
        kwargs = dict(
            config=config,
            inputs_embeds=dummy,
            attention_mask=inputs["attention_mask"],
            allow_is_bidirectional_skip=False,
        )
        full_mask = create_bidirectional_mask(**kwargs)
        inputs["attention_mask"] = (
            {
                "full_attention": full_mask,
                "sliding_attention": create_bidirectional_sliding_window_mask(**kwargs),
            }
            if config.model_type == "modernbert"
            else full_mask
        )
        return inputs

    def prepare_inference_inputs(self, batch):
        """Reuse each fixed pool/test attention mask throughout this group's rounds."""
        key = id(batch)
        if key not in self.inference_inputs:
            self.inference_inputs[key] = (batch, self.prepare_inputs(batch.inputs))
        return self.inference_inputs[key][1]

    def forward(self, inputs, *, shared_inputs, indices=None):
        """Use batched matrix operations with a distinct parameter slice for every learner."""
        parameters = self.params
        buffers = self.buffers
        if indices is not None:
            parameters = {name: value[indices] for name, value in parameters.items()}
            buffers = {name: value[indices] for name, value in buffers.items()}
        if self.precision == "bf16":
            # vmap's linear batching rule can promote BF16 matmuls back to FP32 when
            # adding an FP32 bias. Cast execution weights, retaining differentiable
            # links to FP32 master parameters and their separate AdamW moments.
            parameters = {name: value.to(torch.bfloat16) for name, value in parameters.items()}

        def call(parameters, buffers, inputs):
            return torch.func.functional_call(
                self.template, (parameters, buffers), (), inputs
            ).logits

        with _autocast(self.device, self.precision):
            return torch.vmap(
                call,
                in_dims=(0, 0, None if shared_inputs else 0),
                randomness="different" if self.template.training else "error",
            )(parameters, buffers, inputs)

    def train(self, tokenizer, items, *, seeds, rng_seed, passes, batch_size, max_length, cache):
        """Sum per-model mean losses, preserving each learner's gradient and step count."""
        if len(items) != self.count or len(seeds) != self.count:
            raise ValueError("one item list and shuffle seed are required per model")
        if len({len(lane) for lane in items}) != 1 or not items[0]:
            raise ValueError("batched learners require equal nonzero training budgets")
        loaders = [
            DataLoader(
                lane,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=list,
                generator=torch.Generator().manual_seed(seed),
            )
            for lane, seed in zip(items, seeds, strict=True)
        ]
        losses = []
        self.template.train()
        try:
            with _training_rng(rng_seed):
                for _ in range(passes):
                    for batches in zip(*loaders, strict=True):
                        encoding, targets = encode_targets(
                            tokenizer,
                            [item for batch in batches for item in batch],
                            max_length,
                            cache,
                        )
                        inputs = self.prepare_inputs(encoding)
                        inputs = tree_map(
                            lambda value: value.reshape(self.count, -1, *value.shape[1:]), inputs
                        )
                        targets = targets.to(self.device).reshape(self.count, -1, targets.shape[-1])
                        self.optimizer.zero_grad()
                        logits = self.forward(inputs, shared_inputs=False)
                        token_losses = torch.nn.functional.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            targets.flatten(),
                            ignore_index=-100,
                            reduction="none",
                        ).reshape_as(targets)
                        per_model = token_losses.sum((1, 2)) / (targets != -100).sum((1, 2))
                        per_model.sum().backward()
                        self.optimizer.step()
                        losses.append(per_model.detach())
        finally:
            self.template.eval()
        return torch.stack(losses).mean(0).cpu().tolist()

    @torch.no_grad()
    def score(self, pool_inputs, batches, metrics, excluded):
        """Score all UQ learners together using label-free inputs and FP32 probabilities."""
        self.template.eval()
        scores = [dict() for _ in metrics]
        # UQ models occupy the leading slices; the optional random learner is last.
        offset = 0
        for batch in batches:
            keys, rows, columns = [], [], []
            for row, positions in enumerate(batch.word_positions):
                pool_idx = pool_inputs[offset + row]["pool_idx"]
                for word, column in positions.items():
                    key = (pool_idx, word)
                    if any(key not in selected for selected in excluded):
                        keys.append(key)
                        rows.append(row)
                        columns.append(column)
            offset += len(batch.word_positions)
            if not keys:
                continue
            logits = self.forward(
                self.prepare_inference_inputs(batch),
                shared_inputs=True,
                indices=slice(0, len(metrics)),
            )
            probabilities = logits[:, rows, columns].float().softmax(-1)
            for index, metric in enumerate(metrics):
                values = _uncertainty_values(probabilities[index], metric).cpu().tolist()
                scores[index].update(
                    (key, value)
                    for key, value in zip(keys, values, strict=True)
                    if key not in excluded[index]
                )
        return scores

    @torch.no_grad()
    def evaluate(self, examples, batches):
        """Evaluate all learners together while retaining per-model entity metrics."""
        self.template.eval()
        predictions = [[] for _ in range(self.count)]
        for example, positions in zip(
            examples,
            (positions for batch in batches for positions in batch.word_positions),
            strict=True,
        ):
            if positions.keys() != set(range(len(example["tokens"]))):
                raise ValueError("evaluation sentence was truncated; increase max_length")
        for batch in batches:
            ids = (
                self.forward(self.prepare_inference_inputs(batch), shared_inputs=True)
                .argmax(-1)
                .cpu()
                .tolist()
            )
            for lane in range(self.count):
                for row, positions in enumerate(batch.word_positions):
                    predictions[lane].append(
                        [
                            NER_TAGS[ids[lane][row][positions[word]]]
                            for word in range(len(positions))
                        ]
                    )
        return [evaluate_predictions(examples, prediction) for prediction in predictions]
