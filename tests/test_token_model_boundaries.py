import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from uq_pet.token_model import (
    encode_targets,
    predict_tags,
    prepare_inference_batches,
    score_token_uncertainty,
    scoreable_token_keys,
    set_seed,
    train_items,
)
from uq_pet.utils.truncation import complete_word_positions


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        WordPiece(
            {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "split": 5, "##ting": 6},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )


class TinyClassifier(torch.nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 8)
        self.dropout = torch.nn.Dropout(0.5)
        self.head = torch.nn.Linear(8, 3)
        self.fail = fail

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.head(self.dropout(self.embedding(input_ids)))
        if self.fail:
            random.random()
            np.random.random()
            raise RuntimeError("training failed")
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("max_length", [4, 5, 6])
def test_complete_words_agree_across_scoring_training_and_evaluation(tokenizer, side, max_length):
    tokenizer.truncation_side = side
    pool = [{"pool_idx": 0, "tokens": ["a", "splitting", "a"]}, {"pool_idx": 1, "tokens": ["a"]}]
    expected_words = {
        ("left", 4): {2},
        ("right", 4): {0},
        ("left", 5): {1, 2},
        ("right", 5): {0, 1},
        ("left", 6): {0, 1, 2},
        ("right", 6): {0, 1, 2},
    }[(side, max_length)]
    expected_keys = {(0, word) for word in expected_words} | {(1, 0)}
    kwargs = {"max_length": max_length, "batch_size": 2}
    assert scoreable_token_keys(tokenizer, pool, **kwargs) == expected_keys
    model = TinyClassifier()
    scores = score_token_uncertainty(
        model,
        tokenizer,
        pool,
        metric="entropy",
        excluded=set(),
        device=torch.device("cpu"),
        **kwargs,
    )
    assert set(scores) == expected_keys

    for word in range(3):
        item = {**pool[0], "targets": {word: 1}}
        if word not in expected_words:
            with pytest.raises(ValueError, match="target words were truncated"):
                encode_targets(tokenizer, [item], max_length)
        else:
            encoding, labels = encode_targets(tokenizer, [item], max_length)
            first_piece = encoding.word_ids(0).index(word)
            expected_labels = torch.full_like(labels, -100)
            expected_labels[0, first_piece] = 1
            assert torch.equal(labels, expected_labels)

    if max_length < 6:
        with pytest.raises(ValueError, match="evaluation sentence was truncated"):
            predict_tags(model, tokenizer, pool, device=torch.device("cpu"), **kwargs)
    else:
        predictions = predict_tags(model, tokenizer, pool, device=torch.device("cpu"), **kwargs)
        assert [len(tags) for tags in predictions] == [3, 1]


@pytest.mark.parametrize("side", ["left", "right"])
def test_evaluation_rejects_partial_word_even_when_every_word_id_survives(tokenizer, side):
    tokenizer.truncation_side = side
    words = ["splitting", "a"] if side == "left" else ["a", "splitting"]
    encoding = tokenizer([words], is_split_into_words=True, truncation=True, max_length=4)
    assert set(encoding.word_ids(0)) == {None, 0, 1}
    with pytest.raises(ValueError, match="evaluation sentence was truncated"):
        predict_tags(
            TinyClassifier(),
            tokenizer,
            [{"tokens": words}],
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_word_spanning_multiple_overflow_chunks_is_not_scoreable(tokenizer, side):
    tokenizer.truncation_side = side
    pool = [{"pool_idx": 0, "tokens": ["splittingtingting"]}]
    kwargs = {"max_length": 3, "batch_size": 1}
    assert scoreable_token_keys(tokenizer, pool, **kwargs) == set()
    assert (
        score_token_uncertainty(
            TinyClassifier(),
            tokenizer,
            pool,
            metric="entropy",
            excluded=set(),
            device=torch.device("cpu"),
            **kwargs,
        )
        == {}
    )


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
    ),
]


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("metric", ["entropy", "least_confidence", "margin"])
def test_cached_scoring_matches_scalar_reference_without_retokenizing(
    tokenizer, device_name, metric
):
    device = torch.device(device_name)
    model = TinyClassifier().to(device).eval()
    pool = [
        {"pool_idx": 7, "tokens": ["splitting", "a"]},
        {"pool_idx": 2, "tokens": ["a"]},
        {"pool_idx": 9, "tokens": ["a", "splitting", "a"]},
    ]
    calls = []

    def counting_tokenizer(*args, **kwargs):
        calls.append(args)
        return tokenizer(*args, **kwargs)

    batches = prepare_inference_batches(counting_tokenizer, pool, max_length=5, batch_size=2)
    assert len(calls) == 2
    expected = {}
    with torch.no_grad():
        for start in range(0, len(pool), 2):
            examples = pool[start : start + 2]
            encoding = tokenizer(
                [example["tokens"] for example in examples],
                is_split_into_words=True,
                truncation=True,
                max_length=5,
                padding=True,
                return_tensors="pt",
            )
            probabilities = model(**encoding.to(device)).logits.softmax(-1).cpu()
            for idx, example in enumerate(examples):
                for word, position in complete_word_positions(encoding, idx).items():
                    values = probabilities[idx, position]
                    if metric == "entropy":
                        score = -(values * values.clamp_min(1e-12).log()).sum() / math.log(3)
                    elif metric == "least_confidence":
                        score = 1 - values.max()
                    else:
                        top_two = values.topk(2).values
                        score = 1 - (top_two[0] - top_two[1])
                    expected[(example["pool_idx"], word)] = float(score)

    for excluded in (set(), {(7, 0), (2, 0)}, set(expected)):
        actual = score_token_uncertainty(
            model,
            counting_tokenizer,
            pool,
            metric=metric,
            excluded=excluded,
            max_length=5,
            batch_size=2,
            device=device,
            prepared_batches=batches,
        )
        assert actual == pytest.approx(
            {key: value for key, value in expected.items() if key not in excluded}, abs=1e-6
        )
    assert len(calls) == 2
    assert all(tensor.device.type == "cpu" for batch in batches for tensor in batch.inputs.values())


def test_cached_evaluation_reuses_inputs_and_rejects_truncated_pool_cache(tokenizer):
    model = TinyClassifier().eval()
    examples = [{"tokens": ["a", "splitting", "a"]}, {"tokens": ["a"]}]
    kwargs = {"max_length": 6, "batch_size": 1, "device": torch.device("cpu")}
    expected = predict_tags(model, tokenizer, examples, **kwargs)
    batches = prepare_inference_batches(
        tokenizer, examples, max_length=6, batch_size=1, evaluation=True
    )

    def no_tokenization(*args, **kwargs):
        pytest.fail("cached evaluation must not tokenize again")

    for _ in range(2):
        assert (
            predict_tags(model, no_tokenization, examples, prepared_batches=batches, **kwargs)
            == expected
        )
    truncated = prepare_inference_batches(tokenizer, examples, max_length=5, batch_size=1)
    with pytest.raises(ValueError, match="evaluation sentence was truncated"):
        predict_tags(model, no_tokenization, examples, prepared_batches=truncated, **kwargs)


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("change", ["inputs", "order"])
def test_random_training_is_independent_of_other_arm(tokenizer, device_name, change):
    device = torch.device(device_name)
    set_seed(42)
    base = TinyClassifier().to(device)
    bootstrap_optimizer = torch.optim.AdamW(base.parameters(), lr=0.01)
    train_kwargs = {"passes": 1, "batch_size": 1, "max_length": 16, "device": device}
    fixed_items = [{"tokens": ["a", "splitting"], "targets": {0: 1}}]
    train_items(base, bootstrap_optimizer, tokenizer, fixed_items, seed=0, **train_kwargs)

    def run(*, perturb):
        models = {arm: copy.deepcopy(base) for arm in ("uncertainty", "random")}
        optimizers = {}
        for arm, model in models.items():
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            optimizer.load_state_dict(copy.deepcopy(bootstrap_optimizer.state_dict()))
            optimizers[arm] = optimizer
        order = ("random", "uncertainty") if perturb and change == "order" else tuple(models)
        for round_idx in (1, 2):
            for arm in order:
                items = fixed_items
                if arm == "uncertainty" and perturb and change == "inputs":
                    items = [{"tokens": ["a"] * 7, "targets": {0: 2}}]
                train_items(
                    models[arm],
                    optimizers[arm],
                    tokenizer,
                    items,
                    seed=round_idx * 10 + (arm == "random"),
                    **train_kwargs,
                )
        return models, optimizers

    normal_models, normal_optimizers = run(perturb=False)
    changed_models, changed_optimizers = run(perturb=True)
    for first, second in zip(
        normal_models["random"].parameters(), changed_models["random"].parameters(), strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for parameter, other in zip(
        normal_models["random"].parameters(), changed_models["random"].parameters(), strict=True
    ):
        for key, value in normal_optimizers["random"].state[parameter].items():
            torch.testing.assert_close(
                value, changed_optimizers["random"].state[other][key], rtol=0, atol=0
            )
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(
            base.parameters(), normal_models["random"].parameters(), strict=True
        )
    )


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("fail", [False, True])
def test_training_restores_rng_states_even_on_error(tokenizer, device_name, fail):
    device = torch.device(device_name)
    model = TinyClassifier(fail=fail).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    backend = getattr(torch, device_name) if device_name != "cpu" else None
    device_state = backend.get_rng_state() if backend is not None else None

    def train():
        train_items(
            model,
            optimizer,
            tokenizer,
            [{"tokens": ["a"], "targets": {0: 1}}],
            passes=1,
            batch_size=1,
            max_length=8,
            device=device,
            seed=123,
        )

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            train()
    else:
        train()
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), cpu_state)
    if backend is not None:
        assert torch.equal(backend.get_rng_state(), device_state)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_training_cache_preserves_padding_labels_and_reuses_text(tokenizer, padding_side):
    tokenizer.padding_side = padding_side
    cache = {}
    batches = [
        [{"tokens": ["a", "splitting"], "targets": {1: 2}}, {"tokens": ["a"], "targets": {0: 1}}],
        [{"tokens": ["a", "splitting"], "targets": {0: 3}}],
    ]
    for examples in batches:
        expected_inputs, expected_labels = encode_targets(tokenizer, examples, 8)
        inputs, labels = encode_targets(tokenizer, examples, 8, cache)
        assert torch.equal(labels, expected_labels)
        assert all(torch.equal(inputs[name], value) for name, value in expected_inputs.items())
    assert len(cache) == 2
    with pytest.raises(ValueError, match="truncated"):
        encode_targets(tokenizer, batches[0][:1], 3, cache)


def test_prepared_inference_inputs_use_requested_device(tokenizer):
    batches = prepare_inference_batches(
        tokenizer,
        [{"tokens": ["a"]}],
        max_length=8,
        batch_size=1,
        device=torch.device("meta"),
    )
    assert all(value.device.type == "meta" for value in batches[0].inputs.values())
    assert batches[0].word_positions == [{0: 1}]
