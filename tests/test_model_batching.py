"""Check real transformer vectorization, independent optimizer updates, and BF16 execution."""

import copy
import math
import random

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoModelForTokenClassification,
    BertConfig,
    DebertaV2Config,
    DistilBertConfig,
    ModernBertConfig,
    PreTrainedTokenizerFast,
    RobertaConfig,
)

from uq_pet.token_model import (
    BatchedTokenModels,
    evaluate_model,
    prepare_inference_batches,
    score_token_uncertainty,
    set_seed,
    train_items,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "b": 5}))
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )


def make_model(architecture, *, dropout=0.0):
    common = dict(
        vocab_size=6,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        num_labels=15,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    if architecture == "bert":
        config = BertConfig(**common)
    elif architecture == "roberta":
        config = RobertaConfig(**common)
    elif architecture == "distilbert":
        config = DistilBertConfig(
            **common,
            dim=8,
            hidden_dim=16,
            n_layers=2,
            n_heads=2,
            dropout=dropout,
            attention_dropout=dropout,
        )
    elif architecture == "deberta-v2":
        config = DebertaV2Config(
            **common,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            share_att_key=True,
            norm_rel_ebd="layer_norm",
            conv_kernel_size=3,
        )
    else:
        config = ModernBertConfig(
            **common,
            cls_token_id=1,
            sep_token_id=2,
            local_attention=4,
            layer_types=["full_attention", "sliding_attention"],
            attention_dropout=dropout,
            embedding_dropout=dropout,
            mlp_dropout=dropout,
            classifier_dropout=dropout,
        )
    model = AutoModelForTokenClassification.from_config(config, attn_implementation="eager")
    # Some classifier dropout defaults differ from the encoder's config.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout
    return model


ARCHITECTURES = ["bert", "distilbert", "roberta", "deberta-v2", "modernbert"]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_batched_updates_match_independent_adamw_with_bootstrap_history(tokenizer, architecture):
    set_seed(91)
    base = make_model(architecture)
    optimizer = torch.optim.AdamW(base.parameters(), lr=0.001, weight_decay=0.01)
    kwargs = dict(batch_size=2, max_length=16, device=torch.device("cpu"))
    bootstrap = [{"tokens": ["a", "b"], "targets": {0: 1, 1: 0}}]
    train_items(base, optimizer, tokenizer, bootstrap, passes=2, seed=7, **kwargs)
    group = BatchedTokenModels(
        base, optimizer.state_dict(), 2, device=kwargs["device"], precision="fp32"
    )
    items = [
        [
            {"tokens": ["a", "b"], "targets": {0: 1}},
            {"tokens": ["b"], "targets": {0: 0}},
            {"tokens": ["a", "a"], "targets": {1: 2}},
        ],
        [
            {"tokens": ["b", "a", "b"], "targets": {1: 0}},
            {"tokens": ["a"], "targets": {0: 2}},
            {"tokens": ["b", "b"], "targets": {1: 1}},
        ],
    ]
    independent = [copy.deepcopy(base), copy.deepcopy(base)]
    independent_optimizers = []
    for model in independent:
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        independent_optimizers.append(opt)
    for round_idx in (1, 2):
        seeds = [round_idx * 10, round_idx * 10 + 1]
        losses = group.train(
            tokenizer,
            items,
            seeds=seeds,
            rng_seed=round_idx,
            passes=2,
            batch_size=2,
            max_length=16,
            cache={},
        )
        for lane, (model, opt) in enumerate(zip(independent, independent_optimizers, strict=True)):
            loss = train_items(
                model, opt, tokenizer, items[lane], passes=2, seed=seeds[lane], **kwargs
            )
            assert losses[lane] == pytest.approx(loss, rel=2e-5, abs=2e-6)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    group.params[name][lane], parameter, rtol=2e-4, atol=3e-6
                )
                for moment in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        group.optimizer.state[group.params[name]][moment][lane],
                        opt.state[parameter][moment],
                        rtol=2e-4,
                        atol=3e-7,
                    )
                assert (
                    group.optimizer.state[group.params[name]]["step"]
                    == opt.state[parameter]["step"]
                )
    assert any(not torch.equal(value[0], value[1]) for value in group.params.values())
    pool = [{"pool_idx": 0, "tokens": ["a", "b"]}, {"pool_idx": 1, "tokens": ["b"]}]
    batches = prepare_inference_batches(tokenizer, pool, max_length=16, batch_size=2)
    metrics, excluded = ["entropy", "margin"], [{(0, 0)}, {(1, 0)}]
    scores = group.score(pool, batches, metrics, excluded)
    examples = [{**item, "ner_tags": [1, *([0] * (len(item["tokens"]) - 1))]} for item in pool]
    evaluations = group.evaluate(examples, batches)
    for lane, model in enumerate(independent):
        expected = score_token_uncertainty(
            model, tokenizer, pool, metric=metrics[lane], excluded=excluded[lane], **kwargs
        )
        assert scores[lane] == pytest.approx(expected, rel=2e-5, abs=2e-6)
        assert evaluations[lane] == evaluate_model(model, tokenizer, examples, **kwargs)
    cached = group.prepare_inference_inputs(batches[0])
    assert group.prepare_inference_inputs(batches[0]) is cached


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or not torch.cuda.is_bf16_supported(including_emulation=False),
                reason="native CUDA BF16 unavailable",
            ),
        ),
    ],
)
def test_bf16_vectorized_forward_backward_scoring_and_eval(tokenizer, architecture, device_name):
    device = torch.device(device_name)
    base = make_model(architecture, dropout=0.1)
    optimizer = torch.optim.AdamW(base.parameters(), lr=0.001)
    group = BatchedTokenModels(base, optimizer.state_dict(), 2, device=device, precision="bf16")
    items = [[{"tokens": ["a", "b"], "targets": {0: 1}}], [{"tokens": ["b"], "targets": {0: 0}}]]
    losses = group.train(
        tokenizer, items, seeds=[1, 2], rng_seed=3, passes=2, batch_size=1, max_length=16, cache={}
    )
    assert all(math.isfinite(value) for value in losses)
    assert all(value.dtype == torch.float32 for value in group.params.values())
    assert all(state["exp_avg"].dtype == torch.float32 for state in group.optimizer.state.values())
    pool = [{"pool_idx": 0, "tokens": ["a", "b"]}, {"pool_idx": 1, "tokens": ["b"]}]
    batches = prepare_inference_batches(tokenizer, pool, max_length=16, batch_size=2, device=device)
    logits = group.forward(group.prepare_inputs(batches[0].inputs), shared_inputs=True)
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (2, 2, 4, 15)
    scores = group.score(pool, batches, ["entropy", "margin"], [{(0, 0)}, {(1, 0)}])
    assert set(scores[0]) == {(0, 1), (1, 0)}
    assert set(scores[1]) == {(0, 0), (0, 1)}
    assert all(math.isfinite(value) for lane in scores for value in lane.values())
    evaluations = group.evaluate(
        [{**row, "ner_tags": [1, *([0] * (len(row["tokens"]) - 1))]} for row in pool], batches
    )
    assert len(evaluations) == 2
    assert all(0 <= row["entity_f1"] <= 1 for row in evaluations)


def test_model_lanes_have_separate_gradients_and_reproducible_dropout(tokenizer):
    set_seed(32)
    base = make_model("bert", dropout=0.2)
    optimizer = torch.optim.AdamW(base.parameters())
    group = BatchedTokenModels(
        base, optimizer.state_dict(), 2, device=torch.device("cpu"), precision="fp32"
    )
    group.template.train()
    inputs = group.prepare_inputs(
        tokenizer([["a", "b"]], is_split_into_words=True, return_tensors="pt")
    )
    # Independent dropout produces different outputs even for identical weights and inputs.
    logits = group.forward(inputs, shared_inputs=True)
    assert not torch.equal(logits[0], logits[1])
    logits[0].sum().backward()
    assert all(
        torch.count_nonzero(value.grad[1]) == 0
        for value in group.params.values()
        if value.grad is not None
    )
    items = [[{"tokens": ["a", "b"], "targets": {0: 1}}]] * 2
    state = torch.get_rng_state()
    python_state = random.getstate()
    first = group.train(
        tokenizer, items, seeds=[1, 2], rng_seed=4, passes=1, batch_size=1, max_length=16, cache={}
    )
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == python_state
    other = BatchedTokenModels(
        base, optimizer.state_dict(), 2, device=torch.device("cpu"), precision="fp32"
    )
    second = other.train(
        tokenizer, items, seeds=[1, 2], rng_seed=4, passes=1, batch_size=1, max_length=16, cache={}
    )
    assert first == second
    torch.testing.assert_close(group.params, other.params, rtol=0, atol=0)
