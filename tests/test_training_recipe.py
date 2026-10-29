"""The bootstrap's own learning rate, gradient clipping, and averaged models (ADR 0012)."""

import copy

import pytest
import torch

import uq_pet.active_learning as active_learning
from uq_pet.active_learning import WEIGHT_AVERAGE_DECAY, full_sentence_items
from uq_pet.config import ExperimentConfig, omit_implicit_defaults
from uq_pet.data_prep import NER_TAGS
from uq_pet.token_model import (
    load_token_classifier,
    make_optimizer,
    train_items,
    update_weight_average,
)

CPU = torch.device("cpu")
RECIPE = {"bootstrap_learning_rate": 1e-3, "weight_averaging": True, "max_grad_norm": 1.0}


def tiny_model(kwargs):
    return load_token_classifier(kwargs["checkpoint"], CPU, labels=NER_TAGS)


def train_once(model, tokenizer, optimizer, examples, **settings):
    return train_items(
        model,
        optimizer,
        tokenizer,
        full_sentence_items(examples),
        batch_size=1,
        max_length=8,
        device=CPU,
        seed=0,
        **{"passes": 1, **settings},
    )


@pytest.mark.parametrize("max_grad_norm", [None, 1e-3])
def test_clipping_bounds_each_update(tiny_experiment, max_grad_norm):
    args, kwargs = tiny_experiment
    model, tokenizer = tiny_model(kwargs)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_once(model, tokenizer, optimizer, args[0][:1], max_grad_norm=max_grad_norm)
    step = torch.sqrt(
        sum(
            ((after - old) ** 2).sum()
            for after, old in zip(model.parameters(), before, strict=True)
        )
    )
    if max_grad_norm is None:
        assert step > 1e-3
    else:
        assert step <= max_grad_norm * (1 + 1e-5)


def test_warmup_rises_to_the_peak_then_decays_to_zero(tiny_experiment):
    args, kwargs = tiny_experiment
    model, tokenizer = tiny_model(kwargs)
    rates = []

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            rates.append(self.param_groups[0]["lr"])
            return super().step(closure)

    optimizer = RecordingSGD(model.parameters(), lr=0.1)
    # Two sentences, one per batch, ten passes: 20 updates, the first 2 warming up.
    train_once(model, tokenizer, optimizer, args[0], passes=10, warmup_and_decay=True)
    assert rates == pytest.approx([0.05, 0.1] + [0.1 * (20 - step) / 18 for step in range(2, 20)])
    assert optimizer.param_groups[0]["lr"] == 0


def test_weight_average_keeps_the_decay_share_and_copies_buffers():
    average = torch.nn.BatchNorm1d(2)
    live = copy.deepcopy(average)
    with torch.no_grad():
        average.weight.fill_(1.0)
        live.weight.fill_(3.0)
        live.running_mean.fill_(5.0)
    update_weight_average(average, live, 0.5)
    assert torch.equal(average.weight, torch.full((2,), 2.0))
    assert torch.equal(average.running_mean, live.running_mean)
    update_weight_average(average, live, 0.0)
    assert torch.equal(average.weight, live.weight)
    with pytest.raises(ValueError):
        update_weight_average(average, live, 1.0)


def test_bootstrap_rate_stays_in_the_bootstrap_and_arms_read_averaged_models(
    tiny_experiment, monkeypatch
):
    args, kwargs = tiny_experiment
    settings = {key: value for key, value in kwargs.items() if key != "uq_metric"}
    settings = {**settings, **RECIPE, "model_seeds": [11]}
    trained, evaluated, scored = [], [], []
    original_train = active_learning.train_items
    original_evaluate = active_learning.evaluate_model
    original_score = active_learning.score_token_uncertainty

    def train(model, optimizer, tokenizer, items, **train_settings):
        record = {
            "bootstrap": "key" not in items[0],
            "lr": optimizer.param_groups[0]["lr"],
            "warmup": train_settings.get("warmup_and_decay", False),
            "clip": train_settings["max_grad_norm"],
            "model": model,
            "initial": copy.deepcopy(optimizer.state_dict()),
        }
        loss = original_train(model, optimizer, tokenizer, items, **train_settings)
        trained.append({**record, "state": copy.deepcopy(model.state_dict())})
        return loss

    def evaluate(model, *evaluate_args, **evaluate_kwargs):
        evaluated.append((model, copy.deepcopy(model.state_dict())))
        return original_evaluate(model, *evaluate_args, **evaluate_kwargs)

    def score(model, *score_args, **score_kwargs):
        scored.append(model)
        return original_score(model, *score_args, **score_kwargs)

    monkeypatch.setattr(active_learning, "train_items", train)
    monkeypatch.setattr(active_learning, "evaluate_model", evaluate)
    monkeypatch.setattr(active_learning, "score_token_uncertainty", score)
    active_learning.run_metric_comparisons(*args, **settings, uq_metrics=["entropy"])

    bootstrap, *rounds = trained
    assert bootstrap["bootstrap"] and bootstrap["warmup"] and bootstrap["clip"] == 1.0
    assert bootstrap["lr"] == 1e-3
    assert rounds and all(not row["bootstrap"] and not row["warmup"] for row in rounds)
    assert all(row["clip"] == 1.0 for row in rounds)
    assert {row["lr"] for row in rounds} == {settings["learning_rate"]}
    # Both arms start from the bootstrap's optimizer history, at the round rate.
    first_uncertain, first_random = rounds[:2]
    torch.testing.assert_close(first_uncertain["initial"], first_random["initial"], rtol=0, atol=0)

    live_models = {id(row["model"]) for row in rounds}
    assert scored and not any(id(model) in live_models for model in scored)
    assert not any(id(model) in live_models for model, _ in evaluated)
    # The first round's averaged model sits between the bootstrap and the live weights.
    for name, value in evaluated[1][1].items():
        start, live = bootstrap["state"][name], first_uncertain["state"][name]
        if value.is_floating_point():
            expected = WEIGHT_AVERAGE_DECAY * start + (1 - WEIGHT_AVERAGE_DECAY) * live
            torch.testing.assert_close(value, expected)
        else:
            assert torch.equal(value, live)


def test_recipe_reaches_spawned_seeds_and_changes_results(tiny_experiment):
    args, kwargs = tiny_experiment
    settings = {key: value for key, value in kwargs.items() if key != "uq_metric"}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plain = active_learning.run_metric_comparisons(*args, **settings, uq_metrics=["entropy"])
        sequential = active_learning.run_metric_comparisons(
            *args, **{**settings, **RECIPE}, uq_metrics=["entropy"]
        )
        concurrent = active_learning.run_metric_comparisons(
            *args, **{**settings, **RECIPE, "seed_workers": 2}, uq_metrics=["entropy"]
        )
    finally:
        torch.set_num_threads(old_threads)
    assert concurrent == sequential
    assert sequential["entropy"][0] != plain["entropy"][0]


def test_recipe_defaults_stay_out_of_saved_settings():
    default = ExperimentConfig()
    implicit = omit_implicit_defaults(default.model_dump())
    assert not {"bootstrap_learning_rate", "weight_averaging", "max_grad_norm"} & set(implicit)
    changed = default.model_copy(update={"bootstrap_learning_rate": 1e-4})
    assert changed.resolved_wandb_run_name() != default.resolved_wandb_run_name()
    for bad in ({"bootstrap_learning_rate": 0}, {"max_grad_norm": -1.0}):
        with pytest.raises(ValueError):
            ExperimentConfig(**bad)


def test_optimizer_is_fused_only_on_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: calls.append(kwargs))
    model = torch.nn.Linear(2, 2)
    for device in ("cuda", "cpu", "mps"):
        make_optimizer(model, learning_rate=1e-4, weight_decay=0.01, device=torch.device(device))
    assert [call.get("fused") for call in calls] == [True, None, None]
    assert all(call["lr"] == 1e-4 and call["weight_decay"] == 0.01 for call in calls)
