import copy
import multiprocessing
import os

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

import uq_pet.active_learning as active_learning
from uq_pet.experiment import ExperimentConfig, make_wandb_evaluation_log
from uq_pet.pet_data import NER_TAGS
from uq_pet.token_model import set_seed


@pytest.fixture
def tiny_experiment(tmp_path):
    """Exercise real model loading and spawned training without network access."""
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "b": 5}))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(tmp_path)
    set_seed(72)
    model = BertForTokenClassification(
        BertConfig(
            vocab_size=6,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            num_labels=len(NER_TAGS),
            pad_token_id=3,
        )
    )
    model.save_pretrained(tmp_path)
    examples = [
        {"tokens": ["a", "b"], "ner_tags": [1, 0]},
        {"tokens": ["b", "a", "a"], "ner_tags": [0, 1, 2]},
    ]
    pool = [
        {"pool_idx": idx, "document_name": "tiny", "sentence_id": idx, "tokens": words}
        for idx, words in enumerate((["a", "b"], ["b", "a", "b"], ["a", "a"]))
    ]
    gold = {
        (idx, word): word % 3
        for idx, item in enumerate(pool)
        for word in range(len(item["tokens"]))
    }
    config = ExperimentConfig(
        checkpoint=str(tmp_path),
        model_seeds=[11, 3, 8],
        k=2,
        bootstrap_epochs=1,
        batch_size=2,
        score_batch_size=2,
        max_length=8,
    )
    return (examples, pool, gold, examples), {
        **config.active_learning_kwargs(),
        "device": torch.device("cpu"),
    }


def seed_children():
    return [
        child for child in multiprocessing.active_children() if child.name.startswith("pet-seed-")
    ]


def test_spawned_seeds_match_sequential_and_publish_append_only_paired_progress(tiny_experiment):
    args, kwargs = tiny_experiment
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sequential = active_learning.run_active_learning(*args, **kwargs)
        snapshots = []
        worker_counts = []

        def record(rows):
            previous = snapshots[-1] if snapshots else []
            assert rows[:-2] == previous
            pair = rows[-2:]
            make_wandb_evaluation_log(pair, "entropy")
            snapshots.append(rows)
            worker_counts.append(len(seed_children()))

        concurrent = active_learning.run_active_learning(
            *args, **{**kwargs, "seed_workers": 2}, progress_callback=record
        )
    finally:
        torch.set_num_threads(old_threads)

    assert concurrent == sequential
    assert [row["seed"] for row in concurrent[0]] == [11] * 10 + [3] * 10 + [8] * 10
    assert len(snapshots) == 15  # baseline plus four rounds for each seed
    assert max(worker_counts) == 2
    assert not seed_children()
    assert sorted(snapshots[-1], key=lambda row: (row["seed"], row["round"], row["arm"])) == sorted(
        concurrent[0], key=lambda row: (row["seed"], row["round"], row["arm"])
    )


def test_worker_model_error_reaches_parent_and_stops_siblings(tiny_experiment):
    args, kwargs = tiny_experiment
    with pytest.raises(RuntimeError, match="evaluation sentence was truncated"):
        active_learning.run_active_learning(*args, **{**kwargs, "seed_workers": 2, "max_length": 3})
    assert not seed_children()


def test_callback_error_stops_workers(tiny_experiment):
    args, kwargs = tiny_experiment

    def fail(rows):
        raise ValueError("display failed")

    with pytest.raises(ValueError, match="display failed"):
        active_learning.run_active_learning(
            *args, **{**kwargs, "seed_workers": 2}, progress_callback=fail
        )
    assert not seed_children()


def crash_worker(connection, args, kwargs, cpu_threads):
    os._exit(17)


def test_abrupt_worker_exit_is_detected(monkeypatch, tiny_experiment):
    args, kwargs = tiny_experiment
    monkeypatch.setattr(active_learning, "_seed_worker", crash_worker)
    with pytest.raises(RuntimeError, match="exited without a result.*17"):
        active_learning.run_active_learning(*args, **{**kwargs, "seed_workers": 2})
    assert not seed_children()


@pytest.mark.parametrize("value", [0, -1])
def test_seed_workers_must_be_positive(value):
    with pytest.raises(ValueError, match="seed_workers"):
        ExperimentConfig(seed_workers=value)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_shared_metrics_match_independent_runs_and_train_random_only_once(
    tiny_experiment, monkeypatch, precision
):
    args, kwargs = tiny_experiment
    # CPU autocast exercises real BF16 operations offline; production selection is CUDA-only.
    if precision == "bf16":
        monkeypatch.setattr(active_learning, "resolve_precision", lambda value, device: value)
    kwargs = {**kwargs, "model_seeds": [11], "precision": precision}
    metrics = ["entropy", "least_confidence", "margin"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        independent = {
            metric: active_learning.run_active_learning(*args, **{**kwargs, "uq_metric": metric})
            for metric in metrics
        }
        calls = []
        initial_states = []
        original_train = active_learning.train_items

        def train(model, optimizer, tokenizer, items, **settings):
            bootstrap = "key" not in items[0]
            calls.append((bootstrap, settings["seed"], len(items)))
            if not bootstrap and settings["seed"] in {11 * 100_000 + 10, 11 * 100_000 + 11}:
                initial_states.append(
                    (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
                )
            return original_train(model, optimizer, tokenizer, items, **settings)

        monkeypatch.setattr(active_learning, "train_items", train)
        settings = {key: value for key, value in kwargs.items() if key != "uq_metric"}
        snapshots = {metric: [] for metric in metrics}

        def record(metric, rows):
            previous = snapshots[metric][-1] if snapshots[metric] else []
            assert rows[:-2] == previous
            make_wandb_evaluation_log(rows[-2:], metric)
            snapshots[metric].append(rows)

        shared = active_learning.run_metric_comparisons(
            *args, **settings, uq_metrics=metrics, progress_callback=record
        )
        assert shared == independent
        assert sum(bootstrap for bootstrap, _, _ in calls) == 1
        # Seven pool tokens, K=2: four rounds. Three UQ arms plus one random arm.
        assert len(calls) == 1 + 4 * 4
        assert sum(seed % 10 == 1 for bootstrap, seed, _ in calls if not bootstrap) == 4
        assert all(snapshots[metric][-1] == shared[metric][0] for metric in metrics)
        # All four arms receive identical trained weights AND nonempty AdamW history.
        assert len(initial_states) == 4
        assert initial_states[0][1]["state"]
        for state in initial_states[1:]:
            torch.testing.assert_close(state, initial_states[0], rtol=0, atol=0)
        calls.clear()
        reversed_run = active_learning.run_metric_comparisons(
            *args, **settings, uq_metrics=list(reversed(metrics))
        )
        assert reversed_run == shared
        # A final round of one new token replays only one old token for every arm.
        assert sum(size == 2 for bootstrap, _, size in calls if not bootstrap) == 4
    finally:
        torch.set_num_threads(old_threads)


def test_shared_metrics_match_spawned_seeds_and_publish_separate_paired_streams(tiny_experiment):
    args, kwargs = tiny_experiment
    kwargs = {key: value for key, value in kwargs.items() if key != "uq_metric"}
    metrics = ["entropy", "margin"]
    snapshots = {metric: [] for metric in metrics}

    def record(metric, rows):
        previous = snapshots[metric][-1] if snapshots[metric] else []
        assert rows[:-2] == previous
        make_wandb_evaluation_log(rows[-2:], metric)
        snapshots[metric].append(rows)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sequential = active_learning.run_metric_comparisons(*args, **kwargs, uq_metrics=metrics)
        concurrent = active_learning.run_metric_comparisons(
            *args, **{**kwargs, "seed_workers": 2}, uq_metrics=metrics, progress_callback=record
        )
        assert concurrent == sequential
        assert all(len(stream) == 15 for stream in snapshots.values())
        assert not seed_children()
    finally:
        torch.set_num_threads(old_threads)
