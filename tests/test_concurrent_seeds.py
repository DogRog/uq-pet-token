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
from uq_pet.utils.checkpoints import save_checkpoint


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


@pytest.mark.parametrize("workers", [1, 2])
def test_round_checkpoints_resume_identical_results(
    tiny_experiment, tmp_path, monkeypatch, workers
):
    args, kwargs = tiny_experiment
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checkpoint_dir = tmp_path / "resume"
    try:
        expected = active_learning.run_active_learning(*args, **kwargs)

        def interrupt(rows):
            if rows[-1]["round"] == 1:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            active_learning.run_active_learning(
                *args,
                **{**kwargs, "seed_workers": workers},
                checkpoint_dir=checkpoint_dir,
                progress_callback=interrupt,
            )
        assert list(checkpoint_dir.glob("seed_*.pt"))
        snapshots = []
        actual = active_learning.run_active_learning(
            *args,
            **{**kwargs, "seed_workers": workers},
            checkpoint_dir=checkpoint_dir,
            progress_callback=snapshots.append,
        )
        assert actual == expected
        assert [len(rows) for rows in snapshots] == list(range(2, len(expected[0]) + 1, 2))
        assert all(rows[-1]["round"] == rows[-2]["round"] for rows in snapshots)
        for path in checkpoint_dir.glob("seed_*.pt"):
            state = torch.load(path, weights_only=True)
            assert state["round"] == 4
            assert state["models"] == state["optimizers"] == {}
        monkeypatch.setattr(
            active_learning,
            "load_token_classifier",
            lambda *args: pytest.fail("finished seeds must not reload models"),
        )
        assert (
            active_learning.run_active_learning(*args, **kwargs, checkpoint_dir=checkpoint_dir)
            == expected
        )
    finally:
        torch.set_num_threads(old_threads)
    assert not seed_children()


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_interruption_between_arms_repeats_only_unfinished_round(
    tiny_experiment, tmp_path, monkeypatch, device_name
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    args, kwargs = tiny_experiment
    kwargs = {**kwargs, "model_seeds": [11], "device": torch.device(device_name)}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checkpoint_dir = tmp_path / "half-round"
    train = active_learning.train_items
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 5:  # Bootstrap, both arms in round 1, then interruption in round 2.
            raise KeyboardInterrupt
        return train(*args, **kwargs)

    try:
        expected = active_learning.run_active_learning(*args, **kwargs)
        monkeypatch.setattr(active_learning, "train_items", interrupt)
        with pytest.raises(KeyboardInterrupt):
            active_learning.run_active_learning(*args, **kwargs, checkpoint_dir=checkpoint_dir)
        assert torch.load(checkpoint_dir / "seed_11.pt", weights_only=True)["round"] == 1
        resumed_calls = []

        def resume(*args, **kwargs):
            resumed_calls.append(kwargs["seed"])
            return train(*args, **kwargs)

        monkeypatch.setattr(active_learning, "train_items", resume)
        assert (
            active_learning.run_active_learning(*args, **kwargs, checkpoint_dir=checkpoint_dir)
            == expected
        )
        assert len(resumed_calls) == 6  # Only rounds 2–4; no bootstrap or round 1 update.
    finally:
        torch.set_num_threads(old_threads)


def test_checkpoint_rejects_changed_settings_before_loading_model(
    tiny_experiment, tmp_path, monkeypatch
):
    args, kwargs = tiny_experiment
    kwargs = {**kwargs, "model_seeds": [11]}
    checkpoint_dir = tmp_path / "mismatch"

    def interrupt(rows):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        active_learning.run_active_learning(
            *args, **kwargs, checkpoint_dir=checkpoint_dir, progress_callback=interrupt
        )
    monkeypatch.setattr(
        active_learning,
        "load_token_classifier",
        lambda *args: pytest.fail("must reject before model loading"),
    )
    with pytest.raises(ValueError, match="settings or data do not match"):
        active_learning.run_active_learning(
            *args, **{**kwargs, "k": 3}, checkpoint_dir=checkpoint_dir
        )


def test_failed_checkpoint_write_keeps_previous_round(tmp_path, monkeypatch):
    path = tmp_path / "seed_0.pt"
    save_checkpoint(path, {"round": 1, "weights": torch.ones(2)})

    def fail(state, stream):
        stream.write(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, {"round": 2})
    assert torch.load(path, weights_only=True)["round"] == 1
    assert not path.with_suffix(".tmp").exists()
