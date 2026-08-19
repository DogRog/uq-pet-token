"""model_training.py tests — no checkpoint is downloaded except in the `slow` test."""

import random

import numpy as np
import pytest
import torch

from uq_pet.config import TrainConfig
from uq_pet.model_training import (
    configure_hf_logging,
    copy_model_state,
    encode_batch,
    evaluate,
    fit_token_classifier,
    get_device,
    set_seed,
    split_off_validation,
)

# --- evaluate -----------------------------------------------------------------

GOLD = [
    ["B-Actor", "I-Actor", "B-Activity"],
    ["B-Actor", "O", "B-Activity Data"],
]


def test_evaluate_perfect_predictions():
    metrics = evaluate(GOLD, GOLD)
    assert metrics["entity_f1"] == 1.0
    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["token_accuracy"] == 1.0


def test_evaluate_all_O_predictions():
    predictions = [["O"] * len(seq) for seq in GOLD]
    metrics = evaluate(predictions, GOLD)
    assert metrics["entity_f1"] == 0.0
    assert metrics["per_type_f1"] == {"Actor": 0.0, "Activity": 0.0, "Activity Data": 0.0}


def test_evaluate_partial_matches_hand_computed_micro_f1():
    # gold entities: Actor(0-1), Activity(2), Actor(0), Activity Data(2)  -> 4
    # predicted    : Actor(0-1), Activity(2), Actor(0), Activity(2)       -> 4, 3 correct
    predictions = [
        ["B-Actor", "I-Actor", "B-Activity"],
        ["B-Actor", "O", "B-Activity"],
    ]
    metrics = evaluate(predictions, GOLD)
    assert metrics["entity_precision"] == pytest.approx(0.75)
    assert metrics["entity_recall"] == pytest.approx(0.75)
    assert metrics["entity_f1"] == pytest.approx(0.75)
    assert metrics["token_accuracy"] == pytest.approx(5 / 6)


def test_evaluate_per_type_f1_excludes_averages():
    per_type = evaluate(GOLD, GOLD)["per_type_f1"]
    assert set(per_type) == {"Actor", "Activity", "Activity Data"}


def test_evaluate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([["O", "O"]], [["O"]])


# --- encode_batch -------------------------------------------------------------


class FakeEncoding(dict):
    """Mimics the parts of BatchEncoding that encode_batch touches."""

    def __init__(self, word_id_rows):
        super().__init__(input_ids=torch.zeros(len(word_id_rows), dtype=torch.long))
        self._rows = word_id_rows

    def word_ids(self, batch_index: int):
        return self._rows[batch_index]


class FakeTokenizer:
    def __init__(self, word_id_rows):
        self._rows = word_id_rows
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return FakeEncoding(self._rows)


def test_encode_batch_labels_only_the_first_subword():
    # word 0 -> 2 subwords, word 1 -> 1 subword, wrapped in [CLS]/[SEP]
    tokenizer = FakeTokenizer([[None, 0, 0, 1, None]])
    _, labels = encode_batch(tokenizer, [["signs", "off"]], [[3, 4]], max_length=128)
    assert labels.tolist() == [[-100, 3, -100, 4, -100]]


def test_encode_batch_without_tags_returns_no_labels():
    tokenizer = FakeTokenizer([[None, 0, None]])
    encoding, labels = encode_batch(tokenizer, [["hi"]], None, max_length=128)
    assert labels is None
    assert encoding is not None


def test_encode_batch_handles_padding_rows():
    tokenizer = FakeTokenizer([[None, 0, 1, None], [None, 0, None, None]])
    _, labels = encode_batch(tokenizer, [["a", "b"], ["c"]], [[1, 2], [5]], max_length=128)
    assert labels.tolist() == [[-100, 1, 2, -100], [-100, 5, -100, -100]]


# --- seeding and device -------------------------------------------------------


def test_set_seed_is_reproducible():
    set_seed(0)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(0)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == first


def test_get_device_returns_a_torch_device():
    assert isinstance(get_device(), torch.device)


def test_configure_hf_logging_is_idempotent():
    configure_hf_logging(quiet=True)
    configure_hf_logging(quiet=True)
    configure_hf_logging(quiet=False)
    configure_hf_logging(quiet=True)  # leave the suite quiet


def test_copy_model_state_is_independent_cpu_storage():
    model = torch.nn.Linear(2, 2)
    snapshot = copy_model_state(model)
    original = snapshot["weight"].clone()
    with torch.no_grad():
        model.weight.add_(10)
    assert snapshot["weight"].device.type == "cpu"
    assert torch.equal(snapshot["weight"], original)


def test_zero_epoch_fit_leaves_an_explicit_untrained_ablation():
    model = torch.nn.Linear(2, 2)
    before = copy_model_state(model)
    returned, _tokenizer, info = fit_token_classifier(
        model,
        object(),
        _examples(2),
        seed=0,
        cfg=TrainConfig(epochs=1),
        device=torch.device("cpu"),
        epochs=0,
        early_stopping=False,
    )
    assert info["epochs_run"] == 0
    assert all(torch.equal(before[key], value) for key, value in returned.state_dict().items())


# --- the validation split for early stopping ----------------------------------


def _examples(n: int) -> list[dict]:
    return [{"tokens": [f"w{i}"], "ner-tags": [0]} for i in range(n)]


def test_split_off_validation_partitions_without_overlap():
    examples = _examples(20)
    fit, val = split_off_validation(examples, 0.25, seed=0)
    assert len(val) == 5
    assert len(fit) == 15
    assert [ex["tokens"] for ex in fit + val] != []
    fit_words = {ex["tokens"][0] for ex in fit}
    val_words = {ex["tokens"][0] for ex in val}
    assert not fit_words & val_words
    assert fit_words | val_words == {f"w{i}" for i in range(20)}


def test_split_off_validation_is_seeded_and_seed_dependent():
    examples = _examples(20)
    first = split_off_validation(examples, 0.25, seed=0)[1]
    assert [ex["tokens"] for ex in first] == [
        ex["tokens"] for ex in split_off_validation(examples, 0.25, seed=0)[1]
    ]
    other = split_off_validation(examples, 0.25, seed=1)[1]
    assert [ex["tokens"] for ex in other] != [ex["tokens"] for ex in first]


def test_split_off_validation_does_not_touch_the_global_rng():
    random.seed(0)
    expected = [random.random() for _ in range(3)]
    random.seed(0)
    split_off_validation(_examples(20), 0.25, seed=7)
    assert [random.random() for _ in range(3)] == expected


def test_split_off_validation_keeps_everything_when_the_fraction_rounds_to_nothing():
    # A 3-sentence budget at 20%: int(0.6) == 0, so there is nothing to validate on.
    fit, val = split_off_validation(_examples(3), 0.2, seed=0)
    assert val == []
    assert len(fit) == 3


# --- the real thing, opt-in ---------------------------------------------------


@pytest.mark.slow
def test_train_and_evaluate_on_two_sentences(sample_examples):
    from uq_pet.model_training import train_and_evaluate

    cfg = TrainConfig(epochs=1, batch_size=2, max_length=32, eval_batch_size=2)
    metrics = train_and_evaluate(sample_examples[:2], sample_examples[2:], seed=0, cfg=cfg)
    assert 0.0 <= metrics["entity_f1"] <= 1.0
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    # A fixed budget runs every epoch and has no best epoch to report.
    assert metrics["epochs_run"] == 1
    assert metrics["best_epoch"] is None
    assert metrics["n_val"] == 0


@pytest.mark.slow
def test_early_stopping_halts_before_the_ceiling_and_reports_the_best_epoch(sample_examples):
    from uq_pet.model_training import train_and_evaluate

    cfg = TrainConfig(
        epochs=30,
        batch_size=2,
        max_length=32,
        eval_batch_size=2,
        val_fraction=0.34,  # of 3 sentences: exactly one validation sentence
        early_stopping_patience=1,
    )
    # The fixture has four sentences: three to fit and validate on, one to score.
    metrics = train_and_evaluate(sample_examples[:3], sample_examples[3:], seed=0, cfg=cfg)
    assert metrics["n_val"] == 1
    assert metrics["n_fit"] == 2
    # patience=1 on a 4-sentence fit set cannot plausibly improve for 30 straight
    # epochs; what matters is that it stopped and kept the epoch it stopped for.
    assert metrics["epochs_run"] < cfg.epochs
    assert 1 <= metrics["best_epoch"] <= metrics["epochs_run"]
    assert metrics["best_val_loss"] is not None
