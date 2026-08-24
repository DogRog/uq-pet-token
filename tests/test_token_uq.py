from io import StringIO
from types import SimpleNamespace

import pytest
import torch
from rich.console import Console

import uq_pet.active_learning as active_learning
from uq_pet.active_learning import (
    _round_progress_table,
    acquisition_schedule,
    reveal_pool_items,
    run_active_learning,
    sample_replay,
    select_random,
    select_top_k,
    write_run,
)
from uq_pet.token_model import (
    encode_targets,
    load_token_classifier,
    score_token_uncertainty,
    scoreable_token_keys,
    token_uncertainty,
)


class FakeEncoding(dict):
    def __init__(self, input_ids: torch.Tensor, word_ids: list[list[int | None]]):
        super().__init__(
            input_ids=input_ids,
            attention_mask=(input_ids != 0).long(),
        )
        self.word_id_rows = word_ids

    def word_ids(self, batch_index: int):
        return self.word_id_rows[batch_index]


class FakeTokenizer:
    """Split the literal word 'split' into two pieces and pad with ID zero."""

    def __call__(self, batch_tokens, *, max_length, **kwargs):
        rows = []
        word_rows = []
        for tokens in batch_tokens:
            ids = [9]
            word_ids = [None]
            next_id = 1
            for word_idx, token in enumerate(tokens):
                pieces = 2 if token == "split" else 1
                ids.extend(range(next_id, next_id + pieces))
                word_ids.extend([word_idx] * pieces)
                next_id += pieces
            ids.append(9)
            word_ids.append(None)
            rows.append(ids[:max_length])
            word_rows.append(word_ids[:max_length])

        width = max(len(row) for row in rows)
        for ids, word_ids in zip(rows, word_rows, strict=True):
            padding = width - len(ids)
            ids.extend([0] * padding)
            word_ids.extend([None] * padding)
        return FakeEncoding(torch.tensor(rows), word_rows)


class FakeModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, 3))
        # ID 1 is the first piece of "split" and is deliberately confident. ID 2,
        # its continuation piece, stays uniform and must not affect word uncertainty.
        logits[input_ids == 1, 0] = 10.0
        return SimpleNamespace(logits=logits)


def test_round_progress_renders_arms_in_two_columns():
    output = StringIO()
    console = Console(file=output, width=160, color_system=None)
    common = {"round": 7, "total_rounds": 368, "n_acquired": 112, "token_budget": 5888}
    table = _round_progress_table(
        {
            "uncertainty": {
                **common,
                "entity_f1": 0.3205,
                "train_loss": 0.5357,
            },
            "random": {
                **common,
                "entity_f1": 0.4497,
                "train_loss": 0.2596,
            },
        }
    )

    console.print(table)

    rendered_lines = output.getvalue().splitlines()
    assert len(rendered_lines) == 1
    assert "uncertainty" in rendered_lines[0]
    assert "random" in rendered_lines[0]
    assert rendered_lines[0].index("uncertainty") < rendered_lines[0].index("random")


def test_encode_targets_masks_unselected_and_continuation_subwords():
    tokenizer = FakeTokenizer()
    encoding, labels = encode_targets(
        tokenizer,
        [{"tokens": ["split", "word"], "targets": {0: 4, 1: 7}}],
        max_length=10,
    )

    assert encoding.word_ids(0) == [None, 0, 0, 1, None]
    assert labels.tolist() == [[-100, 4, -100, 7, -100]]


def test_encode_targets_rejects_a_truncated_selected_word():
    with pytest.raises(ValueError, match="target words were truncated"):
        encode_targets(
            FakeTokenizer(),
            [{"tokens": ["first", "second"], "targets": {1: 3}}],
            max_length=2,
        )


def test_uncertainty_uses_first_subword_and_never_needs_pool_labels():
    pool_inputs = [
        {
            "pool_idx": 0,
            "document_name": "doc",
            "sentence_id": 1,
            "tokens": ["split", "word"],
        }
    ]
    scores = score_token_uncertainty(
        FakeModel(),
        FakeTokenizer(),
        pool_inputs,
        metric="entropy",
        excluded=set(),
        max_length=10,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert scores[(0, 0)] < 0.01
    assert scores[(0, 1)] == pytest.approx(1.0)


def test_available_uncertainty_metrics_return_larger_scores_for_ambiguity():
    confident = torch.tensor([0.8, 0.1, 0.1])
    ambiguous = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    for metric in ("entropy", "least_confidence", "margin"):
        assert token_uncertainty(ambiguous, metric) > token_uncertainty(confident, metric)
    assert token_uncertainty(confident, "least_confidence") == pytest.approx(0.2)
    assert token_uncertainty(confident, "margin") == pytest.approx(0.3)


def test_unknown_uncertainty_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown UQ metric"):
        token_uncertainty(torch.tensor([0.5, 0.5]), "mystery")


def test_scoreable_keys_exclude_truncated_words():
    pool_inputs = [
        {"pool_idx": 4, "tokens": ["first", "second"], "document_name": "d", "sentence_id": 0}
    ]
    keys = scoreable_token_keys(FakeTokenizer(), pool_inputs, max_length=2, batch_size=1)
    assert keys == {(4, 0)}


def test_roberta_loader_requests_fast_pretokenized_compatible_tokenizer(monkeypatch):
    tokenizer_calls = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    monkeypatch.setattr(
        "uq_pet.token_model.AutoConfig.from_pretrained",
        lambda checkpoint: SimpleNamespace(model_type="roberta"),
    )

    def fake_tokenizer(checkpoint, **kwargs):
        tokenizer_calls.append((checkpoint, kwargs))
        return SimpleNamespace(is_fast=True)

    monkeypatch.setattr("uq_pet.token_model.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr(
        "uq_pet.token_model.AutoModelForTokenClassification.from_pretrained",
        lambda *args, **kwargs: TinyModel(),
    )

    load_token_classifier("roberta-base", torch.device("cpu"))

    assert tokenizer_calls == [("roberta-base", {"use_fast": True, "add_prefix_space": True})]


def test_selection_is_exact_reproducible_and_without_replacement():
    scores = {(0, 0): 0.2, (0, 1): 0.9, (1, 0): 0.5}
    assert select_top_k(scores, 2) == [(0, 1), (1, 0)]

    available = set(scores)
    first = select_random(available, 2, seed=7)
    assert first == select_random(available, 2, seed=7)
    assert len(first) == len(set(first)) == 2


def test_replay_is_limited_and_seeded():
    items = [{"key": idx} for idx in range(10)]
    replay = sample_replay(items, k=4, ratio=1.0, seed=3)
    assert len(replay) == 4
    assert replay == sample_replay(items, k=4, ratio=1.0, seed=3)
    assert sample_replay(items, k=4, ratio=0.0, seed=3) == []


def test_acquisition_schedule_uses_full_k_sized_rounds():
    assert acquisition_schedule(320, k=8, max_pool_percent=50) == (20, 160)
    assert acquisition_schedule(320, k=16, max_pool_percent=50) == (10, 160)
    assert acquisition_schedule(320, k=32, max_pool_percent=50) == (5, 160)
    assert acquisition_schedule(101, k=32, max_pool_percent=100) == (3, 96)


def test_acquisition_schedule_rejects_invalid_or_too_small_caps():
    with pytest.raises(ValueError, match="must be in"):
        acquisition_schedule(100, k=8, max_pool_percent=0)
    with pytest.raises(ValueError, match="fewer than one full"):
        acquisition_schedule(100, k=32, max_pool_percent=1)


def test_labels_are_revealed_only_for_selected_keys():
    pool_inputs = [
        {
            "pool_idx": 0,
            "tokens": ["one", "two"],
            "document_name": "doc",
            "sentence_id": 0,
        }
    ]
    pool_gold = {(0, 0): 3, (0, 1): 8}
    items = reveal_pool_items([(0, 1)], pool_inputs, pool_gold)

    assert items == [
        {
            "key": ("pool", 0, 1),
            "tokens": ["one", "two"],
            "targets": {1: 8},
        }
    ]


def test_write_run_records_config_results_and_selections(tmp_path):
    run_dir = write_run(
        {"k": 2},
        [{"seed": 0, "arm": "random", "round": 0}],
        [{"seed": 0, "arm": "random", "token": "word"}],
        results_dir=tmp_path,
    )

    assert (run_dir / "config.json").read_text().strip().startswith("{")
    assert "random" in (run_dir / "results.csv").read_text()
    assert "word" in (run_dir / "selections.json").read_text()


def test_run_reports_progress_after_baseline_and_each_complete_round(monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    pool_inputs = [
        {
            "pool_idx": idx,
            "document_name": "doc",
            "sentence_id": idx,
            "tokens": [f"word-{idx}"],
        }
        for idx in range(2)
    ]
    pool_gold = {(idx, 0): 0 for idx in range(2)}

    monkeypatch.setattr(active_learning, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        active_learning,
        "load_token_classifier",
        lambda checkpoint, device: (TinyModel(), object()),
    )
    monkeypatch.setattr(
        active_learning,
        "scoreable_token_keys",
        lambda *args, **kwargs: set(pool_gold),
    )
    monkeypatch.setattr(active_learning, "train_items", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        active_learning,
        "evaluate_model",
        lambda *args, **kwargs: {
            "entity_f1": 0.2,
            "entity_precision": 0.2,
            "entity_recall": 0.2,
            "token_accuracy": 0.8,
        },
    )

    def fake_scores(*args, excluded, **kwargs):
        return {key: 1.0 - key[0] / 10 for key in pool_gold if key not in excluded}

    monkeypatch.setattr(active_learning, "score_token_uncertainty", fake_scores)
    snapshots = []
    results, _ = run_active_learning(
        [{"tokens": ["seed"], "ner_tags": [0]}],
        pool_inputs,
        pool_gold,
        [],
        checkpoint="fake",
        model_seeds=[0],
        uq_metric="entropy",
        k=1,
        max_pool_percent=100,
        bootstrap_epochs=1,
        update_passes=1,
        replay_ratio=1.0,
        learning_rate=5e-5,
        weight_decay=0.01,
        batch_size=1,
        score_batch_size=1,
        max_length=8,
        device=torch.device("cpu"),
        progress_callback=snapshots.append,
    )

    assert [len(snapshot) for snapshot in snapshots] == [2, 4, 6]
    assert [max(row["round"] for row in snapshot) for snapshot in snapshots] == [0, 1, 2]
    assert {row["percent_acquired"] for row in results} == {0.0, 50.0, 100.0}
    assert {row["token_budget"] for row in results} == {2}
    assert snapshots[-1] == results
