from types import SimpleNamespace

import pytest
import torch

from uq_pet.bert_uq import score_pool


class FakeEncoding(dict):
    def __init__(self, rows):
        super().__init__(input_ids=torch.zeros((len(rows), len(rows[0])), dtype=torch.long))
        self.rows = rows

    def word_ids(self, batch_index):
        return self.rows[batch_index]


class FakeTokenizer:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, *args, **kwargs):
        return FakeEncoding(self.rows)


class FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        return SimpleNamespace(logits=self.logits)


def test_score_pool_uses_first_subwords_and_reports_truncation(sample_examples):
    rows = [
        [None, 0, 0, 1, None],
        [None, 0, None, None, None],
    ]
    logits = torch.tensor(
        [
            [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    model = FakeModel(logits)
    records = score_pool(
        model,
        FakeTokenizer(rows),
        sample_examples[:2],
        max_length=8,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert model.eval_called
    assert [record["n_scored_tokens"] for record in records] == [2, 1]
    assert [record["truncated"] for record in records] == [True, True]
    assert records[0]["probabilities"][0][0] > 0.9
    assert records[0]["probabilities"][1][1] > 0.9
    assert sum(records[1]["probabilities"][0]) == pytest.approx(1.0)


def test_score_pool_never_reads_pool_labels(sample_examples):
    example = dict(sample_examples[0])
    example.pop("ner-tags")
    rows = [[None, 0, 1, 2, None]]
    logits = torch.zeros((1, 5, 3))
    records = score_pool(
        FakeModel(logits),
        FakeTokenizer(rows),
        [example],
        max_length=8,
        device=torch.device("cpu"),
    )
    assert len(records[0]["probabilities"]) == 3
