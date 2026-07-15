import math

import pytest
import torch

from uq_pet.hf_scoring import entropy_bits, generated_length


def test_entropy_uniform_distribution():
    # Equal logits -> uniform over 4 outcomes -> 2 bits.
    assert entropy_bits(torch.zeros(4)).item() == pytest.approx(2.0, abs=1e-5)


def test_entropy_near_one_hot_is_near_zero():
    logits = torch.tensor([30.0, 0.0, 0.0, 0.0])
    assert entropy_bits(logits).item() == pytest.approx(0.0, abs=1e-6)


def test_entropy_matches_manual_two_outcomes():
    # p = (0.75, 0.25) -> H = 0.75*log2(4/3) + 0.25*log2(4)
    logits = torch.log(torch.tensor([0.75, 0.25]))
    expected = 0.75 * math.log2(4 / 3) + 0.25 * 2
    assert entropy_bits(logits).item() == pytest.approx(expected, abs=1e-6)


def test_entropy_batched_rows_match_per_row():
    logits = torch.randn(3, 7)
    batched = entropy_bits(logits)
    assert batched.shape == (3,)
    for row, expected in zip(logits, batched, strict=True):
        assert entropy_bits(row).item() == pytest.approx(expected.item(), abs=1e-6)


def test_generated_length_stops_at_first_stop_token():
    # Trailing stop tokens are batch padding and must be trimmed.
    assert generated_length(torch.tensor([5, 9, 2, 2, 2]), stop_ids=[2]) == 3


def test_generated_length_without_stop_token_keeps_full_row():
    assert generated_length(torch.tensor([5, 9, 7]), stop_ids=[2]) == 3


def test_generated_length_honours_any_stop_id():
    assert generated_length(torch.tensor([5, 3, 2]), stop_ids=[2, 3]) == 2
