# Good and Bad Tests

## Good Tests

**Integration-style**: Test through real interfaces, not mocks of internal parts.

```python
# GOOD: Tests observable behavior
def test_random_selection_is_seeded_and_never_repeats_a_candidate():
    available = {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}
    first = select_random(available, 3, seed=7)
    assert first == select_random(available, 3, seed=7)
    assert len(set(first)) == 3
    assert set(first) <= available
```

Characteristics:

- Tests behavior users/callers care about
- Uses public API only
- Survives internal refactors
- Describes WHAT, not HOW
- One logical assertion per test

## Bad Tests

**Implementation-detail tests**: Coupled to internal structure.

```python
# BAD: Tests implementation details
def test_token_uncertainty_calls_uncertainty_values(monkeypatch):
    calls = []
    monkeypatch.setattr(
        token_model, "_uncertainty_values", lambda p, m: calls.append(m) or torch.tensor(0.0)
    )
    token_uncertainty(torch.tensor([0.5, 0.5]), "entropy")
    assert calls == ["entropy"]
```

Red flags:

- Mocking internal collaborators
- Testing private (`_`-prefixed) functions
- Asserting on call counts/order
- Test breaks when refactoring without behavior change
- Test name describes HOW not WHAT
- Verifying through external means instead of interface

```python
# BAD: Bypasses interface to verify
def test_encode_targets_fills_the_cache(tokenizer):
    cache = {}
    encode_targets(tokenizer, [{"tokens": ["a", "b"], "targets": {1: 3}}], 16, cache)
    assert (16, ("a", "b")) in cache

# GOOD: Verifies through interface
def test_cached_encoding_supervises_the_same_first_subwords(tokenizer):
    examples = [{"tokens": ["a", "b"], "targets": {1: 3}}]
    _, plain = encode_targets(tokenizer, examples, 16)
    _, cached = encode_targets(tokenizer, examples, 16, cache={})
    assert torch.equal(plain, cached)
```

**Tautological tests**: Expected value restates the implementation, so the test passes by construction.

```python
# BAD: Expected value is recomputed the way the code computes it
def test_entropy_of_a_uniform_word():
    p = torch.full((4,), 0.25)
    expected = float(-(p * p.log()).sum() / math.log(4))
    assert token_uncertainty(p, "entropy") == pytest.approx(expected)

# GOOD: Expected value is an independent, known literal
def test_entropy_is_one_for_uniform_and_zero_for_certain_words():
    assert token_uncertainty(torch.full((4,), 0.25), "entropy") == pytest.approx(1.0)
    assert token_uncertainty(torch.tensor([1.0, 0, 0, 0]), "entropy") == pytest.approx(0.0)
```
