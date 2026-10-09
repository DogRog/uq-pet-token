# When to Mock

Mock at **system boundaries** only:

- Network downloads (`datasets.load_dataset`, `from_pretrained` on the Hugging Face Hub)
- Experiment tracking (W&B runs and sweeps)
- Time/randomness (pass a seed in; see below)
- File system (prefer pytest's `tmp_path` over a mock)
- Hardware (CUDA, MPS, BF16 support)

Don't mock:

- Your own classes/modules
- Internal collaborators
- Anything you control

Where a real stand-in is cheap, use it instead of a mock: a tiny `BertForTokenClassification` with a word-level tokenizer saved to `tmp_path` (the `tiny_experiment` fixture) exercises real training without a download.

## Designing for Mockability

At system boundaries, design interfaces that are easy to mock:

**1. Use dependency injection**

Pass external dependencies in rather than creating them internally:

```python
# Easy to test: the caller owns the randomness
def select_random(available, k, *, seed):
    return random.Random(seed).sample(sorted(available), k)

# Hard to test: depends on global RNG state someone else seeded
def select_random(available, k):
    return random.sample(sorted(available), k)
```

When a boundary is a module-level import, `monkeypatch.setattr` on the importing module is the Python form of injection, and fine at a true boundary: `monkeypatch.setattr(data_prep, "load_dataset", fake_loader)`.

**2. Keep the boundary narrow and specific**

Build payloads in pure functions and keep the boundary object's surface to the few methods you call:

```python
# GOOD: the payload is a pure function; the fake run needs three plain methods
payload = make_wandb_evaluation_log(...)
run.log(payload)

class FakeRun:
    def __init__(self):
        self.logs = []
    def log(self, payload):
        self.logs.append(payload)
    def define_metric(self, *args, **kwargs): ...
    def finish(self, *, exit_code): ...

# BAD: one generic passthrough; the fake must branch on what it was asked to do
def wandb_call(method_name, *args, **kwargs):
    return getattr(wandb, method_name)(*args, **kwargs)
```

The narrow approach means:
- The payload is tested directly, with no fake at all
- Each fake method records one specific shape
- No conditional logic in test setup
- Easier to see which calls a test exercises
