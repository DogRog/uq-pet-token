# Partial entity match is reported beside exact match, never tuned on

Every evaluation also scores partial-match entities (same type, any shared word), the lenient view `llm-annotation-eval` uses by default, so the two projects' scores can be compared and a boundary-only error does not count as a full miss. Exact-match entity F1 stays the only tuning objective, the validation winner's criterion, and the headline UQ-minus-random gap: it is what earlier sweeps and ADR 0007 were defined on, and choosing between the two after seeing test results would let the test split pick the metric.

## Consequences

Every run saves the predicted tags behind each evaluation row, with the gold tags, in `predictions.npz`, so any entity metric, including a different matching rule, can be recomputed without retraining. Runs saved before this change have neither the partial-match columns nor `predictions.npz`.
