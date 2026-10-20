# Labeling efficiency uses seed-mean curves at fixed targets

Labeling efficiency and every pool-to-target number (chapters 4, 5, and 6 of `notebooks/paper_results.py`) are read where a model's seed-mean learning curve first reaches a target that is one fixed number per model: a share of its seed-mean final random score, a share of its supervised score, or the best LLM's score. The efficiency is random's pool divided by the uncertainty arm's, averaged across models geometrically. This follows Beck et al. (2021), who define labeling efficiency on averaged curves at fixed accuracies, and it keeps one noisy test score from setting the target.

## Considered Options

A per-seed target, a share of each seed's own final random score, was used first and rejected. Final rounds are noisy on the small test splits: ModernBERT seed 0 on Quishpi scores 0.35 entity F1 at 94% of the pool and 0.57 at 100%, so its target is reached only at the very end. A few such runs pulled Quishpi random's mean pool to that target up to 30% against a median of 17%, which said more about final-round noise than about learning speed. Medians over single runs were kept, but only as a robustness view, because they report each seed's spread that mean curves hide.

## Consequences

A random curve that never reaches the target counts as needing the whole pool, so its ratio is a lower bound, marked "≥"; a model whose uncertainty curve never reaches it is left out and counted. Crossings are observed rounds, not interpolated, and the ratio's direction (random over uncertainty) is stated wherever it appears, since papers use both.
