# Uncertainty acquisition can add Gumbel noise and cap repeated word forms, with beta fixed at 1

Top-K selection lets one round spend its budget on near-duplicates: in the DistilBERT entropy run on PET, a third of each early round repeated a word already chosen that round, and `the` was chosen up to 14 times in one round of 64. Two optional changes to the uncertainty arm address this. Gumbel noise adds independent Gumbel(0, 1) noise to `beta * log(score)` and takes the top K, which samples K candidates without replacement in proportion to `score ** beta` (the power acquisition of Kirsch et al., 2023, *Stochastic Batch Acquisition*). The word form cap admits at most N candidates per lowercased word in a round, after any noise, and fills any shortfall from the skipped candidates in rank order so that both arms still acquire the same number of words.

`beta` is fixed at 1, the paper's default, before any stochastic run. Choosing it from test results would tune the uncertainty arm on the test split, which ADR 0007 rules out for headline results, and tuning it on the validation holdout would cost a sweep the comparison does not need.

## Considered Options

A cap that never re-admits a word form across rounds was rejected: in PET, `the` is labelled B-Actor, O, I-Further Specification and B-Condition Specification, so later copies are still informative. A per-sentence cap was rejected because uncertain selections already average about 1.26 words per sentence, close to random's 1.12. Applying either option to the random arm was rejected; random stays the validation-tuned baseline, and its draws are unchanged whether the options are on or off.

## Consequences

Word forms come from the pool inputs, so scoring and selection stay label-free. The noise uses a NumPy generator seeded by model seed and round, separate from random's Python generator, and is shared across UQ metrics. Each uncertainty selection records its `acquisition_score` (the noisy log score, or the raw score without noise) next to `uq_score`. Both options are omitted from saved plans when off, so earlier plans and W&B run names are unchanged. Random-baseline tuning rejects them because it trains no uncertainty arm.
