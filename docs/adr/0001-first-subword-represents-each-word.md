# The first subword represents each word

Uncertainty scoring, supervision, and prediction all use a word's first subword and ignore its other subwords. A single position per word means the uncertainty score needs no invented rule for aggregating several subword distributions into one word score, and the same position is used consistently from scoring through evaluation.
