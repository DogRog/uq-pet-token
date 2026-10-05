# The supervised baseline is a directly comparable upper bound

The supervised baseline trains on the same bootstrap sentences and pool, fully labelled, and tests on the same test split as active learning, so its score is the ceiling active learning is compared against. Its hyperparameters are tuned only on the validation holdout, and each trained model reads the test split once.
