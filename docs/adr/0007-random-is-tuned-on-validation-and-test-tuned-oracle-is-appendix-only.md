# Random is tuned on the validation holdout; a test-tuned random oracle is appendix-only

Headline results tune the random arm's hyperparameters on the validation holdout, freeze each checkpoint's validation winner, and compare every UQ metric against random on the test split. To show that UQ's advantage does not come from an under-tuned random arm, an appendix also reports the test-tuned random oracle: random-only tuning over the same 100 sampled configurations, scored on the test split, with every UQ metric then compared against random at the winning configuration. Oracle results live under `results/oracle_random/`, their winner is saved as `oracle_config.json` rather than `best_config.json`, and it is never copied into a `configs/<dataset>/best_uq/` folder.

## Considered Options

Tuning random on the test split for the headline was rejected. It inflates random's absolute score, has a large winner's curse with 100 configurations scored on PET's 84 test sentences, and reviewers object to selecting on the test split. As an appendix check, the same bias is useful: it favours random twice, through the winner's curse and because UQ runs at random's best settings, so a UQ win against the oracle is conservative.

## Consequences

The 66-sentence validation holdout is about 13 times the 5 bootstrap sentences, which a real low-resource annotation project would not have to spare. That limits the realism of the absolute scores, not the fairness of the comparison, since only the random arm is tuned on it. It is acknowledged in the text rather than addressed with more runs.
