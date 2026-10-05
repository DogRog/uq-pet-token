# Random is tuned on the validation holdout; a test-tuned random oracle is appendix-only

Headline results tune the random arm's hyperparameters on the validation holdout, freeze each checkpoint's winner, and compare every UQ metric against random on the test split. To show that UQ's advantage does not come from an under-tuned random arm, an appendix also reports the test-tuned random oracle: among the same 100 sampled configurations, the one with the best random test score, with every UQ metric compared against random at that configuration. Oracle results live apart from headline results and are never frozen as a configuration.

## Considered Options

Tuning random on the test split for the headline was rejected. It inflates random's absolute score, has a large winner's curse with 100 configurations scored on PET's 84 test sentences, and reviewers object to selecting on the test split. As an appendix check, the same bias is useful: it favours random twice, through the winner's curse and because UQ runs at random's best settings, so a UQ win against the oracle is conservative.

## Consequences

The 66-sentence validation holdout is about 13 times the 5 bootstrap sentences, which a real low-resource annotation project would not have to spare. That limits the realism of the absolute scores, not the fairness of the comparison, since only the random arm is tuned on it. It is acknowledged in the text rather than addressed with more runs.
