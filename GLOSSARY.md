# uq-pet-token

Sequential active learning for NER: an uncertainty arm and a random arm acquire labels for the same number of pool words, and their test scores are compared round by round.

## Language

### Data

**Train split**:
The dataset's own training part, from which the bootstrap sentences and the pool are drawn.
_Avoid_: Train (bare), training set

**Test split**:
The dataset's own held-out part, used only to evaluate.
_Avoid_: Test (bare), eval set, holdout

**Bootstrap sentences**:
The few fully labelled sentences from the train split that the bootstrap trains on before any acquisition.
_Avoid_: Seed sentences, warm-up sentences, seed (bare), seed set

**Pool**:
The rest of the train split, whose labels stay hidden until a word is acquired.
_Avoid_: Training set, unlabelled set, train (bare)

**Tuning pool**:
The pool minus the validation holdout, acquired from only while tuning hyperparameters.
_Avoid_: Train (bare), training set

**Validation holdout**:
Pool sentences set aside to score hyperparameters while tuning; they are never acquired during tuning.
_Avoid_: Val (bare), dev set

**Labelled set**:
What an arm may train on at a given round: the bootstrap sentences plus every word acquired so far.
_Avoid_: Training set, training data

**Candidate**:
A word in a pool sentence that can still be selected for labelling; each candidate is selected at most once.
_Avoid_: Token, sample, instance

**Word**:
The unit that carries one NER label and that a candidate refers to.
_Avoid_: Token

**Subword**:
One tokenizer piece of a word; only a word's first subword stands for it.
_Avoid_: Token (when a piece of a word is meant)

**Pool size**:
The share of the pool kept for an experiment; smaller pools are nested inside larger ones.
_Avoid_: Pool percent (bare), dataset percent

### Acquisition

**Bootstrap**:
The initial training on the bootstrap sentences that both arms start from.
_Avoid_: Warm-up, pretraining

**Arm**:
One of the two learners updated side by side from the same bootstrap model: the uncertainty arm selects the most uncertain candidates, the random arm selects uniformly.
_Avoid_: Branch, condition

**Round**:
One acquisition step in which each arm selects up to K new candidates, learns their labels, and updates; round 0 is the evaluation before any acquisition.
_Avoid_: Iteration, epoch, step

**Replay**:
Previously labelled words mixed into a round's update alongside the newly acquired ones.
_Avoid_: Rehearsal, memory

**Acquisition budget**:
The share of the pool's candidates that will be labelled by the end of the final round.
_Avoid_: Pool percent (bare), max pool percent

### Runs

**Model seed**:
The number that fixes one repetition of an experiment, from model initialisation onward.
_Avoid_: Seed (bare), run seed
