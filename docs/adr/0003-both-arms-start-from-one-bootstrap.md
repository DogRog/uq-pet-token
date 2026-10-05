# Both arms start from one bootstrap

One model is trained on the bootstrap sentences, and its weights and optimizer state are cloned into the uncertainty arm and the random arm. After that, each arm keeps its own weights and optimizer history across rounds. Starting from an identical state means any gap between the arms comes from what they acquire, not from how each one was initialised.
