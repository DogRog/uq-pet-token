# Replay is scaled to each round's new words

Each round, both arms replay previously labelled words, sampled per word and in proportion to the number of words actually acquired that round, including a smaller final round. Both arms share the replay ratio, update passes, batch size, and learning rate, so they receive the same update budget every round and differ only in which words they acquired.
