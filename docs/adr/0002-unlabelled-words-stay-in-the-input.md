# Unlabelled words stay in the input and are excluded only from the loss

A training item feeds the whole real sentence but computes loss only on the acquired word's first subword; every other position is masked with `-100`. We rejected replacing unlabelled input words with `[MASK]`, because a word's tag depends on its neighbours (B-Actor → I-Actor), and hiding them would remove exactly the context the model needs.
