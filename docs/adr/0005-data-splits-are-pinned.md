# Data splits are pinned so saved results stay comparable

`datasets` is pinned to 2.19.2. It was first pinned because the PET loading script broke on newer releases; today the pin keeps its seeded `train_test_split` producing the same bootstrap, pool, and test sentences, and `CONLL_REVISION` does the same for CoNLL-2003. Saved plans leave the full PET pool implicit, so sweeps saved before the dataset settings existed still resume.
