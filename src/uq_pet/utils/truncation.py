"""Word alignment and truncation checks for fast tokenizer outputs."""

from collections.abc import Collection


def complete_word_positions(encoding, batch_idx: int) -> dict[int, int]:
    """Locate first subwords only when every piece of the word was retained."""
    # Fast tokenizers retain overflow metadata even without returning overflow
    # chunks as model inputs. A word shared with any overflow chunk is incomplete,
    # including when left truncation keeps only its continuation pieces.
    truncated_words = {
        word_idx
        for overflow in encoding.encodings[batch_idx].overflowing
        for word_idx in overflow.word_ids
        if word_idx is not None
    }
    positions = {}
    for position, word_idx in enumerate(encoding.word_ids(batch_index=batch_idx)):
        if word_idx is not None and word_idx not in truncated_words:
            positions.setdefault(word_idx, position)
    return positions


def training_word_positions(
    encoding, batch_idx: int, target_words: Collection[int]
) -> dict[int, int]:
    """Require every supervised word to be complete."""
    positions = complete_word_positions(encoding, batch_idx)
    missing = set(target_words) - positions.keys()
    if missing:
        raise ValueError(f"target words were truncated: {sorted(missing)}")
    return positions


def evaluation_word_positions(encoding, batch_idx: int, word_count: int) -> dict[int, int]:
    """Require a complete prediction position for every original word."""
    positions = complete_word_positions(encoding, batch_idx)
    if positions.keys() != set(range(word_count)):
        raise ValueError("evaluation sentence was truncated; increase max_length")
    return positions
