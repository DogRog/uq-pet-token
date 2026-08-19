"""Pool scoring with a PET token-classification transformer."""

import torch

from uq_pet.model_training import encode_batch, get_device


@torch.no_grad()
def score_pool(
    model,
    tokenizer,
    examples: list[dict],
    *,
    max_length: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[dict]:
    """Return one word-probability record per pool sentence.

    Only the first subword represents a word, matching the label alignment used for
    training. Padding and special tokens have no word ID and therefore never affect
    uncertainty. A record also reports truncation so the run metadata can audit it.
    """
    device = device or get_device()
    model.eval()
    records = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        batch_tokens = [example["tokens"] for example in batch]
        encoding, _ = encode_batch(tokenizer, batch_tokens, None, max_length)
        inputs = {key: value.to(device) for key, value in encoding.items()}
        probabilities = model(**inputs).logits.softmax(dim=-1).cpu()

        for batch_index, tokens in enumerate(batch_tokens):
            word_vectors = []
            previous_word = None
            for position, word_id in enumerate(encoding.word_ids(batch_index=batch_index)):
                if word_id is not None and word_id != previous_word:
                    word_vectors.append(probabilities[batch_index, position].tolist())
                previous_word = word_id
            records.append(
                {
                    "idx": start + batch_index,
                    "n_tokens": len(tokens),
                    "n_scored_tokens": len(word_vectors),
                    "truncated": len(word_vectors) < len(tokens),
                    "probabilities": word_vectors,
                }
            )
    return records
