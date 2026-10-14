"""Decoded token text for atlas display. Read-only helpers; they never change sampling."""

from __future__ import annotations


def decode_text(tokenizer, token_ids: list[int]) -> str:
    """Identical to run_round: decode the whole shared sequence, not token pieces."""
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def token_piece(tokenizer, prefix: list[int], token_id: int) -> str:
    """How one token renders after ``prefix``, as the Atlas Explorer shows a split's candidates.

    Decoding in context keeps leading spaces and multi-byte characters that a lone
    ``decode([token_id])`` can drop or mangle. A token that renders as nothing
    (EOS and other special tokens) falls back to its special-token string.
    """
    base = decode_text(tokenizer, list(prefix))
    full = decode_text(tokenizer, [*prefix, token_id])
    piece = full[len(base):] if full.startswith(base) else decode_text(tokenizer, [token_id])
    if piece:
        return piece
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:  # minimal test tokenizers without the keyword
        return tokenizer.decode([token_id])
