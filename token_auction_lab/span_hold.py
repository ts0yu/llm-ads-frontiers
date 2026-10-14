"""Span hold: an opt-in variant of the token auction that keeps words and names whole.

Research extension (docs/SPAN_HOLD.md). The reference auction (``auction.py``,
``runner.run_round``) is unchanged; this module only adds a second sampler.

Why. The reference auction runs one contest per token. A brand name spans several
tokens, so a contest can land inside a name, and the other advertiser's context
then finishes it: "Lantern" + " Bay" + "view" gives "Lantern Bayview", a name no
advertiser owns (docs/LIVE_RESULTS.md, finding 2).

Rule (span-hold-v1). Text is cut into spans: a word, a number, or a run of
capitalised words (a name). When a contested step is won by advertiser W, W owns
the span its token starts (or joins). While W owns a span, each step W proposes
a token from its own full next-token distribution ``p_W``, by inverse CDF with
the step's hold uniform (a separate seeded stream; the reference tape is not
touched). If the proposal continues the span, it is emitted: no contest and no
charge. Otherwise the span is released and that step runs the ordinary reference
auction step with the ordinary tape. A hold lasts at most ``MAX_HOLD`` tokens.

Properties that are kept:
- probabilities are sampled, never argmaxed, averaged as logits, or filtered;
- the same selected token ID is appended to both contexts, each with its own
  prompt, mask, positions and KV cache; bids never enter a prompt;
- bids enter only through ``u < lambda`` at contested steps, so for one seed the
  output is still a step function of lambda with at most 1 + T pieces, and every
  charge is one reference critical bid per contest (held tokens are free).

What changes: the winner of a contest that starts a span also writes the rest of
that span from its own distribution, so one contest can decide several tokens.
That is a different allocation from the reference mixture; nothing here claims
it is better for advertisers or users, only that it is exact and checkable.
Simulated credits only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from auction import RandomTape, TwoBidderAuction, _cdf, _pick
from token_auction_lab.decoder import SharedPrefixDecoder
from token_auction_lab.runner import eos_ids
from token_auction_lab.token_text import decode_text, token_piece

RULE = "span-hold-v1"
MAX_HOLD = 16
# Hold uniforms come from their own generator, numpy.random.default_rng([seed, HOLD_STREAM]),
# so the reference tape sequence (one RandomTape per step) is identical in both samplers.
HOLD_STREAM = 0x5BA9


def hold_uniforms(seed: int, max_tokens: int) -> list[float]:
    rng = np.random.default_rng([seed, HOLD_STREAM])
    return [float(x) for x in rng.random(max_tokens)]


def draw_tapes(seed: int, max_tokens: int) -> list[RandomTape]:
    """The reference tape: step k uses the k-th RandomTape.draw(default_rng(seed))."""
    rng = np.random.default_rng(seed)
    return [RandomTape.draw(rng) for _ in range(max_tokens)]


def _is_name(span: str) -> bool:
    head = span.strip()[:1]
    return head.isalpha() and head.isupper()


def continues_span(span: str, piece: str) -> bool:
    """Does ``piece`` (a token as rendered after the current text) extend ``span``?

    Word-internal pieces (no leading space) extend a span ending in a letter or
    digit: "Lan"+"tern", "Bay"+"view", "Hotel"+"'s", "4"+"."+"9". A piece with one
    leading space and a capital letter extends a name: "Lantern"+" House". Any
    other piece (a new lowercase word, punctuation, a newline, EOS) ends it.
    """
    if not piece or not span.strip():
        return False
    last = span[-1]
    if piece[0] == " " and len(piece) > 1 and not piece[1].isspace():
        word = piece[1:]
        return _is_name(span) and last.isalnum() and word[:1].isalpha() and word[:1].isupper()
    first = piece[0]
    if first.isalnum():
        return last.isalnum() or last in "'’-" or (
            last == "." and len(span) > 1 and span[-2].isdigit() and first.isdigit()
        )
    if first in "'’":
        return last.isalnum() and len(piece) > 1 and piece[1].isalpha()
    if first == "-":
        return last.isalnum() and len(piece) > 1 and piece[1].isalnum()
    if first == ".":
        return last.isdigit() and (len(piece) == 1 or piece[1].isdigit())
    return False


@dataclass(frozen=True)
class SpanState:
    """The current span's text and its owner ("A", "B" or None), plus tokens held so far."""

    span: str = ""
    owner: str | None = None
    held: int = 0

    def released(self) -> SpanState:
        return replace(self, owner=None, held=0)

    def after(self, piece: str, branch: str) -> SpanState:
        """State after emitting ``piece`` on ``branch`` (shared, A, B, hold_A, hold_B)."""
        joined = continues_span(self.span, piece)
        span = self.span + piece if joined else piece
        if branch in ("A", "B"):
            return SpanState(span, branch, 0)
        if branch.startswith("hold_"):
            return SpanState(span, branch[-1], self.held + 1)
        return SpanState(span, None, 0)

    def as_record(self) -> dict:
        return {"span": self.span, "owner": self.owner, "held": self.held}


def owner_token(p: np.ndarray, u: float) -> int:
    """Inverse-CDF draw from a full next-token distribution (same rule as auction.py)."""
    return _pick(_cdf(np.asarray(p, dtype=np.float64)), u)


def hold_proposal(state: SpanState, p_a, p_b, hold_u: float, tokenizer, generated):
    """The owner's proposal and whether it is emitted. Independent of bids."""
    if state.owner is None:
        return None
    p = p_a if state.owner == "A" else p_b
    token = owner_token(p, hold_u)
    piece = token_piece(tokenizer, generated, token)
    keep = state.held < MAX_HOLD and continues_span(state.span, piece)
    return {"token_id": token, "piece": piece, "keep": keep}


def run(
    decoder,
    tokenizer,
    bid_a: float,
    bid_b: float,
    *,
    tapes: list[RandomTape],
    holds: list[float],
    stops: set[int],
    temperature: float,
    max_tokens: int,
) -> dict:
    """The span-hold loop on any decoder with ``distributions``, ``advance``, ``generated``.

    Payments are the reference ``TwoBidderAuction.resolve`` charges at contested
    steps, summed in step order; held steps charge nothing.
    """
    state = SpanState()
    steps: list[dict] = []
    payment_a = payment_b = 0.0
    stop_reason = "length"
    for step in range(max_tokens):
        p_a, p_b = decoder.distributions(temperature)
        auction = TwoBidderAuction(p_a, p_b)
        proposal = hold_proposal(state, p_a, p_b, holds[step], tokenizer, decoder.generated)
        if proposal and proposal["keep"]:
            branch, token = f"hold_{state.owner}", proposal["token_id"]
            record = {"step": step, "branch": branch, "u": tapes[step].contest,
                      "hold_u": holds[step], "shared_mass": auction.shared_mass,
                      "disagreement": auction.disagreement, "token_id": token,
                      "candidate_a": None, "candidate_b": None}
            piece = proposal["piece"]
        else:
            if proposal:
                state = state.released()
            allocation = auction.resolve(bid_a, bid_b, tapes[step])
            payment_a += allocation.payment_a
            payment_b += allocation.payment_b
            branch, token = allocation.branch, allocation.token_id
            record = {"step": step, "branch": branch, "u": tapes[step].contest,
                      "shared_mass": auction.shared_mass, "disagreement": auction.disagreement,
                      "token_id": token, "candidate_a": allocation.candidate_a,
                      "candidate_b": allocation.candidate_b}
            if proposal:
                record["released"] = {"token_id": proposal["token_id"], "piece": proposal["piece"]}
            piece = token_piece(tokenizer, decoder.generated, token)
        state = state.after(piece, branch)
        record["span"] = state.as_record()
        steps.append(record)
        decoder.advance(token)
        if token in stops:
            stop_reason = "eos"
            break
    token_ids = list(decoder.generated)
    return {
        "token_ids": token_ids,
        "text": decode_text(tokenizer, token_ids),
        "stop_reason": stop_reason,
        "steps": steps,
        "payment_a": payment_a,
        "payment_b": payment_b,
    }


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompts: list[tuple[str, list[int]]],
    bid_a: float,
    bid_b: float,
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    """One direct span-hold generation at a fixed bid pair (the atlas's oracle)."""
    if len(prompts) != 2:
        raise ValueError("Span hold is defined for exactly two advertiser contexts.")
    decoder = SharedPrefixDecoder(model, [list(ids) for _, ids in prompts])
    result = run(
        decoder, tokenizer, bid_a, bid_b, tapes=draw_tapes(seed, max_tokens),
        holds=hold_uniforms(seed, max_tokens), stops=eos_ids(model, tokenizer),
        temperature=temperature, max_tokens=max_tokens,
    )
    decoder.assert_shared_suffix()
    return result


def manifest_entry() -> dict:
    return {
        "rule": RULE,
        "max_hold": MAX_HOLD,
        "hold_stream": f"numpy.random.default_rng([seed, {HOLD_STREAM}]).random(max_tokens)",
        "description": "After a contest, the winner owns the word, number or capitalised "
        "name its token starts and proposes each next token from its own distribution; "
        "a proposal that continues the span is emitted without contest or charge, any "
        "other proposal releases the span and the step runs the reference auction.",
        "source": str(Path(__file__).name),
    }
