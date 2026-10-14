"""Experimental, free continuation after a contested win; not the reference auction."""

import math

import numpy as np

from auction import Allocation


def validate_continuation(tokens, threshold):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= 32:
        raise ValueError("Continuation tokens must be an integer in 0..32 (0 selects reference mode).")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not .5 < threshold <= 1):
        raise ValueError("Continuation confidence must be greater than 50% and at most 100%.")


class WinnerContinuation:
    def __init__(self, tokens=0, threshold=.98):
        validate_continuation(tokens, threshold)
        self.tokens, self.threshold = tokens, threshold
        self.winner, self.source_step, self.remaining = None, None, 0

    def take(self, p_a, p_b):
        if not self.remaining:
            return None
        p = p_a if self.winner == "A" else p_b
        # A threshold > .5 permits at most one token. This is a deterministic
        # override, explicitly outside reference-mode probability sampling.
        candidates = np.flatnonzero(p >= self.threshold)
        if not len(candidates):
            self.remaining = 0
            return None
        token = int(candidates[0])
        self.remaining -= 1
        return Allocation(token, self.winner, 0.0, 0.0, None, None), {
            "winner": self.winner, "source_step": self.source_step,
            "position": self.tokens - self.remaining, "limit": self.tokens,
            "remaining": self.remaining, "threshold": self.threshold,
            "probability": float(p[token]), "payment_policy": "free",
        }

    def observe_auction(self, allocation, step):
        self.winner = allocation.branch if allocation.branch in {"A", "B"} else None
        self.source_step = step if self.winner else None
        self.remaining = self.tokens if self.winner else 0
