"""Synthetic cases for the experimental continuation policy, not text-quality claims."""

import unittest

import numpy as np

from auction import Allocation
from token_auction_lab.continuation import WinnerContinuation, validate_continuation


class ContinuationTests(unittest.TestCase):
    def test_three_free_tokens_then_fresh_auction_required_for_both_sides(self):
        for side in ("A", "B"):
            with self.subTest(side=side):
                carry = WinnerContinuation(3, .98)
                carry.observe_auction(Allocation(0, side, .3, .2, 0, 1), 7)
                p = np.array([.98, .02])
                for position in range(1, 4):
                    allocation, info = carry.take(p, p)
                    self.assertEqual((allocation.token_id, allocation.branch), (0, side))
                    self.assertEqual(allocation.payment_a + allocation.payment_b, 0)
                    self.assertEqual(info["source_step"], 7)
                    self.assertEqual(info["position"], position)
                    self.assertEqual(info["remaining"], 3 - position)
                self.assertIsNone(carry.take(p, p))

    def test_confidence_miss_ends_allowance_and_does_not_resume(self):
        carry = WinnerContinuation(3, .98)
        carry.observe_auction(Allocation(0, "A", .3, 0, 0, 1), 0)
        self.assertIsNone(carry.take(np.array([.979999, .020001]), np.array([1., 0.])))
        self.assertIsNone(carry.take(np.array([1., 0.]), np.array([1., 0.])))
        carry.observe_auction(Allocation(1, "B", 0, .4, 0, 1), 1)
        allocation, info = carry.take(np.array([1., 0.]), np.array([.01, .99]))
        self.assertEqual((allocation.token_id, allocation.branch, info["source_step"]), (1, "B", 1))

    def test_shared_outcomes_and_disabled_mode_grant_nothing(self):
        for limit in (0, 3):
            carry = WinnerContinuation(limit)
            p = np.array([1., 0.])
            self.assertIsNone(carry.take(p, p))
            carry.observe_auction(Allocation(0, "shared", 0, 0, None, None), 0)
            self.assertIsNone(carry.take(p, p))
        carry = WinnerContinuation(0)
        carry.observe_auction(Allocation(0, "A", .3, 0, 0, 1), 0)
        self.assertIsNone(carry.take(p, p))

    def test_settings_reject_ambiguous_thresholds_and_invalid_limits(self):
        for tokens, threshold in [(True, .98), (-1, .98), (33, .98), (1.5, .98),
                (3, .5), (3, 0), (3, 1.01), (3, float("nan")), (3, True), (3, None)]:
            with self.subTest(tokens=tokens, threshold=threshold), self.assertRaises(ValueError):
                validate_continuation(tokens, threshold)
        validate_continuation(0, 1)
        validate_continuation(32, .500001)
