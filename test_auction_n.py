"""N-bidder exponential-race sampler: exact n=2 reduction and stability properties."""

import math
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.auction_n import (
    NBidderAuction,
    RaceTape,
    critical_bid,
    race_winner,
)

try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None


def random_distribution(rng, size, sparse=True):
    p = rng.dirichlet(np.full(size, rng.choice([0.2, 1.0, 5.0])))
    if sparse and rng.random() < 0.3:
        p[rng.random(size) < 0.4] = 0.0
        if p.sum() == 0.0:
            p[rng.integers(size)] = 1.0
        p /= p.sum()
    return p


def random_bids(rng, n):
    bids = rng.exponential(2.0, size=n) * (rng.random(n) > 0.15)
    if not bids.any():
        bids[rng.integers(n)] = 1.0
    return tuple(float(b) for b in bids)


class TapeTests(unittest.TestCase):
    def test_draw_is_deterministic_and_consumes_two_n_plus_two(self):
        for n in (2, 3, 5):
            a, b = np.random.default_rng(3), np.random.default_rng(3)
            tape = RaceTape.draw(a, n)
            self.assertEqual(tape, RaceTape.draw(b, n))
            reference = np.random.default_rng(3).random(2 * n + 2)
            self.assertEqual(tape.branch, reference[0])
            self.assertEqual(tape.tokens, tuple(reference[1 : 1 + n]))
            self.assertEqual(tape.race, tuple(reference[1 + n : 1 + 2 * n]))
            self.assertEqual(tape.shared, reference[-1])
            self.assertEqual(a.random(), b.random())
            self.assertTrue(all(e > 0 for e in tape.exponentials))

    def test_invalid_tapes_and_bids_are_rejected(self):
        with self.assertRaises(ValueError):
            RaceTape(0.1, (0.1, 0.2), (0.0, 0.5), 0.3)
        with self.assertRaises(ValueError):
            RaceTape(1.0, (0.1, 0.2), (0.4, 0.5), 0.3)
        with self.assertRaises(ValueError):
            RaceTape(0.1, (0.1,), (0.4,), 0.3)
        auction = NBidderAuction([[0.5, 0.5, 0], [0, 0.5, 0.5], [1, 0, 0]])
        tape = RaceTape.draw(np.random.default_rng(1), 3)
        for bids in [(0, 0, 0), (1, -1, 0), (1, math.inf, 0), (1, 1)]:
            with self.assertRaises(ValueError):
                auction.resolve(bids, tape)
        with self.assertRaises(ValueError):
            auction.resolve((1, 1, 1), RaceTape.draw(np.random.default_rng(1), 2))


class TwoBidderReductionTests(unittest.TestCase):
    def test_exact_reduction_to_reference_two_bidder_auction(self):
        rng = np.random.default_rng(2026)
        contested = shared = 0
        for _ in range(4000):
            size = int(rng.integers(2, 9))
            p_a, p_b = random_distribution(rng, size), random_distribution(rng, size)
            if rng.random() < 0.05:
                p_b = p_a.copy()
            two, race = TwoBidderAuction(p_a, p_b), NBidderAuction([p_a, p_b])
            self.assertEqual(two.shared_mass, race.shared_mass)
            a, b = random_bids(rng, 2)
            tape = RandomTape.draw(rng)
            mapped = RaceTape.from_two_bidder(tape)
            exponentials = mapped.exponentials
            self.assertAlmostEqual(
                exponentials[0] / (exponentials[0] + exponentials[1]), tape.contest, places=14
            )
            expected, actual = two.resolve(a, b, tape), race.resolve((a, b), mapped)
            self.assertEqual(actual.token_id, expected.token_id)
            label = {"shared": "shared", "A": "contested", "B": "contested"}[expected.branch]
            self.assertEqual(actual.branch, label)
            if expected.branch != "shared":
                contested += 1
                self.assertEqual(actual.winner, "AB".index(expected.branch))
                self.assertEqual(actual.candidates, (expected.candidate_a, expected.candidate_b))
            else:
                shared += 1
            for got, want in zip(actual.payments, (expected.payment_a, expected.payment_b)):
                self.assertLessEqual(abs(got - want), 1e-12 * max(1.0, want))
            for got, want in zip(race.expected_payments((a, b)), two.expected_payments(a, b)):
                self.assertLessEqual(abs(got - want), 1e-12 * max(1.0, want))
        self.assertGreater(contested, 1000)
        self.assertGreater(shared, 500)

    def test_reference_counterexample_is_reproduced(self):
        # MECHANISM.md: bids (1,1), u=.4 -> A wins and pays 2/3.
        race = NBidderAuction([[1, 0], [0, 1]])
        tape = RaceTape.from_two_bidder(RandomTape(0.9, 0.0, 0.0, 0.4))
        result = race.resolve((1, 1), tape)
        self.assertEqual((result.winner, result.token_id), (0, 0))
        self.assertAlmostEqual(result.payments[0], 2 / 3, places=14)


class RaceStabilityTests(unittest.TestCase):
    def test_marginal_is_bid_weighted_mixture_by_monte_carlo(self):
        rng = np.random.default_rng(11)
        ps = [[0.5, 0.3, 0.2, 0.0], [0.1, 0.1, 0.4, 0.4], [0.0, 0.6, 0.1, 0.3]]
        auction = NBidderAuction(ps)
        bids = (1.0, 2.5, 0.5)
        target = auction.mixture(bids)
        np.testing.assert_allclose(target, sum(b * np.array(p) for b, p in zip(bids, ps)) / 4)
        draws = 60000
        counts, wins = np.zeros(4), np.zeros(3)
        for _ in range(draws):
            allocation = auction.resolve(bids, RaceTape.draw(rng, 3))
            counts[allocation.token_id] += 1
            if allocation.winner is not None:
                wins[allocation.winner] += 1
        sigma = np.sqrt(target * (1 - target) / draws)
        self.assertTrue((np.abs(counts / draws - target) < 5 * sigma + 1e-12).all())
        win_share = wins / wins.sum()
        self.assertTrue((np.abs(win_share - np.array(bids) / 4) < 0.012).all())

    def test_race_win_probability_is_bid_share(self):
        rng = np.random.default_rng(5)
        bids = np.array([0.2, 1.0, 3.0, 0.0, 0.8])
        exponentials = rng.exponential(size=(200000, 5))
        with np.errstate(divide="ignore"):
            scores = np.where(bids > 0, exponentials / bids, np.inf)
        share = np.bincount(scores.argmin(axis=1), minlength=5) / len(scores)
        np.testing.assert_allclose(share, bids / bids.sum(), atol=0.004)
        for row in exponentials[:500]:
            self.assertEqual(race_winner(row, bids), int(np.argmin(np.where(
                bids > 0, row / np.where(bids > 0, bids, 1), np.inf))))

    def test_fixed_tape_monotonicity_and_payment_bounds(self):
        rng = np.random.default_rng(99)
        checked = 0
        for _ in range(1500):
            n = int(rng.integers(3, 6))
            size = int(rng.integers(3, 8))
            auction = NBidderAuction([random_distribution(rng, size) for _ in range(n)])
            bids = random_bids(rng, n)
            tape = RaceTape.draw(rng, n)
            before = auction.resolve(bids, tape)
            for i, pay in enumerate(before.payments):
                self.assertGreaterEqual(pay, 0.0)
                self.assertLessEqual(pay, bids[i] * (1 + 1e-12))
                if i != before.winner:
                    self.assertEqual(pay, 0.0)
            if before.branch == "shared":
                self.assertEqual(before.token_id, auction.shared_token(tape))
                continue
            i = int(rng.integers(n))
            for factor in (1.0001, 1.5, 4.0, 1e3):
                raised = list(bids)
                raised[i] = (bids[i] or 0.01) * factor
                after = auction.resolve(raised, tape)
                self.assertEqual(after.candidates, before.candidates)
                # Raising b_i only ever switches the outcome TO bidder i.
                self.assertIn(after.winner, (before.winner, i))
                if after.winner != before.winner:
                    self.assertEqual(after.token_id, before.candidates[i])
                    checked += 1
                # Lowering any OTHER bidder's ranking is impossible: if i still
                # loses, the identity of the winner among the others is fixed.
                if after.winner != i:
                    self.assertEqual(after.token_id, before.token_id)
        self.assertGreater(checked, 50)

    def test_payment_equals_bisected_threshold(self):
        rng = np.random.default_rng(1234)
        tested = 0
        while tested < 300:
            n = int(rng.integers(2, 6))
            auction = NBidderAuction([random_distribution(rng, 5, sparse=False)
                                      for _ in range(n)])
            bids = random_bids(rng, n)
            tape = RaceTape.draw(rng, n)
            result = auction.resolve(bids, tape)
            if result.branch == "shared" or sum(b > 0 for b in bids) < 2:
                continue
            w = result.winner

            def wins(x, i=w):
                trial = list(bids)
                trial[i] = x
                return sum(trial) > 0 and auction.resolve(trial, tape).winner == i

            lo, hi = 0.0, bids[w]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if wins(mid) else (mid, hi)
            self.assertAlmostEqual(result.payments[w], hi, delta=1e-12 * max(1, bids[w]))
            # The spec-sheet formula E_w*min_j b_j/E_j is not the threshold for n>=3.
            # A loser's threshold is the same formula with the current winner as a rival.
            loser = next(j for j in range(n) if j != w)
            threshold = critical_bid(tape.exponentials, bids, loser)
            self.assertFalse(wins(threshold * (1 - 1e-9), loser))
            self.assertTrue(wins(threshold * (1 + 1e-9), loser))
            tested += 1

    def test_min_formula_differs_from_threshold_for_three_bidders(self):
        exponentials, bids = (0.1, 1.0, 2.0), (1.0, 1.0, 1.0)
        self.assertEqual(race_winner(exponentials, bids), 0)
        threshold = critical_bid(exponentials, bids, 0)  # 0.1*max(1/1, 1/2) = 0.1
        wrong = 0.1 * min(1 / 1.0, 1 / 2.0)  # 0.05: bidder 0 would LOSE at this bid
        self.assertAlmostEqual(threshold, 0.1)
        self.assertEqual(race_winner(exponentials, (wrong * 1.5, 1.0, 1.0)), 1)

    def test_zero_bid_bidders_never_win_or_pay(self):
        rng = np.random.default_rng(8)
        for _ in range(2000):
            n = int(rng.integers(2, 6))
            auction = NBidderAuction([random_distribution(rng, 6) for _ in range(n)])
            bids = random_bids(rng, n)
            result = auction.resolve(bids, RaceTape.draw(rng, n))
            for i, b in enumerate(bids):
                if b == 0.0:
                    self.assertNotEqual(result.winner, i)
                    self.assertEqual(result.payments[i], 0.0)
            if sum(b > 0 for b in bids) == 1:
                self.assertEqual(sum(result.payments), 0.0)

    def test_overlapping_residuals_can_share_a_candidate(self):
        # n>=3 only: token 0 lies in supp r_1 and supp r_2. The race winner pays
        # its race-critical bid even though the other's win emits the same token.
        auction = NBidderAuction([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        self.assertAlmostEqual(auction.shared_mass, 0.0)
        tape = RaceTape.from_exponentials(0.5, (0.1, 0.1, 0.9), (0.2, 0.3, 5.0), 0.0)
        result = auction.resolve((1, 1, 1), tape)
        self.assertEqual(result.candidates[:2], (0, 0))
        self.assertEqual(result.winner, 0)
        self.assertAlmostEqual(result.payments[0], 0.2 / 0.3)
        self.assertEqual(auction.resolve((0.1, 1, 1), tape).token_id, 0)

    def test_identical_distributions_are_always_shared_and_free(self):
        p = [0.2, 0.3, 0.5]
        auction = NBidderAuction([p, p, p])
        rng = np.random.default_rng(4)
        for _ in range(200):
            result = auction.resolve((1, 2, 3), RaceTape.draw(rng, 3))
            self.assertEqual(result.branch, "shared")
            self.assertEqual(result.payments, (0.0, 0.0, 0.0))
        self.assertEqual(auction.expected_payments((1, 2, 3)), (0.0, 0.0, 0.0))


@unittest.skipIf(scipy is None, "scipy is required for the quadrature oracle.")
class ExpectedPaymentTests(unittest.TestCase):
    def test_quadrature_matches_closed_form(self):
        rng = np.random.default_rng(21)
        for _ in range(25):
            n = int(rng.integers(2, 5))
            auction = NBidderAuction([random_distribution(rng, 6) for _ in range(n)])
            bids = random_bids(rng, n)
            closed = auction.expected_payments(bids)
            for i in range(n):
                reduced = auction.expected_payment_quadrature(bids, i)
                self.assertAlmostEqual(reduced, closed[i], delta=1e-10 * max(1, closed[i]))
                if n <= 3:
                    race = auction.expected_payment_quadrature(bids, i, method="race")
                    self.assertAlmostEqual(race, closed[i], delta=1e-9 * max(1, closed[i]))

    def test_monte_carlo_matches_quadrature(self):
        rng = np.random.default_rng(77)
        ps = [[0.6, 0.2, 0.2, 0.0], [0.1, 0.1, 0.2, 0.6], [0.3, 0.5, 0.1, 0.1]]
        auction = NBidderAuction(ps)
        for bids in [(1.0, 1.0, 1.0), (3.0, 0.5, 1.5)]:
            draws = 60000
            totals, squares = np.zeros(3), np.zeros(3)
            for _ in range(draws):
                pay = np.array(auction.resolve(bids, RaceTape.draw(rng, 3)).payments)
                totals += pay
                squares += pay * pay
            mean = totals / draws
            stderr = np.sqrt(np.maximum(squares / draws - mean**2, 0) / draws)
            for i in range(3):
                oracle = auction.expected_payment_quadrature(bids, i, method="race")
                self.assertLess(abs(mean[i] - oracle), 5 * stderr[i] + 1e-9, (bids, i))

    def test_payment_depends_on_own_bid_and_rival_sum_only(self):
        auction = NBidderAuction([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        a = auction.expected_payments((1.0, 0.5, 1.5))
        b = auction.expected_payments((1.0, 1.0, 1.0))
        self.assertAlmostEqual(a[0], b[0], places=14)
        two = TwoBidderAuction([1, 0], [0, 1]).expected_payments(1.0, 2.0)
        self.assertAlmostEqual(a[0], two[0], places=14)


if __name__ == "__main__":
    unittest.main()
