"""Exact two-Gaussian auction (MOONSHOTS.md section 3). Synthetic Monte Carlo checks only."""

import math
import unittest

import numpy as np
from scipy import integrate
from scipy.special import ndtr

from auction import RandomTape
from token_auction_lab.gaussian_auction import (
    GaussianTape,
    GaussianTwoBidderAuction,
    _log_residual_mass,
    common_quantile,
    mixture_cdf_1d,
    residual_a_quantile,
    residual_b_quantile_b_frame,
    shared_mass_1d,
    tv_distance,
)

KS_CRIT = 2.13  # sqrt(-log(1e-4 / 2) / 2)
DELTAS = (0.0, 1e-8, 1e-3, 0.5, 3.0, 12.0, 40.0)


def make_auction(delta, dim=6, s=0.7, seed=0):
    rng = np.random.default_rng(seed)
    mu_a = rng.normal(size=dim)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    return GaussianTwoBidderAuction(mu_a, mu_a + delta * s * direction, s)


def ks_distance(samples, cdf):
    x = np.sort(np.asarray(samples))
    n = len(x)
    f = cdf(x)
    return max(float(np.max(np.arange(1, n + 1) / n - f)), float(np.max(f - np.arange(n) / n)))


class OneDimensionalPieces(unittest.TestCase):
    def test_tv_formula_matches_numerical_integration(self):
        for delta in (1e-3, 0.5, 3.0, 12.0):
            tv, _ = integrate.quad(
                lambda t: 0.5 * abs(math.exp(-t * t / 2) - math.exp(-(t - delta) ** 2 / 2))
                / math.sqrt(2 * math.pi), -40, 40 + delta, points=[delta / 2], limit=400,
                epsabs=1e-14, epsrel=1e-12)
            self.assertAlmostEqual(tv_distance(delta), tv, delta=1e-10)
            self.assertAlmostEqual(tv_distance(delta) + shared_mass_1d(delta), 1.0, places=15)
        # Small delta: the series 2x/sqrt(pi), x = delta/(2 sqrt 2), without cancellation.
        delta = 1e-8
        self.assertAlmostEqual(tv_distance(delta) / (delta / math.sqrt(2 * math.pi)), 1.0,
                               places=12)
        self.assertEqual((tv_distance(0.0), shared_mass_1d(0.0)), (0.0, 1.0))
        self.assertGreater(shared_mass_1d(40.0), 0.0)  # 5.5e-89, still representable
        self.assertEqual(shared_mass_1d(80.0), 0.0)  # underflows: branch skipped
        self.assertEqual((tv_distance(math.inf), shared_mass_1d(math.inf)), (1.0, 0.0))

    def test_residual_quantile_inverts_its_cdf(self):
        for delta in DELTAS[1:] + (9.999e-3, 1e-2, 1e3, math.inf):
            log_d = _log_residual_mass(delta, min(delta / 2, 40.0)) if math.isfinite(delta) else 0
            for w in (1e-300, 1e-12, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-9, 1 - 1e-16):
                t = residual_a_quantile(delta, w)
                self.assertLessEqual(t, delta / 2)
                self.assertAlmostEqual(_log_residual_mass(delta, t) - log_d, math.log(w),
                                       delta=1e-12, msg=f"delta={delta} w={w}")
            # w = 1 is clamped to 1 - 2^-53: finite, inside the support.
            top = residual_a_quantile(delta, 1.0)
            self.assertTrue(math.isfinite(top))
            self.assertLessEqual(top, delta / 2)
            # Monotone in w (an inverse CDF).
            ts = [residual_a_quantile(delta, w) for w in np.linspace(0.01, 0.99, 25)]
            self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))
            # B is A's mirror: tau_B(w) = -t_A(1-w), and it lies on B's side.
            self.assertEqual(residual_b_quantile_b_frame(delta, 0.3),
                             -residual_a_quantile(delta, 0.7))

    def test_small_delta_residual_is_rayleigh(self):
        # As delta -> 0, (phi(t)-phi(t-delta))/d -> Rayleigh in w = delta/2 - t.
        for v in (0.1, 0.5, 0.9):
            t = residual_a_quantile(1e-8, v)
            w = math.sqrt(-2 * math.log1p(-(1 - v)))  # Rayleigh survival = 1-v
            self.assertAlmostEqual(-t, w, places=6)

    def test_quadrature_and_log_ndtr_forms_agree_at_switch(self):
        for t in (-30.0, -3.0, -1.0, 0.0, 0.004):
            lo, hi = _log_residual_mass(0.01 * (1 - 1e-12), t), _log_residual_mass(0.01, t)
            self.assertAlmostEqual(lo, hi, delta=1e-9 * max(1.0, abs(hi)))

    def test_common_quantile_inverts_its_cdf(self):
        for delta in DELTAS:
            c = shared_mass_1d(delta)
            log_half_c = math.log(c / 2) if c > 0 else None
            for v in (1e-10, 0.2, 0.5, 0.7, 1 - 1e-10):
                t = common_quantile(delta, v)
                if t < delta / 2:  # left half: Phi(t - delta) / C
                    self.assertAlmostEqual(ndtr(t - delta) / c, v, delta=1e-9 * max(v, 1e-3))
                else:  # right half via survival: Phi(-t) / C = 1 - v
                    self.assertAlmostEqual(ndtr(-t) / c, 1 - v, delta=1e-9)
            self.assertIsNotNone(log_half_c)
            self.assertAlmostEqual(common_quantile(delta, 0.5), delta / 2, places=12)


class GaussianAuctionTests(unittest.TestCase):
    def test_tape_draw_is_deterministic_and_reuses_random_tape(self):
        t1 = GaussianTape.draw(np.random.default_rng(5), 7)
        t2 = GaussianTape.draw(np.random.default_rng(5), 7)
        self.assertIsInstance(t1.uniforms, RandomTape)
        self.assertEqual(t1.uniforms, t2.uniforms)
        np.testing.assert_array_equal(t1.z, t2.z)
        rng = np.random.default_rng(5)
        expected = RandomTape.draw(rng)
        np.testing.assert_array_equal(t1.z, rng.standard_normal(7))
        self.assertEqual(t1.uniforms, expected)
        with self.assertRaises(ValueError):
            t1.z[0] = 1.0  # read-only

    def test_marginal_along_e_matches_mixture(self):
        n = 4000
        for delta in (1e-3, 0.5, 3.0, 12.0, 40.0):
            auction = make_auction(delta)
            for a, b in ((1, 1), (3, 1), (1, 4)):
                lam = a / (a + b)
                rng = np.random.default_rng(1234)
                ts = []
                for _ in range(n):
                    out = auction.resolve(a, b, GaussianTape.draw(rng, auction.dim))
                    ts.append(auction.coordinate(out.x))
                ks = ks_distance(ts, lambda x: mixture_cdf_1d(x, auction.delta, lam))
                # 2.13/sqrt(n) is the KS 0.01% critical value (15 correlated checks share
                # one fixed seed, so the run is deterministic).
                self.assertLess(ks, KS_CRIT / math.sqrt(n), msg=f"delta={delta} bids={(a, b)}")

    def test_full_vector_is_gaussian_mixture_and_orthogonal_noise_is_shared(self):
        auction = make_auction(0.5, dim=4)
        rng = np.random.default_rng(99)
        xs, perps = [], []
        for _ in range(3000):
            tape = GaussianTape.draw(rng, auction.dim)
            outs = [auction.resolve(a, b, tape) for a, b in ((1, 0), (1, 1), (0, 1), (5, 2))]
            perp = [(o.x - auction.mu_a) / auction.s - auction.coordinate(o.x) * auction.e
                    for o in outs]
            for p in perp[1:]:
                np.testing.assert_allclose(p, perp[0], atol=1e-12)
            xs.append(outs[1].x)
            perps.append(perp[0])
        perps = np.array(perps)
        # Orthogonal component: standard normal in the D-1 dim complement of e.
        np.testing.assert_allclose(perps @ auction.e, 0.0, atol=1e-12)
        basis = np.linalg.svd(np.eye(4) - np.outer(auction.e, auction.e))[0][:, :3]
        coords = perps @ basis
        np.testing.assert_allclose(coords.mean(0), 0.0, atol=4 / math.sqrt(3000))
        np.testing.assert_allclose(np.cov(coords.T), np.eye(3), atol=0.1)
        for j in range(3):
            self.assertLess(ks_distance(coords[:, j], ndtr), KS_CRIT / math.sqrt(3000))
        # Mean of the (1,1) mixture is the midpoint.
        np.testing.assert_allclose(np.mean(xs, 0), (auction.mu_a + auction.mu_b) / 2,
                                   atol=4 * auction.s / math.sqrt(3000))

    def test_sole_bidder_gets_exact_own_gaussian(self):
        n = 4000
        for delta in (1e-3, 0.5, 3.0, 12.0, 40.0):
            auction = make_auction(delta)
            rng = np.random.default_rng(7)
            ta, tb = [], []
            for _ in range(n):
                tape = GaussianTape.draw(rng, auction.dim)
                out_a = auction.resolve(1, 0, tape)
                out_b = auction.resolve(0, 1, tape)
                self.assertEqual(out_a.payment_a + out_a.payment_b, 0.0)
                self.assertEqual(out_b.payment_a + out_b.payment_b, 0.0)
                self.assertIn(out_a.branch, ("shared", "A"))
                self.assertIn(out_b.branch, ("shared", "B"))
                ta.append(auction.coordinate(out_a.x))
                tb.append(auction.coordinate(out_b.x))
            self.assertLess(ks_distance(ta, ndtr), KS_CRIT / math.sqrt(n))
            self.assertLess(ks_distance(tb, lambda x: ndtr(x - auction.delta)),
                            KS_CRIT / math.sqrt(n))

    def test_candidates_lie_on_their_own_side(self):
        for delta in DELTAS[1:]:
            auction = make_auction(delta)
            rng = np.random.default_rng(3)
            seen = 0
            for _ in range(300):
                out = auction.resolve(1, 1, GaussianTape.draw(rng, auction.dim))
                if out.branch == "shared":
                    continue
                seen += 1
                self.assertLessEqual(auction.coordinate(out.candidate_a), delta / 2 + 1e-9)
                self.assertGreaterEqual(auction.coordinate(out.candidate_b), delta / 2 - 1e-9)
            if delta >= 0.5:
                self.assertGreater(seen, 0)

    def test_expected_payments_monte_carlo(self):
        for delta in (0.5, 3.0):
            auction = make_auction(delta, dim=3)
            for a, b in ((1, 1), (2, 0.5)):
                rng = np.random.default_rng(21)
                n = 20000
                pa = pb = 0.0
                for _ in range(n):
                    out = auction.resolve(a, b, GaussianTape.draw(rng, 3))
                    pa += out.payment_a
                    pb += out.payment_b
                ea, eb = auction.expected_payments(a, b)
                self.assertAlmostEqual(pa / n, ea, delta=0.06 * max(ea, 0.05))
                self.assertAlmostEqual(pb / n, eb, delta=0.06 * max(eb, 0.05))
        # Orthogonal limit (delta = 40, d = 1): log(2) - 1/2 each at bids (1, 1).
        far = make_auction(40.0)
        for value in far.expected_payments(1, 1):
            self.assertAlmostEqual(value, math.log(2) - 0.5, places=12)

    def test_fixed_tape_monotone_threshold_and_payments_match_auction_py(self):
        auction = make_auction(3.0)
        tape = GaussianTape(RandomTape(0.99, 0.2, 0.7, 0.4), np.linspace(-1, 1, auction.dim))
        threshold = 0.4 / 0.6  # bid_a at which u = a/(a+1)
        branches = []
        for a in np.linspace(0.01, 3, 300):
            out = auction.resolve(a, 1.0, tape)
            branches.append(out.branch)
            if out.branch == "A":
                np.testing.assert_array_equal(out.x, out.candidate_a)
                self.assertAlmostEqual(out.payment_a, threshold, places=12)
                self.assertLessEqual(out.payment_a, a)
            else:
                np.testing.assert_array_equal(out.x, out.candidate_b)
                self.assertAlmostEqual(out.payment_b, a * 0.6 / 0.4, places=12)
            self.assertEqual(a > threshold, out.branch == "A")
        switches = sum(x != y for x, y in zip(branches, branches[1:]))
        self.assertEqual(switches, 1)
        out = auction.resolve(1, 1, tape)  # MECHANISM.md fixture: A wins, pays 2/3
        self.assertEqual(out.branch, "A")
        self.assertAlmostEqual(out.payment_a, 2 / 3)

    def test_identical_means_and_zero_delta_always_shared(self):
        auction = make_auction(0.0)
        self.assertEqual((auction.shared_mass, auction.disagreement), (1.0, 0.0))
        rng = np.random.default_rng(4)
        for _ in range(50):
            tape = GaussianTape.draw(rng, auction.dim)
            out = auction.resolve(1, 40, tape)
            self.assertEqual(out.branch, "shared")
            np.testing.assert_array_equal(out.x, auction.mu_a + auction.s * tape.z)
        self.assertEqual(auction.expected_payments(1, 40), (0.0, 0.0))

    def test_tiny_delta_is_almost_always_shared(self):
        auction = make_auction(1e-8)
        # The vector difference carries ~1e-8 relative rounding; use the realized delta.
        self.assertAlmostEqual(auction.delta, 1e-8, delta=1e-15)
        self.assertAlmostEqual(auction.disagreement, auction.delta / math.sqrt(2 * math.pi),
                               delta=1e-22)
        out = auction.resolve(1, 1, GaussianTape(RandomTape(0.5, 0.3, 0.3, 0.5),
                                                 np.ones(auction.dim)))
        self.assertEqual(out.branch, "shared")
        # A contested draw at tiny delta still separates the candidates around delta/2.
        out = auction.resolve(1, 1, GaussianTape(RandomTape(1 - 1e-12, 0.3, 0.3, 0.5),
                                                 np.ones(auction.dim)))
        self.assertIn(out.branch, ("A", "B"))
        self.assertLess(auction.coordinate(out.candidate_a), 0)
        self.assertGreater(auction.coordinate(out.candidate_b), 1e-8)

    def test_large_delta_skips_shared_branch_when_mass_underflows(self):
        auction = make_auction(100.0)
        self.assertEqual(auction.shared_mass, 0.0)
        out = auction.resolve(1, 1, GaussianTape(RandomTape(0.0, 0.5, 0.5, 0.3),
                                                 np.zeros(auction.dim)))
        self.assertEqual(out.branch, "A")
        np.testing.assert_allclose(out.candidate_a, auction.mu_a, atol=1e-12)
        np.testing.assert_allclose(out.candidate_b, auction.mu_b, atol=1e-12)
        # delta = 40: C = 5.5e-89 > 0, so branch = 0.0 still selects the common density.
        near = make_auction(40.0)
        out = near.resolve(1, 1, GaussianTape(RandomTape(0.0, 0.5, 0.5, 0.3),
                                              np.zeros(near.dim)))
        self.assertEqual(out.branch, "shared")
        self.assertAlmostEqual(near.coordinate(out.x), 20.0, places=9)

    def test_point_masses(self):
        mu_a, mu_b = np.array([1.0, 2.0]), np.array([1.0, 3.0])
        tape = GaussianTape(RandomTape(0.1, 0.5, 0.5, 0.25), np.array([0.3, -0.2]))
        auction = GaussianTwoBidderAuction(mu_a, mu_b, 0.0)
        self.assertEqual((auction.shared_mass, auction.disagreement), (0.0, 1.0))
        self.assertTrue(math.isinf(auction.delta))
        out = auction.resolve(1, 1, tape)
        self.assertEqual(out.branch, "A")
        np.testing.assert_array_equal(out.x, mu_a)
        np.testing.assert_array_equal(out.candidate_b, mu_b)
        self.assertAlmostEqual(out.payment_a, 1 / 3)
        out = auction.resolve(1, 4, tape)
        self.assertEqual(out.branch, "B")
        np.testing.assert_array_equal(out.x, mu_b)
        same = GaussianTwoBidderAuction(mu_a, mu_a, 0.0)
        out = same.resolve(1, 1, tape)
        self.assertEqual((out.branch, out.payment_a, out.payment_b), ("shared", 0.0, 0.0))
        np.testing.assert_array_equal(out.x, mu_a)

    def test_high_dimensional_input(self):
        dim = 131072
        rng = np.random.default_rng(0)
        mu_a = rng.normal(size=dim)
        auction = GaussianTwoBidderAuction(mu_a, mu_a + 0.01 * rng.normal(size=dim), 0.5)
        self.assertGreater(auction.delta, 1.0)
        out = auction.resolve(1, 1, GaussianTape.draw(rng, dim))
        self.assertEqual(out.x.shape, (dim,))
        self.assertTrue(np.isfinite(out.x).all())

    def test_input_validation(self):
        auction = make_auction(1.0)
        tape = GaussianTape.draw(np.random.default_rng(0), auction.dim)
        with self.assertRaises(ValueError):
            auction.resolve(0, 0, tape)
        with self.assertRaises(ValueError):
            auction.resolve(-1, 1, tape)
        with self.assertRaises(ValueError):
            auction.resolve(1, 1, GaussianTape.draw(np.random.default_rng(0), auction.dim + 1))
        with self.assertRaises(ValueError):
            GaussianTwoBidderAuction([0.0], [1.0], -1.0)
        with self.assertRaises(ValueError):
            GaussianTwoBidderAuction([0.0], [1.0, 2.0], 1.0)
        with self.assertRaises(ValueError):
            GaussianTwoBidderAuction([math.nan], [1.0], 1.0)


if __name__ == "__main__":
    unittest.main()
