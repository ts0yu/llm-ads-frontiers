"""Exact two-bidder stable sampling for two equal-covariance isotropic Gaussians.

``p_A = N(mu_A, s^2 I)`` and ``p_B = N(mu_B, s^2 I)`` on R^D. This is the continuous
analogue of ``auction.TwoBidderAuction`` (see MOONSHOTS.md section 3, "Continuous
steps"). The maximal coupling of two equal-covariance Gaussians only acts along the
unit vector ``e = (mu_B - mu_A)/|mu_B - mu_A|``; the orthogonal noise is shared.

Coordinates: ``x = mu_A + s*(t*e + z_perp)``, with ``z_perp`` standard normal noise
projected orthogonal to ``e`` and ``t`` one-dimensional. Along ``e`` the problem is
``p_A = N(0, 1)`` versus ``p_B = N(delta, 1)`` with ``delta = |mu_B - mu_A|/s``:

- common density ``min(phi(t), phi(t - delta))``, mass ``C = 2*Phi(-delta/2)``;
- residual A ``(phi(t) - phi(t - delta))/d`` on ``t < delta/2``;
- residual B, its mirror image about ``delta/2``;
- disagreement ``d = 1 - C = TV(p_A, p_B) = 2*Phi(delta/2) - 1``.

Branch, residual and contest uniforms, the ``u < lambda`` contest, and the threshold
payments are exactly those of ``auction.py``; the residual/common draws use inverse
CDFs, so one tape gives one deterministic point per branch. Payments are simulated
credits. No claim is made about truthful bidding for arbitrary utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import erf, erfc, log_ndtr, ndtri_exp

from auction import RandomTape, TwoBidderAuction, _bids

FloatVector = NDArray[np.float64]

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_TINY = math.ulp(0.0)  # smallest positive double; clamps a zero uniform
# Below this delta, Phi(t) - Phi(t - delta) is evaluated by Gauss-Legendre quadrature
# of phi over [t - delta, t] (no cancellation). Above it the log-ndtr form is exact
# to a few ulps; both forms agree to ~1e-15 relative at the switch.
_SMALL_DELTA = 1e-2
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(12)
_GL_NODES = 0.5 * (_GL_NODES + 1.0)  # map [-1, 1] -> [0, 1]
_GL_LOG_WEIGHTS = np.log(0.5 * _GL_WEIGHTS)
# |t| beyond ~38.6 is unreachable from a double uniform (Phi(-38.6) < 2^-1074).
_T_CAP = 40.0


# ---------------------------------------------------------------------------
# One-dimensional pieces: p_A = N(0, 1), p_B = N(delta, 1).
# ---------------------------------------------------------------------------

def tv_distance(delta: float) -> float:
    """TV(N(0,1), N(delta,1)) = 2*Phi(delta/2) - 1 = erf(delta/(2*sqrt 2))."""
    return float(erf(delta / (2.0 * _SQRT2))) if math.isfinite(delta) else 1.0


def shared_mass_1d(delta: float) -> float:
    """Common mass C = 2*Phi(-delta/2) = erfc(delta/(2*sqrt 2)); underflows to 0."""
    return float(erfc(delta / (2.0 * _SQRT2))) if math.isfinite(delta) else 0.0


def _clamp_uniform(v: float) -> float:
    return min(max(float(v), _TINY), 1.0)


def common_quantile(delta: float, v: float) -> float:
    """Inverse CDF of the normalized common density, returned in the A frame (t).

    The common density is symmetric about delta/2. Its left half equals
    phi(t - delta), so CDF(t) = Phi(t - delta)/C there; with C = 2*Phi(-delta/2):
    t = delta + Phi^{-1}(2 v Phi(-delta/2)). The right half follows by symmetry.
    ``ndtri_exp`` keeps this exact even when C is far below the double range.
    """
    if not math.isfinite(delta):
        raise ValueError("The common branch has zero mass for point masses/infinite delta.")
    v = _clamp_uniform(v)
    log_half_c = float(log_ndtr(-0.5 * delta))  # log Phi(-delta/2)
    if v < 0.5:
        return delta + float(ndtri_exp(math.log(2.0 * v) + log_half_c))
    return -float(ndtri_exp(math.log(2.0 * (1.0 - v)) + log_half_c)) if v < 1.0 else math.inf


def _log_residual_mass(delta: float, t: float) -> float:
    """log(Phi(t) - Phi(t - delta)) for t <= delta/2: unnormalized residual-A CDF."""
    if delta < _SMALL_DELTA:
        # Phi(t) - Phi(t - delta) = delta * int_0^1 phi(t - delta + delta*s) ds.
        y = t - delta + delta * _GL_NODES
        terms = _GL_LOG_WEIGHTS - 0.5 * y * y
        peak = float(terms.max())
        return (math.log(delta) - _LOG_SQRT_2PI + peak
                + math.log(float(np.exp(terms - peak).sum())))
    hi = float(log_ndtr(t))
    lo = float(log_ndtr(t - delta)) if math.isfinite(delta) else -math.inf
    return hi + math.log1p(-math.exp(lo - hi))


def residual_a_quantile(delta: float, w: float) -> float:
    """Inverse CDF of residual A, ``(phi(t) - phi(t - delta))/d`` on ``t < delta/2``.

    Solves log(Phi(t) - Phi(t - delta)) = log(w) + log(d) with a bracketed Brent
    root search. ``log d`` is evaluated with the same function at t = delta/2, so
    the upper bracket is exactly non-negative. w is clamped to [2^-1074, 1 - 2^-53].
    """
    if not delta > 0.0:
        raise ValueError("Residual distributions need delta > 0.")
    w = min(_clamp_uniform(w), 1.0 - 2.0**-53)  # open interval: finite quantile
    half = 0.5 * delta
    hi = min(half, _T_CAP)
    log_d = _log_residual_mass(delta, half) if math.isfinite(half) else 0.0
    target = math.log(w) + log_d

    def f(t: float) -> float:
        return _log_residual_mass(delta, t) - target

    f_hi = f(hi)
    if f_hi < 0.0:  # only possible if the cap was too tight; fall back to delta/2
        hi = half
        f_hi = f(hi)
    if f_hi == 0.0:
        return hi
    step, lo = 1.0, hi - 1.0
    while f(lo) > 0.0:
        step *= 2.0
        lo = hi - step
        if step > 1e6:
            raise RuntimeError("Residual quantile bracket did not converge.")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=4.0 * np.finfo(float).eps, maxiter=500))


def residual_b_quantile_b_frame(delta: float, w: float) -> float:
    """Inverse CDF of residual B, in B's own frame (tau = t - delta, tau > -delta/2).

    Residual B is residual A mirrored about delta/2, so tau(w) = -t_A(1 - w).
    Returning B's frame avoids the cancellation in ``t - delta`` for large delta.
    """
    return -residual_a_quantile(delta, 1.0 - w)


def mixture_cdf_1d(t, delta: float, lam: float):
    """CDF of lam*N(0,1) + (1-lam)*N(delta,1) (test/analysis oracle)."""
    from scipy.special import ndtr

    t = np.asarray(t, dtype=np.float64)
    return lam * ndtr(t) + (1.0 - lam) * ndtr(t - delta)


# ---------------------------------------------------------------------------
# Tapes and allocations.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GaussianTape:
    """``auction.RandomTape`` uniforms plus one D-dim standard normal vector ``z``.

    ``draw(rng, dim)`` consumes ``RandomTape.draw(rng)`` (four uniforms) and then
    ``rng.standard_normal(dim)``, in that order. Reuse a tape for fixed-state
    counterfactual bids.
    """

    uniforms: RandomTape
    z: FloatVector

    def __post_init__(self) -> None:
        z = np.array(self.z, dtype=np.float64, copy=True)
        if z.ndim != 1 or len(z) == 0 or not np.isfinite(z).all():
            raise ValueError("z must be a nonempty finite one-dimensional vector.")
        z.flags.writeable = False
        object.__setattr__(self, "z", z)

    @property
    def branch(self) -> float:
        return self.uniforms.branch

    @property
    def token_a(self) -> float:
        return self.uniforms.token_a

    @property
    def token_b(self) -> float:
        return self.uniforms.token_b

    @property
    def contest(self) -> float:
        return self.uniforms.contest

    @classmethod
    def draw(cls, rng: np.random.Generator, dim: int) -> GaussianTape:
        uniforms = RandomTape.draw(rng)
        return cls(uniforms, rng.standard_normal(int(dim)))


Frame = Literal["A", "B"]


@dataclass(frozen=True, eq=False)
class GaussianDraw:
    """Bid-independent candidates for one tape.

    Each point is ``mu_<frame> + s*offset``. Shared points use A's frame; residual A
    uses A's frame and residual B uses B's frame (no cancellation for large delta).
    ``t_*`` are the coordinates along ``e`` in A's standardized frame (NaN for s = 0).
    """

    shared: bool
    offset_shared: FloatVector | None
    offset_a: FloatVector | None
    offset_b: FloatVector | None
    t_shared: float | None
    t_a: float | None
    t_b: float | None


@dataclass(frozen=True, eq=False)
class GaussianAllocation:
    x: FloatVector
    branch: Literal["shared", "A", "B"]
    payment_a: float
    payment_b: float
    candidate_a: FloatVector | None
    candidate_b: FloatVector | None
    frame: Frame
    offset: FloatVector
    t: float | None


class GaussianTwoBidderAuction:
    """One Gaussian step; resolve it at arbitrary nonnegative bid pairs.

    ``shared_mass`` is C and ``disagreement`` is d = TV, computed with erfc/erf so
    neither cancels. ``s = 0`` gives point masses: identical means are always
    shared; different means are fully contested (d = 1, delta = inf).
    """

    def __init__(self, mu_a, mu_b, s: float) -> None:
        self.mu_a = _vector(mu_a, "mu_a")
        self.mu_b = _vector(mu_b, "mu_b")
        if self.mu_a.shape != self.mu_b.shape:
            raise ValueError("Means must have the same shape.")
        s = float(s)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("The standard deviation must be finite and nonnegative.")
        self.s = s
        self.dim = len(self.mu_a)
        diff = self.mu_b - self.mu_a
        self.distance = float(np.linalg.norm(diff))
        if self.distance == 0.0:
            self.delta = 0.0
            self.e = None
        else:
            self.e = diff / self.distance
            self.e.flags.writeable = False
            self.delta = self.distance / s if s > 0.0 else math.inf
        self.disagreement = tv_distance(self.delta)
        self.shared_mass = shared_mass_1d(self.delta)

    # -- bid-independent sampling -------------------------------------------------
    def _offset(self, t: float, z: FloatVector) -> FloatVector:
        z_perp = z - float(z @ self.e) * self.e
        return t * self.e + z_perp

    def draw(self, tape: GaussianTape) -> GaussianDraw:
        if len(tape.z) != self.dim:
            raise ValueError(f"Tape dimension {len(tape.z)} != {self.dim}.")
        if tape.branch < self.shared_mass or self.disagreement == 0.0:
            if self.s == 0.0:
                return GaussianDraw(True, np.zeros(self.dim), None, None, math.nan, None, None)
            if self.e is None:  # identical means: the whole tape noise is shared
                return GaussianDraw(True, tape.z.copy(), None, None, None, None, None)
            t = common_quantile(self.delta, tape.token_a)
            return GaussianDraw(True, self._offset(t, tape.z), None, None, t, None, None)
        if self.s == 0.0:
            zero = np.zeros(self.dim)
            return GaussianDraw(False, None, zero, zero.copy(), None, math.nan, math.nan)
        t_a = residual_a_quantile(self.delta, tape.token_a)
        tau_b = residual_b_quantile_b_frame(self.delta, tape.token_b)
        return GaussianDraw(False, None, self._offset(t_a, tape.z), self._offset(tau_b, tape.z),
                            None, t_a, self.delta + tau_b)

    def point(self, frame: Frame, offset: FloatVector) -> FloatVector:
        mu = self.mu_a if frame == "A" else self.mu_b
        return mu + self.s * offset

    # -- auction -----------------------------------------------------------------
    def resolve(self, bid_a: float, bid_b: float, tape: GaussianTape) -> GaussianAllocation:
        a, b, total = _bids(bid_a, bid_b)
        draw = self.draw(tape)
        if draw.shared:
            x = self.point("A", draw.offset_shared)
            return GaussianAllocation(x, "shared", 0.0, 0.0, None, None, "A",
                                      draw.offset_shared, draw.t_shared)
        cand_a = self.point("A", draw.offset_a)
        cand_b = self.point("B", draw.offset_b)
        u = tape.contest
        if u < a / total:
            # Critical A bid: solve u = a/(a+b) for a, holding b fixed.
            price = b * (u / (1.0 - u))
            return GaussianAllocation(cand_a, "A", price, 0.0, cand_a, cand_b, "A",
                                      draw.offset_a, draw.t_a)
        # Critical B bid: solve u = a/(a+b) for b, holding a fixed.
        price = a * ((1.0 - u) / u)
        return GaussianAllocation(cand_b, "B", 0.0, price, cand_a, cand_b, "B",
                                  draw.offset_b, draw.t_b)

    def expected_payments(self, bid_a: float, bid_b: float) -> tuple[float, float]:
        """Same closed form as ``auction.TwoBidderAuction`` with d = TV of the Gaussians.

        The reference implementation reads only ``self.disagreement``, so it is reused
        verbatim rather than re-derived.
        """
        return TwoBidderAuction.expected_payments(self, bid_a, bid_b)

    def coordinate(self, x: FloatVector) -> float:
        """Standardized coordinate of x along e, in A's frame (analysis helper)."""
        if self.e is None or self.s == 0.0:
            raise ValueError("No contest direction (identical means or s = 0).")
        return float((np.asarray(x, dtype=np.float64) - self.mu_a) @ self.e) / self.s


def _vector(values, name: str) -> FloatVector:
    v = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    if len(v) == 0 or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    v.flags.writeable = False
    return v
