"""N-bidder "exponential race" stable sampler: research extension, not the demo.

The Milestone 1 demo stays two-bidder (`auction.py`). This module generalizes
the same common-mass coupling to n >= 2 advertiser distributions over ONE
vocabulary, for fixed-prefix research (simplex atlases). Simulated credits only.

Construction
------------
Inputs: distributions p_1..p_n over the same token IDs, bids b_i >= 0, not all
zero. Let c = min_i p_i (tokenwise), C = sum(c), d = 1 - C. Every residual
r_i = (p_i - c)/d sums to one, because sum(p_i - c) = 1 - C for each i.

Tape (`RaceTape`, deterministic `draw(rng, n)`): one branch uniform, one
residual uniform per bidder, one race uniform U_i in (0,1) per bidder with
E_i = -log(U_i) ~ Exp(1), and one shared-token uniform.

- Shared branch (branch < C, or all p_i identical): emit a token from c/C.
  Bids are irrelevant and nobody pays.
- Contested branch: candidate_i = inverse-CDF pick from r_i. Among bidders with
  b_i > 0, the winner is w = argmin E_i/b_i; emit candidate_w.

Since E_i/b_i ~ Exp(rate b_i) independently, P(w = i) = b_i/B with B = sum(b).
The marginal is C*(c/C) + d*sum_i (b_i/B) r_i = sum_i (b_i/B) p_i exactly.

Payments (critical bids)
------------------------
Fix the tape and the rival bids. Let R = min_{j != w, b_j > 0} E_j/b_j, the best
rival score. Bidder w wins iff E_w/b_w < R, i.e. iff b_w > E_w/R. The critical
bid, charged to the winner, is

    pay_w = E_w / R = E_w * max_{j != w, b_j > 0} b_j/E_j,

zero if w is the only positive bidder. Others pay zero.

NOTE: the critical bid uses the MAXIMUM of b_j/E_j over rivals (equivalently the
minimum rival score E_j/b_j). Writing `min_j b_j/E_j` coincides for n = 2 (one
rival) but is wrong for n >= 3: it is not a threshold, and tests that bisect the
winning threshold would reject it.

Raising b_i only lowers E_i/b_i, so it can only switch the winner TO i; the
ordering among the other bidders is unchanged. Hence on a fixed tape the
outcome changes, if at all, only to candidate_i, at the single threshold above.

Caveat for n >= 3: residual supports need not be disjoint (if p_1[t] = p_2[t] >
p_3[t], token t is in supp r_1 and supp r_2). Two bidders can share a
candidate, so a winner can be charged its race-critical bid even though a
rival win would have emitted the same token. The charge is the critical bid for
winning the race, not for changing the emitted token. For n = 2 the supports
are disjoint and the two notions coincide.

Expected payment (test oracle)
------------------------------
The contested branch has probability d and is independent of the race, so

    E[pay_i] = d * E[ pay_i * 1{i wins} | contested ]
             = d * E_R[ E_x[ (x/R) * 1{x < b_i R} ] ],      x = E_i ~ Exp(1),

where R = min_{j != i, b_j > 0} E_j/b_j ~ Exp(rate B_-i), B_-i = sum_{j != i} b_j.
With g(a) = E[x 1{x < a}] = 1 - (1 + a) e^{-a}:

    E[pay_i] = d * integral_0^inf B_-i e^{-B_-i r} g(b_i r)/r dr      (1)
             = d * B_-i * [ log(1 + b_i/B_-i) - b_i/(b_i + B_-i) ]     (2)

using a Frullani integral for the e^{-B r}(1 - e^{-b_i r})/r part. (2) is the
two-bidder formula of MECHANISM.md with the rival bid replaced by the SUM of
rival bids: from bidder i's side, the rivals' race is one exponential clock of
rate B_-i. `expected_payment_quadrature` evaluates (1) numerically (method
"reduced"), or integrates over every rival exponential explicitly without the
min-of-exponentials reduction (method "race"); `expected_payments` is (2).

Exact n = 2 reduction
---------------------
`RaceTape.from_two_bidder(tape)` sets E_A = s*u, E_B = s*(1 - u) for u =
tape.contest and any s > 0 (only ratios matter), and reuses branch, token_a,
token_b; the shared-token uniform is token_a, which is what `auction.py` uses
for the shared branch. Then E_A/a < E_B/b iff u < a/(a + b), and the critical
bids are b*u/(1-u) and a*(1-u)/u: `auction.TwoBidderAuction.resolve` exactly,
up to rounding of exp/log (ties are probability zero).

Nonclaims: no truthful-bidding claim over complete text, no production ledger,
no >1 tokenizer. Floating point is not a payment ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from auction import RandomTape, _cdf, _distribution, _pick

FloatVector = np.ndarray

# Any positive scale gives an exact reduction; a larger one keeps -log(exp(-E))
# relatively precise when the contest uniform is small.
TWO_BIDDER_SCALE = 32.0


def _bids(bids: Sequence[float], n: int) -> tuple[float, ...]:
    values = tuple(float(b) for b in bids)
    if len(values) != n:
        raise ValueError(f"Expected {n} bids, got {len(values)}.")
    if not all(math.isfinite(b) and b >= 0.0 for b in values):
        raise ValueError("Bids must be finite and nonnegative.")
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("Bid sum overflow; rescale the credit unit.")
    if total == 0.0:
        raise ValueError("All-zero bids: use the separate no-ad path, not this auction.")
    return values


@dataclass(frozen=True)
class RaceTape:
    """Independent uniforms for one step. E_i = -log(race[i]) ~ Exp(1)."""

    branch: float
    tokens: tuple[float, ...]
    race: tuple[float, ...]
    shared: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(float(x) for x in self.tokens))
        object.__setattr__(self, "race", tuple(float(x) for x in self.race))
        if len(self.tokens) != len(self.race) or len(self.race) < 2:
            raise ValueError("Need one residual and one race uniform per bidder, n >= 2.")
        if not all(math.isfinite(x) and 0.0 <= x < 1.0
                   for x in (self.branch, self.shared, *self.tokens)):
            raise ValueError("Branch/token uniforms must lie in [0, 1).")
        if not all(math.isfinite(x) and 0.0 < x < 1.0 for x in self.race):
            raise ValueError("Race uniforms must lie strictly between zero and one.")

    @property
    def n(self) -> int:
        return len(self.race)

    @property
    def exponentials(self) -> tuple[float, ...]:
        return tuple(-math.log(u) for u in self.race)

    @classmethod
    def draw(cls, rng: np.random.Generator, n: int) -> RaceTape:
        """Consume exactly 2n+2 uniforms: branch, n residual, n race, shared."""
        if n < 2:
            raise ValueError("Need at least two bidders.")
        values = rng.random(2 * n + 2)
        race = values[1 + n : 1 + 2 * n]
        race[race == 0.0] = np.nextafter(0.0, 1.0)
        return cls(
            float(values[0]),
            tuple(float(x) for x in values[1 : 1 + n]),
            tuple(float(x) for x in race),
            float(values[-1]),
        )

    @classmethod
    def from_exponentials(cls, branch, tokens, exponentials, shared) -> RaceTape:
        race = []
        for e in exponentials:
            if not math.isfinite(e) or e <= 0.0:
                raise ValueError("Exponentials must be finite and positive.")
            u = math.exp(-e)
            race.append(min(max(u, np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0)))
        return cls(branch, tuple(tokens), tuple(race), shared)

    @classmethod
    def from_two_bidder(cls, tape: RandomTape, scale: float = TWO_BIDDER_SCALE) -> RaceTape:
        """Map an `auction.RandomTape` to the race with E_A/(E_A+E_B) = tape.contest."""
        u = tape.contest
        return cls.from_exponentials(
            tape.branch, (tape.token_a, tape.token_b), (scale * u, scale * (1.0 - u)),
            tape.token_a,
        )


def race_winner(exponentials: Sequence[float], bids: Sequence[float]) -> int:
    """argmin E_i/b_i over positive bidders (exact ties: lowest index, measure zero)."""
    best, winner = math.inf, -1
    for i, (e, b) in enumerate(zip(exponentials, bids)):
        if b > 0.0:
            score = e / b
            if score < best or winner < 0:
                best, winner = score, i
    if winner < 0:
        raise ValueError("No positive bid.")
    return winner


def critical_bid(exponentials: Sequence[float], bids: Sequence[float], winner: int) -> float:
    """Infimum winning bid for `winner`, rivals fixed: E_w * max_{j!=w, b_j>0} b_j/E_j."""
    e_w = exponentials[winner]
    rivals = [b * (e_w / e) for j, (e, b) in enumerate(zip(exponentials, bids))
              if j != winner and b > 0.0]
    return max(rivals) if rivals else 0.0


@dataclass(frozen=True)
class RaceAllocation:
    token_id: int
    branch: Literal["shared", "contested"]
    winner: int | None
    payments: tuple[float, ...]
    candidates: tuple[int, ...] | None


def _g(a: float) -> float:
    """E[x 1{x < a}] for x ~ Exp(1): 1 - (1+a)e^{-a}, series for small a."""
    if a < 0.05:
        return sum(((-1.0) ** k) * (k - 1) * a**k / math.factorial(k) for k in range(2, 12))
    return -math.expm1(-a) - a * math.exp(-a)


class NBidderAuction:
    """Prepare one token step for n bidders; resolve it at arbitrary bid vectors."""

    def __init__(self, distributions: Sequence[Sequence[float]]) -> None:
        ps = [_distribution(p) for p in distributions]
        if len(ps) < 2:
            raise ValueError("Need at least two distributions.")
        if any(p.shape != ps[0].shape for p in ps):
            raise ValueError("Distributions must use the same vocabulary shape.")
        self.n = len(ps)
        self.distributions = ps
        common = np.minimum.reduce(np.stack(ps), axis=0)
        self.shared_mass = min(1.0, max(0.0, float(common.sum())))
        # d == 0 iff all distributions are identical (then every p_i equals c).
        self.contested_possible = any(not np.array_equal(p, common) for p in ps)
        self.residual_masses = tuple(float((p - common).sum()) for p in ps)
        self.disagreement = float(np.mean(self.residual_masses)) if self.contested_possible else 0.0
        self._shared_cdf = _cdf(common)
        self._cdfs = [_cdf(p - common) for p in ps]

    def mixture(self, bids: Sequence[float]) -> FloatVector:
        b = _bids(bids, self.n)
        total = math.fsum(b)
        return sum((bi / total) * p for bi, p in zip(b, self.distributions))

    def is_shared(self, tape: RaceTape) -> bool:
        return tape.branch < self.shared_mass or not self.contested_possible

    def shared_token(self, tape: RaceTape) -> int:
        return _pick(self._shared_cdf, tape.shared)

    def candidates(self, tape: RaceTape) -> tuple[int, ...]:
        return tuple(_pick(cdf, u) for cdf, u in zip(self._cdfs, tape.tokens))

    def resolve(self, bids: Sequence[float], tape: RaceTape) -> RaceAllocation:
        b = _bids(bids, self.n)
        if tape.n != self.n:
            raise ValueError("Tape and auction have different bidder counts.")
        zeros = (0.0,) * self.n
        if self.is_shared(tape):
            return RaceAllocation(self.shared_token(tape), "shared", None, zeros, None)
        candidates = self.candidates(tape)
        exponentials = tape.exponentials
        winner = race_winner(exponentials, b)
        payments = list(zeros)
        payments[winner] = critical_bid(exponentials, b, winner)
        return RaceAllocation(candidates[winner], "contested", winner, tuple(payments),
                              candidates)

    def expected_payments(self, bids: Sequence[float]) -> tuple[float, ...]:
        """Closed form (2): d*B_-i*[log(1+b_i/B_-i) - b_i/(b_i+B_-i)]."""
        b = _bids(bids, self.n)
        result = []
        for i, own in enumerate(b):
            rival = math.fsum(x for j, x in enumerate(b) if j != i)
            if own == 0.0 or rival == 0.0 or self.disagreement == 0.0:
                result.append(0.0)
                continue
            ratio = own / rival
            if ratio < 1e-4:
                factor = sum(((-1.0) ** k) * ((k - 1.0) / k) * ratio**k for k in range(2, 9))
            else:
                factor = math.log1p(ratio) - own / (own + rival)
            result.append(self.disagreement * rival * max(0.0, factor))
        return tuple(result)

    def expected_payment_quadrature(
        self, bids: Sequence[float], bidder: int, method: str = "reduced"
    ) -> float:
        """Numerical E[pay_i] = d * E[critical bid * 1{i wins}] (see module docstring).

        "reduced": 1-D quadrature of (1) over the best rival score R.
        "race": explicit quadrature over each positive rival's E_j (uniform
        coordinates U_j in (0,1)), with the Exp(1) integral over E_i in closed
        form g(b_i R)/R. Practical for up to three positive rivals.
        """
        from scipy import integrate

        b = _bids(bids, self.n)
        own = b[bidder]
        rivals = [x for j, x in enumerate(b) if j != bidder and x > 0.0]
        if own == 0.0 or not rivals or self.disagreement == 0.0:
            return 0.0
        if method == "reduced":
            rate = math.fsum(rivals)

            def integrand(r: float) -> float:
                return 0.0 if r <= 0.0 else rate * math.exp(-rate * r) * _g(own * r) / r

            scale = 1.0 / rate
            value = 0.0
            # Split at multiples of the rival clock's mean for robust convergence.
            edges = [0.0, 0.1 * scale, scale, 10.0 * scale, 60.0 * scale]
            if own > 0:
                edges = sorted(set(edges + [min(60 * scale, 1.0 / own)]))
            for lo, hi in zip(edges, edges[1:]):
                value += integrate.quad(integrand, lo, hi, epsabs=1e-14, epsrel=1e-12,
                                        limit=200)[0]
            return self.disagreement * value
        if method == "race":
            if len(rivals) > 3:
                raise ValueError("Race quadrature is limited to three positive rivals.")

            def integrand(*us: float) -> float:
                r = min(-math.log(u) / bj for u, bj in zip(us, rivals))
                return _g(own * r) / r if r > 0 else 0.0

            def level_options(k: int):
                # min_j E_j/b_j has a kink in U_k where E_k/b_k = E_j/b_j for each
                # outer coordinate j > k, i.e. U_k = U_j^(b_k/b_j). Tell QUADPACK.
                def options(*outer: float) -> dict:
                    points = sorted({u ** (rivals[k] / rivals[k + 1 + m])
                                     for m, u in enumerate(outer) if 0.0 < u < 1.0})
                    result = {"epsabs": 1e-11, "epsrel": 1e-9, "limit": 200}
                    if points:
                        result["points"] = points
                    return result
                return options

            value = integrate.nquad(integrand, [(0.0, 1.0)] * len(rivals),
                                    opts=[level_options(k) for k in range(len(rivals))])[0]
            return self.disagreement * value
        raise ValueError("method must be 'reduced' or 'race'.")
