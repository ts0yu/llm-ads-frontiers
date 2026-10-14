"""Keyed sponsor watermark for the token auction (research, opt-in).

Goal: from the token IDs of an answer alone (no model, no prompts), let a holder of
the keys test which sponsor's tokens it contains. Docs: docs/WATERMARK.md.

Scheme keyed-exp-v1. Each role has a key: the platform (shared tokens), sponsor A
and sponsor B. At every step the auction decides the branch exactly as the reference
does (``TwoBidderAuction.resolve`` on the ordinary tape: shared or contested, the
contest uniform ``u``, the payments). Only the *token draws* change. A token that
the reference draws from distribution ``q`` with an inverse-CDF uniform is instead
drawn with the exponential race

    token = argmin_v  -ln(r_v) / q_v        (over v with q_v > 0),

where ``r = keyed_uniforms(key_of_role, previous CONTEXT_TOKENS generated tokens)``.
If ``r`` is uniform this picks ``v`` with probability exactly ``q_v`` (the minimum of
independent exponentials ``E_v / q_v``), so averaged over keys the sampler is the
reference sampler: probabilities are sampled, nothing is filtered or reweighted.
The argmin is over random keyed variables, not over probabilities.

Roles: the shared distribution uses the platform key; A's residual (when A's
candidate is drawn) uses A's key; B's uses B's key. Under span hold the owner's
proposals use the owner's key. The emitted token carries a high ``r`` under the
key that drew it and a uniform ``r`` under every other key.

Detection: score each generated token with ``-ln(1 - r)`` under a key, skipping
repeated (context, token) pairs. Without that key's influence the scores are
independent Exp(1), so their sum over n tokens is Gamma(n, 1) and gives an exact
p-value. Power depends on how many tokens the role drew and their entropy;
nothing here claims robustness to paraphrase or editing.

Keys are secrets in any real use. ``demo_keys`` derives PUBLIC demo keys from fixed
strings so that results can be reproduced; they must never be used for anything
but this research demo. Manifests record key fingerprints, never keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from auction import Allocation, RandomTape, TwoBidderAuction

SCHEME = "keyed-exp-v1"
CONTEXT_TOKENS = 4
ROLES = ("platform", "A", "B")
DEMO_SOURCE = "public-demo-v1"


@dataclass(frozen=True)
class WatermarkKeys:
    platform: bytes
    a: bytes
    b: bytes
    source: str = "private"

    def key(self, role: str) -> bytes:
        return {"platform": self.platform, "A": self.a, "B": self.b}[role]

    def fingerprints(self) -> dict[str, str]:
        return {role: hashlib.sha256(b"fingerprint|" + self.key(role)).hexdigest()[:16]
                for role in ROLES}

    def manifest(self) -> dict:
        return {
            "scheme": SCHEME,
            "context_tokens": CONTEXT_TOKENS,
            "key_source": self.source,
            "key_fingerprints": self.fingerprints(),
            "note": "Token draws use keyed exponential races; branch, contest uniform and "
            "payments are the reference auction's. Keys are not stored here.",
        }


def demo_keys(bidders: Sequence[str]) -> WatermarkKeys:
    """PUBLIC research keys, derived from fixed strings. Not secret; demo use only."""

    def derive(role: str) -> bytes:
        return hashlib.sha256(
            f"token-auction-lab public demo watermark key v1|{role}".encode()
        ).digest()

    return WatermarkKeys(
        derive("platform"), derive(f"A|{bidders[0]}"), derive(f"B|{bidders[1]}"), DEMO_SOURCE
    )


def keys_for_manifest(manifest: dict, bidders: Sequence[str],
                      keys: WatermarkKeys | None = None) -> WatermarkKeys:
    """The keys an atlas was built with: demo keys are rebuilt, others must be passed."""
    spec = manifest["watermark"]
    if keys is None:
        if spec["key_source"] != DEMO_SOURCE:
            raise ValueError("This atlas used private keys; pass them explicitly.")
        keys = demo_keys(bidders)
    if keys.fingerprints() != spec["key_fingerprints"]:
        raise ValueError("Watermark key fingerprints do not match the atlas manifest.")
    return keys


# ----------------------------------------------------------------------------- keyed draws


def _context_seed(key: bytes, context: Sequence[int]) -> int:
    window = np.asarray(list(context)[-CONTEXT_TOKENS:], dtype="<i8").tobytes()
    digest = hashlib.blake2b(key + b"|ctx|" + window, digest_size=16).digest()
    return int.from_bytes(digest, "little")


def keyed_uniforms(key: bytes, context: Sequence[int], size: int) -> np.ndarray:
    """``size`` uniforms in (0, 1) from Philox keyed by (key, last generated tokens).

    Raw 64-bit Philox outputs are converted here (53-bit mantissa, half-offset) so the
    values do not depend on numpy's float-generation helpers.
    """
    generator = np.random.Philox(key=_context_seed(key, context))
    raw = generator.random_raw(size).astype(np.uint64)
    return ((raw >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53


def keyed_uniform(key: bytes, context: Sequence[int], token_id: int) -> float:
    return float(keyed_uniforms(key, context, token_id + 1)[token_id])


def exp_sample(q: np.ndarray, r: np.ndarray) -> int:
    """argmin over q_v > 0 of -ln(r_v)/q_v: an exact draw from q when r is uniform."""
    q = np.asarray(q, dtype=np.float64)
    support = np.flatnonzero(q > 0.0)
    if len(support) == 0:
        raise RuntimeError("Attempted to sample zero probability mass.")
    times = -np.log(r[support]) / q[support]
    return int(support[int(np.argmin(times))])


def keyed_draw(weights: np.ndarray, key: bytes, context: Sequence[int]) -> int:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("Attempted to sample zero probability mass.")
    return exp_sample(weights / total, keyed_uniforms(key, context, len(weights)))


def keyed_candidates(auction: TwoBidderAuction, keys: WatermarkKeys,
                     context: Sequence[int]) -> dict[str, int | None]:
    """Keyed draws from the shared part and from each residual. Independent of bids."""
    common = np.minimum(auction.p_a, auction.p_b)
    residual_a, residual_b = auction.p_a - common, auction.p_b - common
    return {
        "platform": keyed_draw(common, keys.platform, context) if common.sum() > 0 else None,
        "A": keyed_draw(residual_a, keys.a, context) if residual_a.sum() > 0 else None,
        "B": keyed_draw(residual_b, keys.b, context) if residual_b.sum() > 0 else None,
    }


def resolve(auction: TwoBidderAuction, bid_a: float, bid_b: float, tape: RandomTape,
            keys: WatermarkKeys, context: Sequence[int]) -> tuple[Allocation, str]:
    """Reference branch and payments; keyed token. Returns (allocation, drawing role)."""
    reference = auction.resolve(bid_a, bid_b, tape)
    picks = keyed_candidates(auction, keys, context)
    if reference.branch == "shared":
        return replace(reference, token_id=picks["platform"]), "platform"
    token = picks["A"] if reference.branch == "A" else picks["B"]
    return replace(reference, token_id=token, candidate_a=picks["A"],
                   candidate_b=picks["B"]), reference.branch


# ----------------------------------------------------------------------------- detection


def gamma_upper_tail(n: int, s: float) -> float:
    """P(Gamma(n, 1) >= s) = exp(-s) * sum_{k<n} s^k / k!, computed in log space."""
    if n <= 0:
        return 1.0
    if s <= 0:
        return 1.0
    terms = [k * math.log(s) - math.lgamma(k + 1) for k in range(n)]
    top = max(terms)
    log_tail = -s + top + math.log(sum(math.exp(t - top) for t in terms))
    return min(1.0, math.exp(log_tail))


def token_scores(token_ids: Sequence[int], key: bytes) -> list[float | None]:
    """-ln(1 - r) per generated token under ``key``; None for repeated (context, token)."""
    seen = set()
    scores: list[float | None] = []
    for t, token in enumerate(token_ids):
        context = tuple(token_ids[max(0, t - CONTEXT_TOKENS):t])
        if (context, token) in seen:
            scores.append(None)
            continue
        seen.add((context, token))
        scores.append(-math.log1p(-keyed_uniform(key, context, int(token))))
    return scores


def detect(token_ids: Sequence[int], keys: WatermarkKeys) -> dict:
    """Per-role test: does this text lean on that role's key? Exact Gamma p-values."""
    result = {}
    for role in ROLES:
        scores = token_scores(token_ids, keys.key(role))
        used = [s for s in scores if s is not None]
        total = math.fsum(used)
        n = len(used)
        result[role] = {
            "tokens_scored": n,
            "score": total,
            "mean_score": total / n if n else None,
            "p_value": gamma_upper_tail(n, total),
            "token_scores": scores,
        }
    return result


# ----------------------------------------------------------------------------- CLI


def _atlases(paths):
    for raw in paths:
        path = Path(raw)
        for file in [path] if path.is_file() else sorted(path.rglob("*.json")):
            try:
                data = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("schema") == "bid-atlas-v1" \
                    and data.get("kind") == "text":
                yield file, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect sponsor watermarks in stored text atlases, leaf by leaf. "
        "Uses only token IDs and the keys; demo keys are rebuilt from the bidders."
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = []
    for file, atlas in _atlases(args.paths):
        keys = demo_keys(atlas["bidders"])
        watermarked = "watermark" in atlas["manifest"]
        for leaf in atlas["leaves"]:
            found = detect(leaf["token_ids"], keys)
            truth = None
            if watermarked:
                roles = [s.get("key") for s in leaf["steps"]]
                truth = {role: roles.count(role) for role in ROLES}
            report.append({
                "file": f"{file.parent.name}/{file.name}",
                "watermarked": watermarked,
                "seed": atlas["seed"],
                "leaf": leaf["id"],
                "lo": leaf["lo"],
                "hi": leaf["hi"],
                "tokens_drawn_by_role": truth,
                "p_value": {role: found[role]["p_value"] for role in ROLES},
                "mean_score": {role: found[role]["mean_score"] for role in ROLES},
                "detected": {role: found[role]["p_value"] < args.alpha for role in ROLES},
            })
    if not report:
        print("no text atlases found", file=sys.stderr)
        return 1
    for row in report:
        flags = " ".join(f"{r}:{row['p_value'][r]:.1e}" for r in ROLES)
        print(f"{row['file'][-38:]:38} wm={int(row['watermarked'])} "
              f"({row['lo']:.3f},{row['hi']:.3f}] {flags}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
