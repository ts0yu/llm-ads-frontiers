"""Replay attribution for the keyed sponsor watermark (needs keys, model and prompts).

Text-only detection (``watermark.detect``) scores every token under each key, so a
sponsor's few keyed tokens are diluted by all the others. An auditor who also holds
the model and the two advertiser prompts (the platform does) can do much better:
replay the answer and, at every step, recompute each role's keyed draws and check
whether the emitted token equals one of them.

For role R in {A, B} the keyed draws at a step are: R's residual (when R wins a
contest), R's full distribution (span-hold proposals) and, under floor keying, the
shared part. A residual draw of A can never equal a token of B's residual (the two
residuals have disjoint supports), so B's tokens do not dilute A's evidence.

Test. For text not drawn with R's key, the keyed draws at a step are draws with an
unrelated key, so the step matches with probability at most
``p_t = sum over R's keyed distributions q of q(token_t)`` (a union bound). The
number of matches is then stochastically at most Poisson-binomial(p_1..p_T); its
upper tail is computed exactly by dynamic programming and is a valid (conservative)
p-value. Per-token matches give a map of which role's key could have drawn each
token. Simulated credits; synthetic and live labels as in the atlases.

Near-certain tokens (``p_t`` close to 1) match under both hypotheses and carry no
evidence, so the reported test counts matches only at *informative* steps,
``p_t <= INFORMATIVE`` (fixed in advance at 0.5). Which steps are informative
depends only on the model and the text, not on the key, so the test stays exact.
The count over all steps is also reported.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from auction import TwoBidderAuction
from token_auction_lab import watermark as wm
from token_auction_lab.decoder import SharedPrefixDecoder

INFORMATIVE = 0.5


def poisson_binomial_upper_tail(probabilities: Sequence[float], k: int) -> float:
    """P(sum of independent Bernoulli(p_t) >= k), exact by dynamic programming."""
    if k <= 0:
        return 1.0
    dist = np.zeros(len(probabilities) + 1)
    dist[0] = 1.0
    for n, p in enumerate(probabilities, start=1):
        p = min(1.0, max(0.0, float(p)))
        dist[1:n + 1] = dist[1:n + 1] * (1 - p) + dist[:n] * p
        dist[0] *= 1 - p
    return float(min(1.0, dist[k:].sum()))


def _keyed_sources(auction: TwoBidderAuction, role: str, keys: wm.WatermarkKeys,
                   span_hold: bool) -> list[np.ndarray]:
    """The distributions a role's key draws from at one step."""
    common = np.minimum(auction.p_a, auction.p_b)
    own = auction.p_a if role == "A" else auction.p_b
    sources = [own - common]  # residual
    if span_hold:
        sources.append(own)  # span-hold proposals
    if keys.shared == "floor" or role == "platform":
        sources.append(common)  # shared tokens (floor keying, or the platform role)
    return [s for s in sources if s.sum() > 0]


@torch.inference_mode()
def attribute(
    model,
    prompts: list[list[int]],
    token_ids: Sequence[int],
    keys: wm.WatermarkKeys,
    *,
    temperature: float,
    span_hold: bool,
) -> dict:
    """Per-role match counts, null bounds, p-values and a per-token match map.

    Repeated (context, token) pairs are counted once: their keyed draws repeat, so
    they are not independent evidence.
    """
    decoder = SharedPrefixDecoder(model, [list(p) for p in prompts])
    matches = {role: [] for role in wm.ROLES}
    null_p = {role: [] for role in wm.ROLES}
    steps = {role: [] for role in wm.ROLES}
    seen = set()
    for t, token in enumerate(token_ids):
        token = int(token)
        p_a, p_b = decoder.distributions(temperature)
        context = list(decoder.generated)
        pair = (tuple(context[-wm.CONTEXT_TOKENS:]), token)
        if pair not in seen:
            seen.add(pair)
            auction = TwoBidderAuction(p_a, p_b)
            for role in wm.ROLES:
                if role == "platform":
                    sources = [np.minimum(auction.p_a, auction.p_b)]
                else:
                    sources = _keyed_sources(auction, role, keys, span_hold)
                hit, bound = False, 0.0
                for q in sources:
                    bound += float(q[token]) / float(q.sum())
                    hit = hit or wm.keyed_draw(q, keys.key(role), context) == token
                matches[role].append(hit)
                null_p[role].append(min(1.0, bound))
                if hit:
                    steps[role].append(t)
        decoder.advance(token)
    result = {"tokens": len(token_ids), "scored": len(seen), "informative_threshold":
              INFORMATIVE, "roles": {}}
    for role in wm.ROLES:
        k = sum(matches[role])
        informative = [(m, q) for m, q in zip(matches[role], null_p[role]) if q <= INFORMATIVE]
        k_inf = sum(m for m, _ in informative)
        bounds = [q for _, q in informative]
        result["roles"][role] = {
            "informative_steps": len(informative),
            "matches": k_inf,
            "null_expected_matches_at_most": math.fsum(bounds),
            "p_value": poisson_binomial_upper_tail(bounds, k_inf),
            "all_steps": {"matches": k, "null_expected_matches_at_most": math.fsum(null_p[role]),
                          "p_value": poisson_binomial_upper_tail(null_p[role], k)},
            "matched_steps": steps[role],
            # Per informative step: (null match bound, matched?) for pooling across answers.
            "informative": [[q, bool(m)] for m, q in informative],
        }
    return result


def pooled_p_value(rows: Sequence[dict], role: str) -> dict:
    """Pool the informative steps of several answers into one exact test for ``role``."""
    steps = [step for row in rows for step in row["roles"][role]["informative"]]
    k = sum(m for _, m in steps)
    bounds = [q for q, _ in steps]
    return {"answers": len(rows), "informative_steps": len(steps), "matches": k,
            "null_expected_matches_at_most": math.fsum(bounds),
            "p_value": poisson_binomial_upper_tail(bounds, k)}


def main(argv: list[str] | None = None) -> int:
    """Replay attribution on the answers at given lambdas of stored text atlases."""
    import argparse
    import json
    from pathlib import Path

    from token_auction_lab.atlas import _load

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.local_files_only = True
    model, _, _ = _load(args)
    rows = []
    for raw in args.paths:
        for file in sorted(Path(raw).rglob("*.json")):
            atlas = json.loads(file.read_text())
            if atlas.get("schema") != "bid-atlas-v1" or atlas.get("kind") != "text":
                continue
            manifest = atlas["manifest"]
            if manifest.get("model_revision") != args.revision:
                continue
            watermarked = "watermark" in manifest
            keys = (wm.keys_for_manifest(manifest, atlas["bidders"]) if watermarked
                    else wm.demo_keys(atlas["bidders"]))
            prompts = [p["token_ids"] for p in manifest["prompts"]]
            for lam in args.lambdas:
                leaf = next(x for x in atlas["leaves"] if x["lo"] < lam <= x["hi"])
                found = attribute(model, prompts, leaf["token_ids"], keys,
                                  temperature=manifest["temperature"],
                                  span_hold="span_hold" in manifest)
                truth = None
                if watermarked:
                    truth = {r: [t for t, s in enumerate(leaf["steps"]) if s.get("key") == r]
                             for r in wm.ROLES}
                row = {"file": f"{file.parent.name}/{file.name}", "seed": atlas["seed"],
                       "scenario": manifest.get("scenario"), "lambda": lam,
                       "watermarked": watermarked, "shared_tokens": keys.shared,
                       "span_hold": "span_hold" in manifest, "truth_steps": truth, **found}
                rows.append(row)
                summary = {r: (v["matches"], round(v["null_expected_matches_at_most"], 2),
                               f"{v['p_value']:.1e}") for r, v in found["roles"].items()}
                print(json.dumps({k: row[k] for k in ("file", "lambda", "watermarked")}
                                 | summary), flush=True)
                args.output.write_text(json.dumps(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
