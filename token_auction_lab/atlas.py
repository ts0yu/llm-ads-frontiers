"""Exact two-bidder bid-response atlas for text (MOONSHOTS.md sections 1, 2, 4, 5).

For one seed, the reference sampler reads bids only through ``u < lambda`` with
``lambda = bid_A / (bid_A + bid_B)``. The random tapes are the runner's per-step
draws from ``numpy.random.default_rng(seed)``, independent of bids. So the whole
generation is a piecewise-constant function of lambda. This module finds every
piece by interval search, forking the shared-prefix decoder where a contest
uniform falls strictly inside the current interval. A T-step generation has at
most 1 + T leaves.

The atlas is a deterministic function of the model numerics. It is verified by
direct ``run_round`` calls at chosen lambdas; it does not claim truthful bidding,
answer quality, or latency. All charges are simulated credits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import auction as auction_module
from auction import RandomTape, TwoBidderAuction
from token_auction_lab.decoder import Context, SharedPrefixDecoder
from token_auction_lab.prompts import ADVERTISER_PROMPT_POLICY, prompt_hash
from token_auction_lab.runner import RoundConfig, eos_ids, run_round
from token_auction_lab.span_hold import SpanState, hold_proposal, hold_uniforms
from token_auction_lab.span_hold import generate as span_hold_generate
from token_auction_lab.span_hold import manifest_entry as span_hold_manifest
from token_auction_lab.token_text import token_piece

SCHEMA = "bid-atlas-v1"
PARAMETER = "lambda = bid_A / (bid_A + bid_B)"
MODES = {"live_model", "synthetic_model_test", "synthetic_fixture"}
PAYMENT_TOLERANCE = 1e-9


def draw_tapes(seed: int, max_tokens: int) -> list[RandomTape]:
    """The runner's tape sequence for an advertiser round: one draw per step, in order."""
    rng = np.random.default_rng(seed)
    return [RandomTape.draw(rng) for _ in range(max_tokens)]


@torch.inference_mode()
def fork_decoder(decoder: SharedPrefixDecoder) -> SharedPrefixDecoder:
    """Copy every context's cache, token IDs, mask and logits. The model is shared."""
    clone = SharedPrefixDecoder.__new__(SharedPrefixDecoder)
    clone.model = decoder.model
    clone.device = decoder.device
    clone.generated = list(decoder.generated)
    # The memo maps the model to itself so that no cache field can drag a model copy.
    memo = {id(decoder.model): decoder.model}
    clone.contexts = [
        Context(
            context.prefix,
            list(context.token_ids),
            context.attention_mask.clone(),
            copy.deepcopy(context.cache, memo),
            context.logits.clone(),
        )
        for context in decoder.contexts
    ]
    for original, forked in zip(decoder.contexts, clone.contexts):
        if forked.cache is original.cache or forked.logits.data_ptr() == original.logits.data_ptr():
            raise RuntimeError("A forked context aliases its parent state.")
    if len(clone.contexts) == 2 and clone.contexts[0].cache is clone.contexts[1].cache:
        raise RuntimeError("Advertiser contexts unexpectedly share a cache.")
    return clone


def replay_decoder(model, prefixes: list[list[int]], generated: list[int]) -> SharedPrefixDecoder:
    """Rebuild a branch state with the same prefill-then-advance sequence as run_round."""
    decoder = SharedPrefixDecoder(model, prefixes)
    for token_id in generated:
        decoder.advance(token_id)
    return decoder


def decode_text(tokenizer, token_ids: list[int]) -> str:
    # Identical to run_round: decode the whole shared sequence, not token pieces.
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def outcome_values(text: str, bidders: list[str]) -> dict[str, bool]:
    """Exact, case-sensitive substring outcomes; no normalization or fuzzy matching."""
    return {f"mentions:{brand}": brand in text for brand in bidders}


@dataclass
class _Branch:
    leaf: int
    decoder: SharedPrefixDecoder | None
    generated: list[int]
    steps: list[dict]
    pending: int | None = None
    extra: dict = field(default_factory=dict)


@torch.inference_mode()
def build_atlas(
    model,
    tokenizer,
    prompts: list[tuple[str, list[int]]],
    bidders: list[str],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
    metadata: dict | None = None,
    mode: str = "live_model",
    fork: str = "copy",
    on_progress: Callable[[dict], None] | None = None,
    title: str | None = None,
    user_prompt: str | None = None,
    span_hold: bool = False,
) -> dict:
    """Interval search over lambda in (0, 1]. ``fork`` is "copy" or "replay".

    ``title`` and ``user_prompt`` (the shared user task, without advertiser
    briefs) are display fields for the Atlas Explorer; they do not affect decoding.
    ``span_hold`` selects the opt-in span-hold sampler (token_auction_lab.span_hold);
    held steps do not depend on bids, so the interval search is unchanged.
    """
    if len(prompts) != 2 or len(bidders) != 2:
        raise ValueError("The atlas is defined for exactly two bidders and two contexts.")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}.")
    if fork not in ("copy", "replay"):
        raise ValueError("fork must be 'copy' or 'replay'.")
    # Reuse the runner's own validation of temperature and token cap.
    RoundConfig(1.0, 1.0, seed=seed, temperature=temperature, max_tokens=max_tokens)
    started = time.perf_counter()
    prefixes = [list(ids) for _, ids in prompts]
    stops = eos_ids(model, tokenizer)
    tapes = draw_tapes(seed, max_tokens)
    holds = hold_uniforms(seed, max_tokens) if span_hold else None
    leaves: dict[int, dict] = {
        0: {"id": 0, "lo": 0.0, "hi": 1.0, "parent": None, "split_step": None}
    }
    splits: list[dict] = []
    advances = 0
    stack = [_Branch(0, SharedPrefixDecoder(model, prefixes), [], [],
                     extra={"span_state": SpanState()} if span_hold else {})]
    while stack:
        branch = stack.pop()
        leaf = leaves[branch.leaf]
        decoder = branch.decoder
        if decoder is None:
            decoder = replay_decoder(model, prefixes, branch.generated)
            advances += len(branch.generated)
        stop_reason = "length"
        if branch.pending is not None:
            decoder.advance(branch.pending)
            advances += 1
            if branch.pending in stops:
                stop_reason = "eos"
        branch.decoder = None  # Do not keep a second reference while running.
        steps = branch.steps
        state = branch.extra.get("span_state")
        while stop_reason != "eos" and len(decoder.generated) < max_tokens:
            step = len(decoder.generated)
            p_a, p_b = decoder.distributions(temperature)
            step_auction = TwoBidderAuction(p_a, p_b)
            tape = tapes[step]
            proposal = None
            if span_hold:
                proposal = hold_proposal(state, p_a, p_b, holds[step], tokenizer, decoder.generated)
                if proposal and proposal["keep"]:
                    held = {
                        "step": step,
                        "branch": f"hold_{state.owner}",
                        "u": tape.contest,
                        "hold_u": holds[step],
                        "shared_mass": step_auction.shared_mass,
                        "disagreement": step_auction.disagreement,
                        "token_id": proposal["token_id"],
                        "candidate_a": None,
                        "candidate_b": None,
                    }
                    state = state.after(proposal["piece"], held["branch"])
                    held["span"] = state.as_record()
                    steps.append(held)
                    decoder.advance(held["token_id"])
                    advances += 1
                    if held["token_id"] in stops:
                        stop_reason = "eos"
                    continue
                if proposal:
                    state = state.released()
            # Candidates and the shared/contested branch do not depend on bids.
            allocation = step_auction.resolve(1.0, 1.0, tape)
            u = tape.contest
            record = {
                "step": step,
                "branch": "shared",
                "u": u,
                "shared_mass": step_auction.shared_mass,
                "disagreement": step_auction.disagreement,
                "token_id": allocation.token_id,
                "candidate_a": allocation.candidate_a,
                "candidate_b": allocation.candidate_b,
            }
            if proposal:
                record["released"] = {"token_id": proposal["token_id"], "piece": proposal["piece"]}
            if allocation.branch != "shared":
                lo, hi = leaf["lo"], leaf["hi"]
                if u <= lo:  # every lambda in (lo, hi] exceeds u: A wins
                    record.update(branch="A", token_id=allocation.candidate_a)
                elif u >= hi:  # every lambda in (lo, hi] is at most u: B wins
                    record.update(branch="B", token_id=allocation.candidate_b)
                else:
                    child_id = len(leaves)
                    leaves[child_id] = {
                        "id": child_id,
                        "lo": u,
                        "hi": hi,
                        "parent": leaf["id"],
                        "split_step": step,
                    }
                    leaf["hi"] = u
                    splits.append(
                        {
                            "step": step,
                            "u": u,
                            "lo": lo,
                            "hi": hi,
                            "parent": leaf["id"],
                            "child": child_id,
                            "b_token": allocation.candidate_b,
                            "a_token": allocation.candidate_a,
                            "b_token_text": token_piece(
                                tokenizer, decoder.generated, allocation.candidate_b
                            ),
                            "a_token_text": token_piece(
                                tokenizer, decoder.generated, allocation.candidate_a
                            ),
                            "prefix_token_count": step,
                            "prefix_text": decode_text(tokenizer, decoder.generated),
                        }
                    )
                    child_record = {**record, "branch": "A", "token_id": allocation.candidate_a}
                    child_extra = {}
                    if span_hold:
                        child_state = state.after(splits[-1]["a_token_text"], "A")
                        child_record["span"] = child_state.as_record()
                        child_extra = {"span_state": child_state}
                    child_steps = [*steps, child_record]
                    stack.append(
                        _Branch(
                            child_id,
                            fork_decoder(decoder) if fork == "copy" else None,
                            list(decoder.generated),
                            child_steps,
                            pending=allocation.candidate_a,
                            extra=child_extra,
                        )
                    )
                    record.update(branch="B", token_id=allocation.candidate_b)
                    if on_progress:
                        on_progress(
                            {
                                "type": "split",
                                "step": step,
                                "u": u,
                                "parent": leaf["id"],
                                "child": child_id,
                                "pending": len(stack),
                            }
                        )
            if span_hold:
                piece = token_piece(tokenizer, decoder.generated, record["token_id"])
                state = state.after(piece, record["branch"])
                record["span"] = state.as_record()
            steps.append(record)
            decoder.advance(record["token_id"])
            advances += 1
            if record["token_id"] in stops:
                stop_reason = "eos"
        token_ids = list(decoder.generated)
        if [s["token_id"] for s in steps] != token_ids:
            raise RuntimeError("Leaf step records disagree with the decoded token IDs.")
        decoder.assert_shared_suffix()
        text = decode_text(tokenizer, token_ids)
        leaf.update(
            token_ids=token_ids,
            tokens=len(token_ids),
            text=text,
            stop_reason=stop_reason,
            steps=steps,
            outcomes=outcome_values(text, bidders),
        )
        del decoder
        if on_progress:
            on_progress(
                {
                    "type": "leaf",
                    "leaf": leaf["id"],
                    "tokens": len(token_ids),
                    "stop_reason": stop_reason,
                    "pending": len(stack),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    ordered = sorted(leaves.values(), key=lambda value: value["lo"])
    sampler = {"sampler": "auction.TwoBidderAuction"}
    if span_hold:
        sampler = {
            "sampler": "auction.TwoBidderAuction + token_auction_lab.span_hold",
            "span_hold": span_hold_manifest(),
        }
    atlas = {
        "schema": SCHEMA,
        "kind": "text",
        "mode": mode,
        "title": title,
        "label": "SPONSORED SIMULATION",
        "credit_unit": "simulated",
        "bidders": list(bidders),
        "seed": seed,
        "parameter": PARAMETER,
        "max_steps": max_tokens,
        "manifest": {
            **(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_prompt": user_prompt,
            "prompts": [
                {"text": text, "sha256": prompt_hash(text), "token_ids": list(ids)}
                for text, ids in prompts
            ],
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "rng": "numpy.PCG64",
            "tape_rule": "Step k uses the k-th RandomTape.draw(numpy.random.default_rng(seed)), "
            "exactly as run_round draws advertiser-round tapes, independent of bids.",
            **sampler,
            "eos_token_ids": sorted(stops),
            "sampling_filters": None,
            "fork": fork,
            "interval_rule": "A wins a contested step iff u < lambda. At interval (lo, hi]: "
            "u <= lo gives A, u >= hi gives B, otherwise (lo, u] takes B and (u, hi] takes A.",
        },
        "build": {"seconds": time.perf_counter() - started, "decoder_advances": advances},
        "leaves": ordered,
        "splits": splits,
        "verification": [],
    }
    check_partition(atlas)
    return atlas


def check_partition(atlas: dict) -> None:
    leaves = atlas["leaves"]
    if not leaves or leaves[0]["lo"] != 0.0 or leaves[-1]["hi"] != 1.0:
        raise AssertionError("Leaves must cover (0, 1].")
    for left, right in zip(leaves, leaves[1:]):
        if left["hi"] != right["lo"]:
            raise AssertionError("Leaves must be contiguous.")
    if any(not leaf["lo"] < leaf["hi"] for leaf in leaves):
        raise AssertionError("Every leaf must be a nonempty interval.")
    if len(leaves) > 1 + atlas["max_steps"]:
        raise AssertionError("More leaves than the 1 + T bound.")
    if "span_hold" not in atlas.get("manifest", {}) and any(
        step["branch"].startswith("hold_") for leaf in leaves for step in leaf.get("steps", [])
    ):
        raise AssertionError("Held steps in an atlas without the span-hold sampler.")
    split_steps = [split["step"] for split in atlas["splits"]]
    if len(set(split_steps)) != len(split_steps):
        raise AssertionError("Two branches split at the same step.")


def leaf_for(atlas: dict, lam: float) -> dict:
    """The leaf containing lambda; lambda = 0 is the limit 0+ (bids (0, b))."""
    for leaf in atlas["leaves"]:
        if leaf["lo"] < lam <= leaf["hi"] or (lam == 0.0 and leaf["lo"] == 0.0):
            return leaf
    raise ValueError(f"lambda={lam} is outside (0, 1].")


def _bid_lambda(bid_a: float, bid_b: float) -> tuple[float, float, float]:
    a, b = float(bid_a), float(bid_b)
    if not all(math.isfinite(x) and x >= 0.0 for x in (a, b)) or a + b == 0.0:
        raise ValueError("Bids must be finite, nonnegative, and not both zero.")
    total = a + b
    return a, b, a / total  # Same float expression as TwoBidderAuction.resolve.


def payments(leaf: dict, bid_a: float, bid_b: float) -> dict:
    """Exact cumulative simulated charges for a bid pair whose lambda lies in this leaf.

    A contested step won by A charges A ``b*u/(1-u)``; one won by B charges B
    ``a*(1-u)/u`` (``S*(1-lambda) = b`` and ``S*lambda = a``). Expressions and
    summation order match run_round, so the totals are reproduced bit-for-bit.
    """
    a, b, lam = _bid_lambda(bid_a, bid_b)
    if not (leaf["lo"] < lam <= leaf["hi"] or (lam == 0.0 and leaf["lo"] == 0.0)):
        raise ValueError(f"lambda={lam} is not in leaf ({leaf['lo']}, {leaf['hi']}].")
    payment_a = payment_b = 0.0
    contested = 0
    for step in leaf["steps"]:
        if step["branch"] == "shared" or step["branch"].startswith("hold_"):
            continue  # Shared and held (span-hold) steps have no contest and no charge.
        contested += 1
        u = step["u"]
        a_wins = u < lam
        if a_wins != (step["branch"] == "A"):
            raise AssertionError(f"Step {step['step']} winner disagrees with lambda={lam}.")
        if a_wins:
            payment_a += b * (u / (1.0 - u))
        else:
            payment_b += a * ((1.0 - u) / u)
    return {
        "lambda": lam,
        "payment_a": payment_a,
        "payment_b": payment_b,
        "contested_steps": contested,
    }


def analyze(atlas: dict) -> dict:
    """Exact outcome sets over lambda, monotonicity audit, and the 1 + T leaf bound.

    Lambda is treated as uniform on (0, 1], so an outcome's measure is the total
    length of its true set. ``constant`` outcomes are monotone in both directions.
    Critical lambdas are leaf boundaries; they carry no truthfulness claim.
    """
    leaves = sorted(atlas["leaves"], key=lambda leaf: leaf["lo"])
    bidders = atlas["bidders"]
    expected = {f"mentions:{bidders[0]}": "increasing", f"mentions:{bidders[1]}": "decreasing"}
    keys = sorted({key for leaf in leaves for key in leaf["outcomes"]})
    outcomes = {}
    for key in keys:
        values = [bool(leaf["outcomes"].get(key, False)) for leaf in leaves]
        intervals: list[list[float]] = []
        for leaf, value in zip(leaves, values):
            if not value:
                continue
            if intervals and intervals[-1][1] == leaf["lo"]:
                intervals[-1][1] = leaf["hi"]
            else:
                intervals.append([leaf["lo"], leaf["hi"]])
        switches = sum(left != right for left, right in zip(values, values[1:]))
        critical = None
        if switches == 0:
            classification = "constant"
        elif switches == 1:
            boundary = next(
                left["hi"]
                for left, l_value, r_value in zip(leaves, values, values[1:])
                if l_value != r_value
            )
            critical = boundary
            classification = "monotone_increasing" if values[-1] else "monotone_decreasing"
        else:
            classification = "non_monotone"
        increasing = classification in ("constant", "monotone_increasing")
        decreasing = classification in ("constant", "monotone_decreasing")
        direction = expected.get(key)
        outcomes[key] = {
            "true_intervals": intervals,
            "interval_semantics": "(lo, hi]",
            "measure": math.fsum(hi - lo for lo, hi in intervals),
            "classification": classification,
            "switches": switches,
            "monotone_increasing": increasing,
            "monotone_decreasing": decreasing,
            "critical_lambda": critical,
            "expected_direction": direction,
            "consistent_with_expected": None
            if direction is None
            else (increasing if direction == "increasing" else decreasing),
        }
    lengths = [len(leaf["token_ids"]) for leaf in leaves]
    return {
        "leaves": len(leaves),
        "splits": len(atlas["splits"]),
        "bound": 1 + atlas["max_steps"],
        "realized_bound": 1 + max(lengths, default=0),
        "within_bound": len(leaves) <= 1 + atlas["max_steps"],
        "split_steps_distinct": len({s["step"] for s in atlas["splits"]}) == len(atlas["splits"]),
        "distinct_texts": len({leaf["text"] for leaf in leaves}),
        "outcomes": outcomes,
    }


def verify_point(
    atlas: dict,
    leaf: dict,
    bid_a: float,
    bid_b: float,
    model,
    tokenizer,
    *,
    workdir: Path,
    kind: str,
) -> dict:
    """Run run_round at one bid pair and compare it with the atlas leaf.

    Span-hold atlases are compared with ``span_hold.generate`` instead, the direct
    (non-tree) span-hold sampler, since run_round is the reference sampler.
    """
    expected = payments(leaf, bid_a, bid_b)
    manifest = atlas["manifest"]
    prompts = [(p["text"], list(p["token_ids"])) for p in manifest["prompts"]]
    if "span_hold" in manifest:
        return _verify_point_span(atlas, leaf, bid_a, bid_b, model, tokenizer, prompts, kind)
    config = RoundConfig(
        bid_a,
        bid_b,
        seed=atlas["seed"],
        temperature=manifest["temperature"],
        max_tokens=atlas["max_steps"],
        checkpoint_steps=(),
    )
    index = len(list(workdir.iterdir()))
    output = workdir / f"verify-{index:03}-leaf{leaf['id']}-{kind}"
    result = run_round(
        model, tokenizer, prompts, config, output, {"atlas_verification": True}, mode=atlas["mode"]
    )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    events = events[:-1]
    steps_match = len(events) == len(leaf["steps"]) and all(
        event["branch"] == step["branch"]
        and event["token_id"] == step["token_id"]
        and event["tape"]["contest"] == step["u"]
        and event["shared_mass"] == step["shared_mass"]
        for event, step in zip(events, leaf["steps"])
    )
    error_a = abs(result["payment_a"] - expected["payment_a"])
    error_b = abs(result["payment_b"] - expected["payment_b"])
    tokens_match = result["token_ids"] == leaf["token_ids"]
    payments_match = max(error_a, error_b) <= PAYMENT_TOLERANCE
    return {
        "leaf": leaf["id"],
        "kind": kind,
        "lambda": expected["lambda"],
        "bid_a": bid_a,
        "bid_b": bid_b,
        "token_ids_match": tokens_match,
        "text_match": result["text"] == leaf["text"],
        "stop_reason_match": result["stop_reason"] == leaf["stop_reason"],
        "steps_match": steps_match,
        "payment_a_direct": result["payment_a"],
        "payment_b_direct": result["payment_b"],
        "payment_a_atlas": expected["payment_a"],
        "payment_b_atlas": expected["payment_b"],
        "max_payment_abs_error": max(error_a, error_b),
        "payments_match": payments_match,
        "matches_direct_run": tokens_match
        and payments_match
        and steps_match
        and result["text"] == leaf["text"]
        and result["stop_reason"] == leaf["stop_reason"],
        "direct_run_seconds": result["seconds"],
    }


def _verify_point_span(atlas, leaf, bid_a, bid_b, model, tokenizer, prompts, kind) -> dict:
    expected = payments(leaf, bid_a, bid_b)
    started = time.perf_counter()
    result = span_hold_generate(
        model, tokenizer, prompts, bid_a, bid_b, seed=atlas["seed"],
        temperature=atlas["manifest"]["temperature"], max_tokens=atlas["max_steps"],
    )
    steps_match = len(result["steps"]) == len(leaf["steps"]) and all(
        direct["branch"] == step["branch"]
        and direct["token_id"] == step["token_id"]
        and direct["u"] == step["u"]
        and direct["shared_mass"] == step["shared_mass"]
        and direct.get("span") == step.get("span")
        for direct, step in zip(result["steps"], leaf["steps"])
    )
    error_a = abs(result["payment_a"] - expected["payment_a"])
    error_b = abs(result["payment_b"] - expected["payment_b"])
    tokens_match = result["token_ids"] == leaf["token_ids"]
    payments_match = max(error_a, error_b) <= PAYMENT_TOLERANCE
    return {
        "leaf": leaf["id"],
        "kind": kind,
        "lambda": expected["lambda"],
        "bid_a": bid_a,
        "bid_b": bid_b,
        "sampler": "span_hold.generate",
        "token_ids_match": tokens_match,
        "text_match": result["text"] == leaf["text"],
        "stop_reason_match": result["stop_reason"] == leaf["stop_reason"],
        "steps_match": steps_match,
        "payment_a_direct": result["payment_a"],
        "payment_b_direct": result["payment_b"],
        "payment_a_atlas": expected["payment_a"],
        "payment_b_atlas": expected["payment_b"],
        "max_payment_abs_error": max(error_a, error_b),
        "payments_match": payments_match,
        "matches_direct_run": tokens_match
        and payments_match
        and steps_match
        and result["text"] == leaf["text"]
        and result["stop_reason"] == leaf["stop_reason"],
        "direct_run_seconds": time.perf_counter() - started,
    }


def midpoint_bids(leaf: dict) -> tuple[float, float]:
    lam = (leaf["lo"] + leaf["hi"]) / 2
    return lam, 1.0 - lam


def verify(
    atlas: dict,
    model,
    tokenizer,
    n: int = 3,
    *,
    endpoints: bool = True,
    workdir: Path | None = None,
    sample_seed: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Direct run_round checks at leaf midpoints: both extreme leaves plus n others.

    The n interior leaves are drawn without replacement from a generator seeded by
    ``sample_seed`` (default: the atlas seed). With ``endpoints``, bids (1, 0) and
    (0, 1) are also checked. Results are appended to ``atlas["verification"]``.
    """
    leaves = atlas["leaves"]
    chosen = [leaves[0]] + ([leaves[-1]] if len(leaves) > 1 else [])
    middle = leaves[1:-1]
    if middle and n > 0:
        rng = np.random.default_rng(atlas["seed"] if sample_seed is None else sample_seed)
        picks = rng.choice(len(middle), size=min(n, len(middle)), replace=False)
        chosen[1:1] = [middle[i] for i in sorted(int(i) for i in picks)]
    plan = [(leaf, *midpoint_bids(leaf), "midpoint") for leaf in chosen]
    if endpoints:
        plan += [
            (leaf_for(atlas, 1.0), 1.0, 0.0, "endpoint"),
            (leaf_for(atlas, 0.0), 0.0, 1.0, "endpoint"),
        ]
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) if workdir is None else workdir
        root.mkdir(parents=True, exist_ok=True)
        for leaf, bid_a, bid_b, kind in plan:
            record = verify_point(
                atlas, leaf, bid_a, bid_b, model, tokenizer, workdir=root, kind=kind
            )
            records.append(record)
            atlas["verification"].append(record)
            if on_progress:
                on_progress(
                    {
                        "type": "verified",
                        **{
                            k: record[k]
                            for k in (
                                "leaf",
                                "kind",
                                "lambda",
                                "matches_direct_run",
                                "max_payment_abs_error",
                            )
                        },
                    }
                )
    return records


# ---------------------------------------------------------------------------- CLI


def _live_default_prompts(tokenizer) -> tuple[list[tuple[str, list[int]]], list[str], dict]:
    from token_auction_lab.live import LiveRequest, default_inputs, live_messages

    request = LiveRequest.parse(default_inputs())
    prompts = []
    for advertiser in (0, 1):
        text = tokenizer.apply_chat_template(
            live_messages(request, advertiser), tokenize=False, add_generation_prompt=True
        )
        prompts.append((text, tokenizer.encode(text, add_special_tokens=False)))
    # Bids in the defaults are ignored: the atlas covers every bid ratio.
    inputs = {k: v for k, v in asdict(request).items() if k != "advertisers"}
    inputs["advertisers"] = [
        {k: v for k, v in asdict(ad).items() if k != "bid"} for ad in request.advertisers
    ]
    return prompts, [ad.brand for ad in request.advertisers], inputs


def _scenario_prompts(tokenizer, name: str):
    from token_auction_lab.prompts import render, scenarios

    scenario = scenarios()[name]
    prompts = [render(tokenizer, scenario, i) for i in (0, 1)]
    return prompts, [ad["brand"] for ad in scenario["advertisers"]], None


def _display(name: str, inputs: dict | None) -> tuple[str, str]:
    """Explorer title and the shared user task (no advertiser briefs, no bids)."""
    from token_auction_lab.prompts import scenarios

    if inputs is not None:
        return "Live Lab defaults", inputs["prompt"]
    scenario = scenarios()[name]
    return scenario.get("title", name), scenario["task"]


def _load(args):
    """Mirrors cli.py: device/dtype choice, a resolved revision SHA, sdpa attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = (
            "bfloat16"
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else ("float16" if device == "cuda" else "float32")
        )
    if re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        revision = args.revision
    else:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(args.model, revision=args.revision).sha
    print(
        json.dumps(
            {
                "status": "loading",
                "model": args.model,
                "revision": revision,
                "device": device,
                "dtype": dtype_name,
            }
        ),
        flush=True,
    )
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, local_files_only=args.local_files_only
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=revision,
            dtype=getattr(torch, dtype_name),
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        )
        .to(device)
        .eval()
    )
    root = Path(__file__).resolve().parent
    metadata = {
        "model_id": args.model,
        "model_revision": revision,
        "tokenizer": args.model,
        "tokenizer_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "model_vocab_size": model.config.vocab_size,
        "tokenizer_size": len(tokenizer),
        "device": device,
        "dtype": dtype_name,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "platform": platform.platform(),
        "attention_implementation": "sdpa",
        "advertiser_prompt_policy": ADVERTISER_PROMPT_POLICY,
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "python": platform.python_version(),
        "model_load_seconds": time.perf_counter() - start,
        "local_files_only": args.local_files_only,
        "torch_num_threads": torch.get_num_threads(),
        "reference_sha256": hashlib.sha256(Path(auction_module.__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted(root.glob("*.py")),
                root / "fact_sheets.json",
                root / "live_defaults.json",
                Path(auction_module.__file__).resolve(),
            ]
        },
    }
    return model, tokenizer, metadata


def _summary(atlas: dict, path: Path) -> dict:
    analysis = analyze(atlas)
    return {
        "file": path.name,
        "scenario": atlas["manifest"].get("scenario"),
        "seed": atlas["seed"],
        "mode": atlas["mode"],
        "leaves": analysis["leaves"],
        "bound": analysis["bound"],
        "distinct_texts": analysis["distinct_texts"],
        "build_seconds": atlas["build"]["seconds"],
        "outcomes": {
            k: {
                "classification": v["classification"],
                "measure": v["measure"],
                "critical_lambda": v["critical_lambda"],
            }
            for k, v in analysis["outcomes"].items()
        },
        "verification": {
            "runs": len(atlas["verification"]),
            "all_match": all(r["matches_direct_run"] for r in atlas["verification"])
            if atlas["verification"]
            else None,
            "status": atlas.get("verification_status"),
        },
    }


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None, *, mode: str = "live_model") -> None:
    """``mode`` stays live_model for real inference; tests pass synthetic_model_test."""
    from token_auction_lab.prompts import scenarios

    parser = argparse.ArgumentParser(description="Build exact two-bidder bid atlases (text).")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", help="A commit SHA; resolved to a SHA if omitted.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use cached files; requires a full revision SHA.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--scenario", choices=tuple(scenarios()), default="transit-hotel")
    parser.add_argument(
        "--live-defaults",
        action="store_true",
        help="Use the Live Lab defaults (Fort Mason prompt) instead of a scenario.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--fork",
        choices=("copy", "replay"),
        default="copy",
        help="copy: deepcopy KV caches at splits; replay: re-prefill (less memory).",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=3,
        help="Interior leaves to check by direct run_round, besides both extremes.",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Do not start a new seed (or its verification) after this budget.",
    )
    parser.add_argument(
        "--span-hold",
        action="store_true",
        help="Opt-in span-hold sampler (docs/SPAN_HOLD.md) instead of the reference sampler.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/atlas"))
    args = parser.parse_args(argv)
    if args.local_files_only and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        parser.error("--local-files-only requires --revision with a full 40-character commit SHA.")
    RoundConfig(1.0, 1.0, temperature=args.temperature, max_tokens=args.max_tokens)
    started = time.perf_counter()

    def over_budget() -> bool:
        return (
            args.time_budget_seconds is not None
            and time.perf_counter() - started >= args.time_budget_seconds
        )

    model, tokenizer, metadata = _load(args)
    if args.live_defaults:
        prompts, bidders, inputs = _live_default_prompts(tokenizer)
        label = "live-defaults"
        metadata.update(scenario=label, live_inputs_without_bids=inputs)
    else:
        prompts, bidders, inputs = _scenario_prompts(tokenizer, args.scenario)
        label = args.scenario
        metadata.update(scenario=label)
    title, user_prompt = _display(label, inputs)
    if args.span_hold:
        title = f"{title} · span hold"
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    index = {
        "schema": "bid-atlas-index-v1",
        "mode": mode,
        "scenario": label,
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "argv": vars(args) | {"output": str(args.output)},
        "atlases": [],
        "skipped_seeds": [],
    }

    def progress(event: dict) -> None:
        print(json.dumps({"seed": current_seed, **event}), flush=True)

    for current_seed in args.seeds:
        if over_budget():
            index["skipped_seeds"].append(current_seed)
            progress({"type": "skipped", "reason": "time budget exhausted"})
            _write_json(index_path, index)
            continue
        torch.manual_seed(current_seed)
        progress({"type": "atlas_start", "scenario": label})
        atlas = build_atlas(
            model,
            tokenizer,
            prompts,
            bidders,
            seed=current_seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            metadata={**metadata, "torch_seed": current_seed},
            mode=mode,
            fork=args.fork,
            on_progress=progress,
            title=title,
            user_prompt=user_prompt,
            span_hold=args.span_hold,
        )
        atlas["analysis"] = analyze(atlas)
        atlas["verification_status"] = "pending"
        path = args.output / f"{label}-seed{current_seed}.json"
        _write_json(path, atlas)
        index["atlases"].append(_summary(atlas, path))
        _write_json(index_path, index)
        progress(
            {
                "type": "atlas_written",
                "file": str(path),
                "leaves": len(atlas["leaves"]),
                "seconds": atlas["build"]["seconds"],
            }
        )
        if args.no_verify:
            atlas["verification_status"] = "not_requested"
        elif over_budget():
            atlas["verification_status"] = "skipped_time_budget"
        else:
            verify(atlas, model, tokenizer, args.verify, on_progress=progress)
            atlas["verification_status"] = (
                "all_match"
                if all(r["matches_direct_run"] for r in atlas["verification"])
                else "MISMATCH"
            )
        _write_json(path, atlas)
        index["atlases"][-1] = _summary(atlas, path)
        _write_json(index_path, index)
        progress({"type": "atlas_done", "verification_status": atlas["verification_status"]})


if __name__ == "__main__":
    main()
