"""Exact bid-simplex atlas for n-bidder text generation (research extension).

For a fixed seed, step k's `RaceTape` is the k-th draw from
`numpy.random.default_rng(seed)`, regardless of bids. Bids enter only through
the race winner argmin_i E_i/b_i, which depends on b only through its
direction lambda = b/sum(b). So the whole generation is a piecewise-constant
function of lambda on the simplex (given model numerics).

At a contested step with exponentials E, bidder i wins on the convex cone
{lambda : lambda_i/E_i >= lambda_j/E_j for all j}, apex lambda* = E/sum(E).
Starting from the whole simplex, each live cell is clipped by those cones;
pieces with positive area fork the decoder (pieces with the same candidate
token share one decoder state) and continue with the winner's candidate.
Shared steps never split. Cells are convex polygons (n = 3) or intervals
(n = 2), stored as barycentric vertex lists; 2-D measure uses
(lambda_1, lambda_2), so the cells' areas sum to 1/2 (n = 3) or 1 (n = 2).

Payments are not constant within a cell: at bids b in a cell, a contested
step won by w charges E_w * max_{j != w} b_j/E_j (piecewise linear in b).
`cell_payments` evaluates the exact sum from the stored per-step E.

Bound: for n = 2 at most 1 + T outcomes (MOONSHOTS.md). For n = 3 each
contested step adds a fan of 3 rays from lambda*; cells are faces of (a
coarsening of) the arrangement of <= 3T rays, so the count is O(T^2), not
O(T). In general O(T^(n-1)).

Labels: tiny random models are `synthetic_model_test`; the CLI is
`live_model`; viewing stored JSON is REPLAY. Simulated credits only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from auction import RandomTape
from token_auction_lab.auction_n import NBidderAuction, RaceTape, critical_bid
from token_auction_lab.token_text import token_piece

SCHEMA = "simplex-atlas-v1"
MIN_AREA = 1e-12

TapeSource = Callable[[np.random.Generator, int], RaceTape]


def race_tapes(rng: np.random.Generator, n: int) -> RaceTape:
    return RaceTape.draw(rng, n)


def two_bidder_tapes(rng: np.random.Generator, n: int) -> RaceTape:
    """n = 2 only: draw `auction.RandomTape` and map it exactly to the race."""
    if n != 2:
        raise ValueError("The two-bidder tape mapping needs exactly two bidders.")
    return RaceTape.from_two_bidder(RandomTape.draw(rng))


TAPE_SOURCES: dict[str, TapeSource] = {
    "race": race_tapes,
    "two-bidder-random-tape": two_bidder_tapes,
}


# --------------------------------------------------------------------------- geometry


def simplex(n: int) -> np.ndarray:
    if n not in (2, 3):
        raise NotImplementedError("Atlases are implemented for n = 2 (intervals) and n = 3.")
    return np.eye(n, dtype=np.float64)


def simplex_measure(n: int) -> float:
    return {2: 1.0, 3: 0.5}[n]


def measure(polygon: np.ndarray) -> float:
    """Interval length in lambda_1 (n=2) or area in (lambda_1, lambda_2) (n=3)."""
    if len(polygon) < 2:
        return 0.0
    if polygon.shape[1] == 2:
        return float(polygon[:, 0].max() - polygon[:, 0].min())
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def centroid(polygon: np.ndarray) -> np.ndarray:
    if polygon.shape[1] == 2 or len(polygon) < 3:
        point = polygon.mean(axis=0)
    else:
        x, y = polygon[:, 0], polygon[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2
        if abs(area) < 1e-300:
            point = polygon.mean(axis=0)
        else:
            cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
            cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
            point = np.array([cx, cy, 1.0 - cx - cy])
    point = np.clip(point, 0.0, None)
    return point / point.sum()


def random_interior_point(polygon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform point in a convex cell (fan triangulation for n = 3)."""
    if polygon.shape[1] == 2:
        t = rng.random()
        point = polygon[0] + t * (polygon[-1] - polygon[0])
    else:
        triangles = [(polygon[0], polygon[k], polygon[k + 1]) for k in range(1, len(polygon) - 1)]
        areas = np.array([measure(np.array(t)) for t in triangles])
        a, b, c = triangles[int(rng.choice(len(triangles), p=areas / areas.sum()))]
        r1, r2 = rng.random(2)
        s = math.sqrt(r1)
        point = (1 - s) * a + s * (1 - r2) * b + s * r2 * c
    point = np.clip(point, 0.0, None)
    return point / point.sum()


def _dedupe(points: list[np.ndarray], closed: bool) -> np.ndarray:
    result: list[np.ndarray] = []
    for p in points:
        if not result or np.abs(p - result[-1]).max() > 1e-15:
            result.append(p)
    if closed and len(result) > 1 and np.abs(result[0] - result[-1]).max() <= 1e-15:
        result.pop()
    return np.array(result) if result else np.zeros((0, points[0].shape[0] if points else 0))


def clip(polygon: np.ndarray, normal: np.ndarray, tol: float) -> np.ndarray:
    """Keep {lambda : normal . lambda >= 0} of a convex cell (Sutherland-Hodgman).

    Values within `tol` of zero are snapped onto the line and count as inside,
    so a cell that merely touches the line yields a degenerate (empty) piece.
    """
    s = polygon @ normal
    s[np.abs(s) <= tol] = 0.0
    if (s >= 0).all():
        return polygon
    if (s <= 0).all():
        return polygon[s == 0]

    def cut(p, q, sp, sq):
        return p + (sp / (sp - sq)) * (q - p)

    if polygon.shape[1] == 2:  # interval: two endpoints
        (p, q), (sp, sq) = polygon, s
        return np.array([p, cut(p, q, sp, sq)]) if sp > 0 else np.array([cut(p, q, sp, sq), q])
    out: list[np.ndarray] = []
    count = len(polygon)
    for k in range(count):
        cur, prev = polygon[k], polygon[k - 1]
        sc, sp = s[k], s[k - 1]
        if sc >= 0:
            if sp < 0:
                out.append(cut(prev, cur, sp, sc))
            out.append(cur)
        elif sp > 0:
            out.append(cut(prev, cur, sp, sc))
    return _dedupe(out, closed=True)


def winner_region_normals(exponentials: Sequence[float], i: int) -> list[np.ndarray]:
    """lambda_i/E_i >= lambda_j/E_j  <=>  E_j*lambda_i - E_i*lambda_j >= 0."""
    n = len(exponentials)
    normals = []
    for j in range(n):
        if j != i:
            a = np.zeros(n)
            a[i], a[j] = exponentials[j], -exponentials[i]
            normals.append(a)
    return normals


def partition(
    polygon: np.ndarray, exponentials: Sequence[float], min_area: float = MIN_AREA
) -> tuple[list[tuple[int, np.ndarray]], list[float]]:
    """Split a cell by race winner. Returns (pieces, dropped sliver areas)."""
    n = len(exponentials)
    regions = [winner_region_normals(exponentials, i) for i in range(n)]
    tols = [[1e-13 * float(np.abs(a).sum()) for a in normals] for normals in regions]
    for i, normals in enumerate(regions):
        if all(((polygon @ a) >= -t).all() for a, t in zip(normals, tols[i])):
            return [(i, polygon)], []  # Whole cell in one region: no arithmetic on it.
    pieces, slivers = [], []
    for i, normals in enumerate(regions):
        piece = polygon
        for a, t in zip(normals, tols[i]):
            if len(piece) < (2 if n == 2 else 3):
                break
            piece = clip(piece, a, t)
        if len(piece) < (2 if n == 2 else 3):
            continue
        area = measure(piece)
        if area < min_area:
            if area > 0:
                slivers.append(area)
            continue
        pieces.append((i, piece))
    return pieces, slivers


# --------------------------------------------------------------------------- generation


def decode_text(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def cell_payments(steps: Sequence[dict], bids: Sequence[float]) -> list[float]:
    """Exact cumulative charges at `bids` for a cell's recorded steps."""
    bids = [float(b) for b in bids]
    payments = [0.0] * len(bids)
    for record in steps:
        if record["branch"] == "contested":
            w = record["winner"]
            charge = [0.0] * len(bids)
            charge[w] = critical_bid(record["E"], bids, w)
            for i, c in enumerate(charge):
                payments[i] += c
    return payments


def direct_generate(
    model,
    prefixes: list[list[int]],
    bids: Sequence[float],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
    eos_ids: set[int],
    tape_source: TapeSource = race_tapes,
) -> dict:
    """Ordinary (non-tree) n-bidder generation at one fixed bid vector."""
    from token_auction_lab.multi_decoder import MultiPrefixDecoder

    n = len(prefixes)
    rng = np.random.default_rng(seed)
    decoder = MultiPrefixDecoder(model, prefixes)
    payments = [0.0] * n
    steps, token_ids, stop_reason = [], [], "length"
    for step in range(max_tokens):
        tape = tape_source(rng, n)
        auction = NBidderAuction(decoder.distributions(temperature))
        allocation = auction.resolve(bids, tape)
        for i, c in enumerate(allocation.payments):
            payments[i] += c
        steps.append({"step": step, "branch": allocation.branch, "winner": allocation.winner,
                      "token_id": allocation.token_id})
        token_ids.append(allocation.token_id)
        if allocation.token_id in eos_ids:
            stop_reason = "eos"
            break
        decoder.advance(allocation.token_id)
        if decoder.generated != token_ids:
            raise RuntimeError("Direct run lost its shared suffix.")
    return {"token_ids": token_ids, "payments": payments, "stop_reason": stop_reason,
            "steps": steps}


@dataclass
class Cell:
    id: int
    polygon: np.ndarray
    parent: int | None
    split_step: int | None
    steps: list[dict] = field(default_factory=list)
    token_ids: list[int] | None = None
    stop_reason: str | None = None
    truncation: dict | None = None
    cache_check: dict | None = None


@dataclass
class _Branch:
    decoder: object
    cells: list[Cell]


def cache_tolerance(model) -> dict:
    import torch

    parameter = next(model.parameters())
    if parameter.dtype == torch.float32:
        atol = 3e-4 if parameter.device.type == "mps" else 1e-4
        return {"atol": atol, "rtol": 1e-4, "max_probability_tv": 1e-4}
    return {"atol": 0.08, "rtol": 0.02, "max_probability_tv": None}


def build_atlas(
    model,
    tokenizer,
    prefixes: list[list[int]],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
    eos_ids: set[int],
    bidders: Sequence[str] | None = None,
    tape_source: str = "race",
    max_cells: int = 400,
    time_budget_seconds: float | None = None,
    min_area: float = MIN_AREA,
    check_cache: bool = True,
    mode: str = "synthetic_model_test",
    manifest: dict | None = None,
    title: str | None = None,
) -> dict:
    """Depth-first exact atlas over the bid simplex; returns `simplex-atlas-v1` JSON.

    ``title`` is a display field for the Atlas Explorer and does not affect decoding.
    """
    from token_auction_lab.multi_decoder import MultiPrefixDecoder

    n = len(prefixes)
    if n not in (2, 3):
        raise NotImplementedError("Atlases are implemented for n = 2 and n = 3 bidders.")
    if max_cells < 1 or max_tokens < 1:
        raise ValueError("max_cells and max_tokens must be positive.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    bidders = list(bidders or [f"Bidder {i + 1}" for i in range(n)])
    source = TAPE_SOURCES[tape_source]
    rng = np.random.default_rng(seed)
    tapes = [source(rng, n) for _ in range(max_tokens)]
    started = time.perf_counter()
    deadline = None if time_budget_seconds is None else started + time_budget_seconds
    tolerance = cache_tolerance(model)

    cells: dict[int, Cell] = {0: Cell(0, simplex(n), None, None)}
    splits: list[dict] = []
    slivers: list[float] = []
    truncations = {"max_cells": 0, "time_budget": 0}
    forward_passes = 0
    stack = [_Branch(MultiPrefixDecoder(model, prefixes), [cells[0]])]

    def finish(branch: _Branch, group: list[Cell], reason: str, tokens: list[int],
               truncation: dict | None = None) -> None:
        check = None
        if check_cache:
            errors = branch.decoder.check_cache(temperature=temperature, **tolerance)
            check = {"max_absolute_errors": errors,
                     "probability_tvs": branch.decoder.cache_probability_tvs}
        for cell in group:
            cell.token_ids, cell.stop_reason = list(tokens), reason
            cell.truncation, cell.cache_check = truncation, check

    while stack:
        branch = stack.pop()
        decoder = branch.decoder
        step = len(decoder.generated)
        if step >= max_tokens:
            finish(branch, branch.cells, "length", decoder.generated)
            forward_passes += decoder.forward_passes
            continue
        if deadline is not None and time.perf_counter() > deadline:
            truncations["time_budget"] += len(branch.cells)
            finish(branch, branch.cells, "truncated", decoder.generated,
                   {"reason": "time_budget", "step": step})
            forward_passes += decoder.forward_passes
            continue
        auction = NBidderAuction(decoder.distributions(temperature))
        tape = tapes[step]
        groups: dict[int, list[Cell]] = {}
        if auction.is_shared(tape):
            token = auction.shared_token(tape)
            for cell in branch.cells:
                cell.steps.append({"step": step, "branch": "shared", "winner": None,
                                   "token_id": token, "shared_mass": auction.shared_mass})
            groups[token] = list(branch.cells)
        else:
            candidates = auction.candidates(tape)
            exponentials = list(tape.exponentials)
            truncated_here = []
            for cell in branch.cells:
                pieces, dropped = partition(cell.polygon, exponentials, min_area)
                slivers.extend(dropped)
                if not pieces:
                    raise RuntimeError("A cell vanished entirely; geometry tolerance error.")
                if len(pieces) > 1 and len(cells) + len(pieces) - 1 > max_cells:
                    truncated_here.append(cell)
                    continue
                owners = [cell]
                for _ in pieces[1:]:
                    child = Cell(len(cells), cell.polygon, cell.id, step,
                                 [dict(r) for r in cell.steps])
                    cells[child.id] = child
                    owners.append(child)
                if len(pieces) > 1:
                    texts = {t: token_piece(tokenizer, decoder.generated, t)
                             for t in {candidates[w] for w, _ in pieces}}
                    splits.append({
                        "step": step,
                        "parent": cell.id,
                        "children": [c.id for c in owners],
                        "winners": [w for w, _ in pieces],
                        "tokens": [candidates[w] for w, _ in pieces],
                        "E": exponentials,
                        "apex": [e / sum(exponentials) for e in exponentials],
                        "pieces": [{"cell": c.id, "winner": w, "token": candidates[w],
                                    "token_text": texts[candidates[w]]}
                                   for c, (w, _) in zip(owners, pieces)],
                        "prefix_text": decode_text(tokenizer, decoder.generated),
                    })
                for (winner, piece), owner in zip(pieces, owners):
                    owner.polygon = piece
                    owner.steps.append({
                        "step": step, "branch": "contested", "winner": winner,
                        "token_id": candidates[winner], "E": exponentials,
                        "candidates": list(candidates), "shared_mass": auction.shared_mass,
                    })
                    groups.setdefault(candidates[winner], []).append(owner)
            if truncated_here:
                truncations["max_cells"] += len(truncated_here)
                finish(branch, truncated_here, "truncated", decoder.generated,
                       {"reason": "max_cells", "step": step})
        live = [(t, g) for t, g in groups.items() if t not in eos_ids]
        for token, group in groups.items():
            if token in eos_ids:
                finish(branch, group, "eos", decoder.generated + [token])
        decoders = [decoder] + [decoder.fork() for _ in live[1:]] if live else []
        if not live:
            forward_passes += decoder.forward_passes
        children = []
        for (token, group), child_decoder in zip(live, decoders):
            child_decoder.advance(token)
            children.append(_Branch(child_decoder, group))
        stack.extend(reversed(children))

    elapsed = time.perf_counter() - started
    ordered = [cells[i] for i in sorted(cells)]
    total_area = math.fsum(measure(c.polygon) for c in ordered)
    new_per_step = [0] * max_tokens
    for s in splits:
        new_per_step[s["step"]] += len(s["children"]) - 1
    cells_after_step = list(np.cumsum([1 + new_per_step[0]] + new_per_step[1:]).tolist())
    payload_cells = []
    for cell in ordered:
        point = centroid(cell.polygon)
        text = decode_text(tokenizer, cell.token_ids)
        payload_cells.append({
            "id": cell.id,
            "polygon": cell.polygon.tolist(),
            "area": measure(cell.polygon),
            "centroid": point.tolist(),
            "parent": cell.parent,
            "split_step": cell.split_step,
            "token_ids": [int(t) for t in cell.token_ids],
            "text": text,
            "stop_reason": cell.stop_reason,
            "truncation": cell.truncation,
            "steps": cell.steps,
            "winner_counts": [sum(r["winner"] == i for r in cell.steps) for i in range(n)],
            "payments_at_centroid_unit_total_bid": cell_payments(cell.steps, point),
            "outcomes": {f"mentions:{brand}": brand in text for brand in bidders},
            "cache_check": cell.cache_check,
        })
    return {
        "schema": SCHEMA,
        "kind": "text",
        "mode": mode,
        "title": title,
        "label": {"live_model": "LIVE MODEL OUTPUT", "synthetic_model_test":
                  "SYNTHETIC MODEL TEST (random weights)", "synthetic_fixture":
                  "SYNTHETIC FIXTURE"}.get(mode, mode),
        "bidders": bidders,
        "seed": seed,
        "parameter": "lambda_i = bid_i / sum(bids); polygon vertices are barycentric "
                     "(lambda_1..lambda_n); 2-D measure uses (lambda_1, lambda_2)",
        "max_steps": max_tokens,
        "temperature": temperature,
        "tape_source": tape_source,
        "tape_layout": "per step: rng.random(2n+2) = [branch, residual_1..n, race U_1..n, "
                       "shared]; E_i = -log(U_i)" if tape_source == "race" else
                       "per step: auction.RandomTape.draw(rng) mapped by "
                       "RaceTape.from_two_bidder",
        "manifest": manifest or {},
        "cells": payload_cells,
        "splits": splits,
        "verification": [],
        "stats": {
            "cells": len(ordered),
            "distinct_outputs": len({tuple(c["token_ids"]) for c in payload_cells}),
            "splits": len(splits),
            "cells_after_step": cells_after_step,
            "area_total": total_area,
            "simplex_area": simplex_measure(n),
            "dropped_slivers": {"count": len(slivers), "area": math.fsum(slivers),
                                "min_area": min_area},
            "max_cells": max_cells,
            "truncated": any(truncations.values()),
            "truncated_cells": truncations,
            "forward_passes": forward_passes,
            "elapsed_seconds": elapsed,
            "time_budget_seconds": time_budget_seconds,
            "cache_checked": check_cache,
            "cache_tolerance": tolerance if check_cache else None,
        },
    }


def verify(
    atlas: dict,
    model,
    prefixes: list[list[int]],
    *,
    sample: int,
    rng: np.random.Generator | None = None,
    points: str = "centroid",
    eos_ids: set[int],
    cell_ids: Sequence[int] | None = None,
) -> list[dict]:
    """Rerun ordinary n-bidder generation inside sampled cells; compare ids and charges."""
    rng = rng or np.random.default_rng(atlas["seed"] + 1_000_003)
    cells = atlas["cells"]
    if cell_ids is None:
        chosen = sorted(rng.choice(len(cells), size=min(sample, len(cells)), replace=False))
        cell_ids = [cells[k]["id"] for k in chosen]
    by_id = {c["id"]: c for c in cells}
    source = TAPE_SOURCES[atlas["tape_source"]]
    records = []
    for cell_id in cell_ids:
        cell = by_id[cell_id]
        polygon = np.array(cell["polygon"])
        point = (np.array(cell["centroid"]) if points == "centroid"
                 else random_interior_point(polygon, rng))
        truncated = cell["stop_reason"] == "truncated"
        steps = len(cell["steps"]) if truncated else atlas["max_steps"]
        direct = direct_generate(
            model, prefixes, point, seed=atlas["seed"], temperature=atlas["temperature"],
            max_tokens=max(steps, 1), eos_ids=eos_ids, tape_source=source,
        ) if steps else {"token_ids": [], "payments": [0.0] * len(prefixes)}
        expected = cell_payments(cell["steps"], point)
        tokens_ok = direct["token_ids"][: len(cell["token_ids"])] == cell["token_ids"] and (
            truncated or direct["token_ids"] == cell["token_ids"])
        diff = max(abs(a - b) for a, b in zip(direct["payments"], expected))
        payments_ok = diff <= 1e-12 * max(1.0, max(expected))
        records.append({
            "cell": cell_id,
            "lambda": point.tolist(),
            "point": points,
            "truncated_prefix_only": truncated,
            "token_ids_match": tokens_ok,
            "payments_match": payments_ok,
            "max_payment_abs_diff": diff,
            "direct_payments": direct["payments"],
            "matches_direct_run": tokens_ok and payments_ok,
        })
    return records


# --------------------------------------------------------------------------- CLI


def _source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("auction_n.py", "multi_decoder.py", "decoder.py",
                                      "simplex_atlas.py", "prompts.py", "prompts_n.py",
                                      "cli.py", "fact_sheets_n.json", "live_defaults.json")]
    paths.append(root.parent / "auction.py")
    return {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def write_atlases(
    model,
    tokenizer,
    prompts: list[tuple[str, list[int]]],
    *,
    seeds: Sequence[int],
    temperature: float,
    max_tokens: int,
    output: Path,
    bidders: Sequence[str],
    eos_ids: set[int],
    verify_cells: int = 0,
    max_cells: int = 400,
    time_budget_seconds: float | None = None,
    mode: str = "live_model",
    metadata: dict | None = None,
    log: Callable[[dict], None] | None = None,
    title: str | None = None,
) -> dict:
    """One JSON per seed, written as each finishes; index.json rewritten after each."""
    from token_auction_lab.prompts import prompt_hash

    output.mkdir(parents=True, exist_ok=True)
    prefixes = [ids for _, ids in prompts]
    index = {"schema": SCHEMA + "-index", "mode": mode, "bidders": list(bidders),
             "created_at": datetime.now(timezone.utc).isoformat(), "atlases": []}
    for seed in seeds:
        manifest = {
            **(metadata or {}),
            "seed": seed,
            "rng": "numpy.PCG64 via numpy.random.default_rng(seed)",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "sampling_filters": None,
            "sampler": "token_auction_lab.auction_n.NBidderAuction (exponential race)",
            "credit_unit": "simulated",
            "bids": "Varied over the whole simplex; never in prompts; fixed per generation.",
            "prompts": [{"text": text, "sha256": prompt_hash(text), "token_ids": ids}
                        for text, ids in prompts],
            "source_sha256": _source_hashes(),
            "replay_note": "Viewing this file is REPLAY. A seed alone does not guarantee "
                           "identical regeneration across software or hardware.",
        }
        atlas = build_atlas(
            model, tokenizer, prefixes, seed=seed, temperature=temperature,
            max_tokens=max_tokens, eos_ids=eos_ids, bidders=bidders, max_cells=max_cells,
            time_budget_seconds=time_budget_seconds, mode=mode, manifest=manifest,
            title=title,
        )
        if verify_cells:
            started = time.perf_counter()
            atlas["verification"] = verify(atlas, model, prefixes, sample=verify_cells,
                                           eos_ids=eos_ids)
            atlas["stats"]["verification_seconds"] = time.perf_counter() - started
        name = f"atlas-seed{seed}.json"
        (output / name).write_text(json.dumps(atlas, indent=1, ensure_ascii=False,
                                              allow_nan=False) + "\n")
        summary = {
            "seed": seed, "file": name, "cells": atlas["stats"]["cells"],
            "distinct_outputs": atlas["stats"]["distinct_outputs"],
            "truncated": atlas["stats"]["truncated"],
            "area_total": atlas["stats"]["area_total"],
            "elapsed_seconds": atlas["stats"]["elapsed_seconds"],
            "verified": len(atlas["verification"]),
            "verification_ok": all(v["matches_direct_run"] for v in atlas["verification"]),
        }
        index["atlases"].append(summary)
        (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
        if log:
            log(summary)
    return index


def main(argv: Sequence[str] | None = None, *, mode: str = "live_model") -> None:
    """``mode`` stays live_model for real inference; tiny random models pass
    synthetic_model_test (tests and scripts/make_model_test_atlases.py)."""
    import torch

    from token_auction_lab.cli import add_model_arguments, load_model, pinned_revision
    from token_auction_lab.prompts_n import PROMPT_POLICY_N, brands, render_n, scenarios_n
    from token_auction_lab.runner import eos_ids

    available = scenarios_n()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_model_arguments(parser)
    parser.add_argument("--scenario", choices=tuple(available), default="transit-two-hotels")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/simplex-atlas"))
    parser.add_argument("--verify", type=int, default=3, help="Cells re-run per seed.")
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--time-budget-seconds", type=float, default=None,
                        help="Per seed; remaining cells are marked truncated, never dropped.")
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 96:
        parser.error("--max-tokens must be 1..96.")
    pinned = pinned_revision(parser, args)
    scenario = available[args.scenario]
    if len(scenario["advertisers"]) != 3:
        parser.error("Simplex atlases use exactly three advertisers.")
    tokenizer, model, revision, device, dtype_name, load_seconds = load_model(
        args, pinned, args.seeds[0]
    )
    from importlib.metadata import distributions

    metadata = {
        "model_id": args.model,
        "model_revision": revision,
        "tokenizer": args.model,
        "tokenizer_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "model_vocab_size": model.config.vocab_size,
        "device": device,
        "dtype": dtype_name,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "attention_implementation": "sdpa",
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "scenario": args.scenario,
        "scenario_definition": scenario,
        "user_prompt": scenario.get("task", scenario.get("prompt")),
        "advertiser_prompt_policy": PROMPT_POLICY_N,
        "local_files_only": args.local_files_only,
        "model_load_seconds": load_seconds,
        "max_cells": args.max_cells,
        "time_budget_seconds_per_seed": args.time_budget_seconds,
        "eos_token_ids": sorted(eos_ids(model, tokenizer)),
    }
    prompts = [render_n(tokenizer, scenario, i) for i in range(3)]
    write_atlases(
        model, tokenizer, prompts, seeds=args.seeds, temperature=args.temperature,
        max_tokens=args.max_tokens, output=args.output, bidders=brands(scenario),
        eos_ids=eos_ids(model, tokenizer), verify_cells=args.verify,
        max_cells=args.max_cells, time_budget_seconds=args.time_budget_seconds,
        mode=mode, metadata=metadata,
        log=lambda s: print(json.dumps(s), flush=True),
        title=scenario.get("title", args.scenario),
    )


if __name__ == "__main__":
    main()
