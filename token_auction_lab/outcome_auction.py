"""Outcome auctions on stored bid atlases (docs/OUTCOME_AUCTION.md).

Sponsors report a value for one pre-specified, checkable OUTCOME of the whole
generation ("the answer mentions my brand"), instead of bidding per token. A set
of stored ``bid-atlas-v1`` files (one per seed, same prompt, text or image) gives,
exactly for those seeds, the step functions

    P_i(lambda)  share of seeds whose leaf at lambda satisfies sponsor i's outcome
    K_A(lambda)  mean over seeds of sum over A-won steps of u/(1-u)
    K_B(lambda)  mean over seeds of sum over B-won steps of (1-u)/u

with ``lambda = bid_A / (bid_A + bid_B)``. This module ports the piece grid and the
outcome-level mechanisms of ``walkthrough/mechanism_lab.mjs`` (``prepareLab``,
``vcgExAnte``, ``vcgExPost``, ``proportionalMyerson``, ``monotoneChain``) to
Python with the same arithmetic order, so the two agree on the same inputs:

* ``vcg``          outcome VCG over one lambda (ex ante), Clarke payments;
* ``proportional`` lambda = V_A/(V_A+V_B) with outcome-level Myerson payments,
                   after the monotone-chain repair;
* ``leaf-vcg``     per seed, the leaf maximizing reported welfare (ex post),
                   per-seed Clarke payments.

Pure numpy and the standard library: no model is loaded. Values, payments,
revenue and utilities are simulated credits. Truthfulness statements hold with
respect to the sampled seeds and the chosen outcome predicates only; nothing here
measures real advertiser utility. Atlas modes are reported exactly as stored.

CLI::

    python -m token_auction_lab.outcome_auction ATLAS_DIR \\
        --value "Bayview Hotel=40" --value "Lantern House=40" \\
        [--outcome "Bayview Hotel=mentions:Bayview Hotel"] \\
        [--mechanism vcg|proportional|leaf-vcg|all] [--curves] [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

SCHEMA = "bid-atlas-v1"
EPS = 1e-12  # Tie tolerance, as in mechanism_lab.mjs.
MECHANISMS = ("vcg", "proportional", "leaf-vcg")
MODE_LABELS = {
    "live_model": "LIVE MODEL OUTPUT",
    "synthetic_model_test": "SYNTHETIC MODEL TEST (random-weight model: checks the pipeline, "
    "not model quality)",
    "synthetic_fixture": "SYNTHETIC FIXTURE (toy generator, not a model)",
}
# Manifest fields that must agree for atlases to describe the same prompt and setup.
SETUP_KEYS = (
    "model_id", "model", "model_revision", "revision", "tokenizer", "tokenizer_revision",
    "pipeline", "scheduler", "scenario", "user_prompt", "prompts", "temperature",
    "max_tokens", "sampling_filters", "guidance", "steps", "height", "width",
)
NONCLAIMS = (
    "Simulated credits. Values are reported per outcome occurrence; probabilities are "
    "exact for the loaded seeds only. Truthfulness is with respect to these seeds and "
    "these outcome predicates; no claim about real advertiser utility, model quality "
    "or latency."
)

Predicate = Callable[[dict], bool]


# --------------------------------------------------------------------------- loading


@dataclass
class AtlasSet:
    """Stored atlases for one prompt, one per seed, sorted by seed."""

    atlases: list[dict]
    paths: list[str | None]
    skipped: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.atlases[0]["mode"]

    @property
    def kind(self) -> str:
        return self.atlases[0].get("kind", "text")

    @property
    def bidders(self) -> list[str]:
        return list(self.atlases[0]["bidders"])

    @property
    def seeds(self) -> list:
        return [a.get("seed") for a in self.atlases]

    @classmethod
    def from_docs(cls, docs: Sequence[dict], paths: Sequence[str | None] | None = None,
                  skipped: Iterable[str] = ()) -> "AtlasSet":
        if not docs:
            raise ValueError("No bid-atlas-v1 atlases to load.")
        paths = list(paths) if paths is not None else [None] * len(docs)
        for doc, path in zip(docs, paths):
            check_atlas(doc, path)
        check_same_setup(docs, paths)
        order = sorted(range(len(docs)), key=lambda i: _seed_key(docs[i].get("seed")))
        return cls([docs[i] for i in order], [paths[i] for i in order], list(skipped))


def _seed_key(seed):
    return (0, seed, "") if isinstance(seed, (int, float)) else (1, 0, str(seed))


def load_atlases(paths: Sequence[str | Path]) -> AtlasSet:
    """Load atlases from files and/or directories (searched recursively for *.json).

    Inside a directory, JSON files that are not ``bid-atlas-v1`` (indexes, simplex
    atlases, publish records) are skipped and listed; a named file must be one.
    """
    docs, sources, skipped = [], [], []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = sorted(f for f in path.rglob("*.json") if not f.name.startswith("."))
            for f in files:
                doc = json.loads(f.read_text())
                if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
                    docs.append(doc)
                    sources.append(str(f))
                else:
                    skipped.append(str(f))
        else:
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
                schema = doc.get("schema") if isinstance(doc, dict) else None
                raise ValueError(f"{path}: schema {schema!r} is not {SCHEMA!r}.")
            docs.append(doc)
            sources.append(str(path))
    if not docs:
        raise ValueError(f"No {SCHEMA} atlases found in {', '.join(map(str, paths))}.")
    return AtlasSet.from_docs(docs, sources, skipped)


def check_atlas(doc: dict, path: str | None = None) -> None:
    """Schema, two bidders, and leaves that partition (0, 1] in order."""
    where = path or f"seed {doc.get('seed')}"
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{where}: schema {doc.get('schema')!r} is not {SCHEMA!r}.")
    if "mode" not in doc:
        raise ValueError(f"{where}: atlas has no mode label.")
    if len(doc.get("bidders", [])) != 2:
        raise ValueError(f"{where}: outcome auctions here need exactly two bidders.")
    if doc.get("complete") is False or doc.get("pending"):
        raise ValueError(f"{where}: atlas is incomplete (pending intervals).")
    leaves = doc.get("leaves") or []
    if not leaves or leaves[0]["lo"] != 0.0 or leaves[-1]["hi"] != 1.0:
        raise ValueError(f"{where}: leaves must cover (0, 1].")
    for left, right in zip(leaves, leaves[1:]):
        if left["hi"] != right["lo"]:
            raise ValueError(f"{where}: leaves must be sorted and contiguous.")
    if any(not leaf["lo"] < leaf["hi"] for leaf in leaves):
        raise ValueError(f"{where}: every leaf must be a nonempty interval.")


def check_same_setup(docs: Sequence[dict], paths: Sequence[str | None]) -> None:
    """All atlases must share mode, kind, bidders and prompt/model setup; seeds differ."""
    first = docs[0]

    def setup(doc):
        manifest = doc.get("manifest") or {}
        return {
            "mode": doc.get("mode"),
            "kind": doc.get("kind", "text"),
            "bidders": list(doc.get("bidders", [])),
            "max_steps": doc.get("max_steps"),
            **{k: manifest.get(k) for k in SETUP_KEYS},
        }

    ref = setup(first)
    for doc, path in zip(docs[1:], paths[1:]):
        got = setup(doc)
        diff = sorted(k for k in ref if ref[k] != got[k])
        if diff:
            raise ValueError(
                f"{path or doc.get('seed')}: differs from {paths[0] or first.get('seed')} in "
                f"{', '.join(diff)}; an outcome auction needs one prompt and setup."
            )
    seeds = [d.get("seed") for d in docs]
    if len(set(map(repr, seeds))) != len(seeds):
        raise ValueError(f"Duplicate seeds: {seeds}.")


def mode_label(mode: str) -> str:
    """Display label for a stored mode. Unknown modes are shown as stored."""
    return MODE_LABELS.get(mode, f"mode {mode!r} (unrecognized; shown as stored)")


# --------------------------------------------------------------------------- predicates


def outcome_predicate(spec: str) -> Predicate:
    """A leaf test from a stored outcome key.

    ``KEY`` must be a boolean outcome (for example ``mentions:Bayview Hotel``, the exact,
    case-sensitive substring test of ``atlas.outcome_values``). ``KEY>=x``, ``KEY>x``,
    ``KEY<=x`` or ``KEY<x`` thresholds a numeric outcome (for example a stored CLIP
    score). A leaf without the key raises instead of counting as False.
    """
    match = re.fullmatch(r"(.+?)\s*(>=|<=|>|<)\s*([-+0-9.eE]+)", spec)
    key, op, threshold = (match.group(1), match.group(2), float(match.group(3))) \
        if match else (spec, None, None)
    compare = {">=": lambda x: x >= threshold, ">": lambda x: x > threshold,
               "<=": lambda x: x <= threshold, "<": lambda x: x < threshold}.get(op)

    def test(leaf: dict) -> bool:
        outcomes = leaf.get("outcomes") or {}
        if key not in outcomes:
            raise KeyError(
                f"leaf {leaf.get('id')} has no stored outcome {key!r} (stored: "
                f"{sorted(outcomes)}); choose one with --outcome NAME=KEY."
            )
        value = outcomes[key]
        if compare is None:
            if not isinstance(value, bool):
                raise TypeError(f"outcome {key!r} is {value!r}, not boolean; add a threshold.")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"outcome {key!r} is {value!r}, not numeric.")
        return bool(compare(value))

    test.spec = spec
    return test


def mention(brand: str) -> Predicate:
    """Default outcome: the stored exact, case-sensitive ``mentions:<brand>``."""
    return outcome_predicate(f"mentions:{brand}")


# --------------------------------------------------------------------------- lab


@dataclass
class LeafSummary:
    index: int  # position in the atlas's sorted leaves (JS ``index``)
    leaf_id: object
    lo: float
    hi: float
    p0: int  # pieces [p0, p1) lie in this leaf
    p1: int
    xa: int
    xb: int
    ka: float
    kb: float
    na: int
    nb: int


@dataclass
class SeedLeaves:
    seed: object
    leaves: list[LeafSummary]


@dataclass
class OutcomeLab:
    """Exact step functions on the common piece grid of all seeds' leaf boundaries."""

    n: int
    lo: np.ndarray
    hi: np.ndarray
    PA: np.ndarray
    PB: np.ndarray
    KA: np.ndarray
    KB: np.ndarray
    NA: np.ndarray  # mean number of A-won contested steps
    NB: np.ndarray
    seeds: list[SeedLeaves]
    _chain: dict | None = None

    @property
    def m(self) -> int:
        return len(self.lo)

    def curves(self) -> dict:
        keys = ("lo", "hi", "PA", "PB", "KA", "KB", "NA", "NB")
        return {k: getattr(self, k).tolist() for k in keys}


def prepare_lab(atlases: Sequence[dict], test_a: Predicate, test_b: Predicate) -> OutcomeLab:
    """Port of ``prepareLab``: same pieces, same summation order, same values."""
    cuts = {0.0, 1.0}
    for atlas in atlases:
        for leaf in atlas["leaves"]:
            cuts.add(float(leaf["lo"]))
            cuts.add(float(leaf["hi"]))
    edges = sorted(cuts)
    lo, hi = [], []
    for i in range(len(edges) - 1):
        if edges[i + 1] > edges[i]:
            lo.append(edges[i])
            hi.append(edges[i + 1])
    m, n = len(lo), len(atlases)
    acc = {k: [0.0] * (m + 1) for k in ("PA", "PB", "KA", "KB", "NA", "NB")}
    seeds = []
    for atlas in atlases:
        p = 0
        summaries = []
        for index, leaf in enumerate(atlas["leaves"]):
            u_a = sorted(float(s["u"]) for s in leaf["steps"] if s["branch"] == "A")
            u_b = sorted(float(s["u"]) for s in leaf["steps"] if s["branch"] == "B")
            p0 = p
            while p < m and (lo[p] + hi[p]) / 2 <= leaf["hi"]:
                p += 1
            ka = kb = 0.0
            for u in u_a:
                ka = ka + u / (1 - u)
            for u in u_b:
                kb = kb + (1 - u) / u
            s = LeafSummary(index, leaf.get("id", index), float(leaf["lo"]), float(leaf["hi"]),
                            p0, p, 1 if test_a(leaf) else 0, 1 if test_b(leaf) else 0,
                            ka, kb, len(u_a), len(u_b))
            for key, value in (("PA", s.xa), ("PB", s.xb), ("KA", ka), ("KB", kb),
                               ("NA", len(u_a)), ("NB", len(u_b))):
                acc[key][p0] += value
                acc[key][p] -= value
            summaries.append(s)
        seeds.append(SeedLeaves(atlas.get("seed"), summaries))

    def integrate(arr):
        out = np.empty(m)
        run = 0.0
        for i in range(m):
            run += arr[i]
            out[i] = run / n
        return out

    return OutcomeLab(n, np.array(lo), np.array(hi), *(integrate(acc[k]) for k in
                      ("PA", "PB", "KA", "KB", "NA", "NB")), seeds)


def build_lab(atlas_set: AtlasSet, test_a: Predicate | None = None,
              test_b: Predicate | None = None) -> OutcomeLab:
    """Default predicates: exact ``mentions:<bidder>`` outcomes stored in each leaf."""
    a, b = atlas_set.bidders
    return prepare_lab(atlas_set.atlases, test_a or mention(a), test_b or mention(b))


def piece_index(lab: OutcomeLab, lam: float) -> int:
    """Piece containing lambda in (lo, hi]; lambda <= 0 maps to the first piece."""
    a, b = 0, lab.m - 1
    while a < b:
        mid = (a + b) >> 1
        if lam > lab.hi[mid]:
            a = mid + 1
        else:
            b = mid
    return a if lam > lab.lo[a] else 0


def leaf_index_at(seed: SeedLeaves, lam: float) -> int:
    for leaf in seed.leaves:
        if leaf.lo < lam <= leaf.hi:
            return leaf.index
    return 0


def curve_at(lab: OutcomeLab, lam: float) -> dict:
    j = piece_index(lab, lam)
    return {"piece": j, "PA": float(lab.PA[j]), "PB": float(lab.PB[j]),
            "KA": float(lab.KA[j]), "KB": float(lab.KB[j])}


def _closest_in_piece(lab: OutcomeLab, j: int, ref: float) -> float:
    lo, hi = float(lab.lo[j]), float(lab.hi[j])
    if lo < ref <= hi:
        return ref
    if ref > hi:
        return hi
    return lo + (hi - lo) * 1e-6


def _check_values(va: float, vb: float, positive: bool = False) -> tuple[float, float]:
    va, vb = float(va), float(vb)
    for v in (va, vb):
        if not math.isfinite(v) or v < 0 or (positive and v == 0):
            raise ValueError(f"Values must be finite and {'positive' if positive else '>= 0'}.")
    return va, vb


def _default_ref(va: float, vb: float) -> float:
    return va / (va + vb) if va + vb > 0 else 0.5


def _result(name, va, vb, pa, pb, pay_a, pay_b, **extra) -> dict:
    return {
        "mechanism": name,
        **extra,
        "allocation": {"A": float(pa), "B": float(pb)},
        "payments": {"A": float(pay_a), "B": float(pay_b)},
        "utility": {"A": float(va * pa - pay_a), "B": float(vb * pb - pay_b)},
        "revenue": float(pay_a + pay_b),
        "welfare": float(va * pa + vb * pb),
    }


# --------------------------------------------------------------------------- mechanisms


def vcg_ex_ante(lab: OutcomeLab, va: float, vb: float, ref: float | None = None) -> dict:
    """Port of ``vcgExAnte``: one lambda maximizing V_A P_A + V_B P_B, Clarke payments.

    Ties go to the piece nearest ``ref`` (default V_A/(V_A+V_B)), and the lambda is the
    point of the chosen piece nearest ``ref``. Truthful for each sponsor whatever the
    shape of P_i, when the generation's seed is drawn uniformly from the same seeds.
    """
    va, vb = _check_values(va, vb)
    ref = _default_ref(va, vb) if ref is None else float(ref)
    lo, hi, pa, pb = lab.lo, lab.hi, lab.PA, lab.PB

    def dist(k):
        return 0.0 if lo[k] < ref <= hi[k] else min(abs(ref - lo[k]), abs(ref - hi[k]))

    j, max_a, max_b = 0, -math.inf, -math.inf
    for k in range(lab.m):
        w, wj = va * pa[k] + vb * pb[k], va * pa[j] + vb * pb[j]
        if w > wj + EPS or (abs(w - wj) <= EPS and dist(k) < dist(j)):
            j = k
        max_a = max(max_a, va * pa[k])
        max_b = max(max_b, vb * pb[k])
    pay_a, pay_b = max_b - vb * pb[j], max_a - va * pa[j]
    return _result("vcg", va, vb, pa[j], pb[j], pay_a, pay_b,
                   **{"lambda": _closest_in_piece(lab, j, ref), "piece": j,
                      "piece_interval": [float(lo[j]), float(hi[j])], "ref": ref})


def monotone_chain(lab: OutcomeLab) -> dict:
    """Port of ``monotoneChain``: greedy left-to-right chain on which P_A never falls
    and P_B never rises; every piece maps to the last chain piece at or before it."""
    if lab._chain is not None:
        return lab._chain
    mapping, last, remapped = [0] * lab.m, 0, 0.0
    for j in range(lab.m):
        if lab.PA[j] >= lab.PA[last] and lab.PB[j] <= lab.PB[last]:
            last = j
        else:
            remapped += lab.hi[j] - lab.lo[j]
        mapping[j] = last
    lab._chain = {"map": mapping, "remapped_measure": float(remapped)}
    return lab._chain


def monotonicity(lab: OutcomeLab) -> dict:
    drop_a = rise_b = 0.0
    n_a = n_b = 0
    for j in range(1, lab.m):
        if lab.PA[j] < lab.PA[j - 1]:
            n_a += 1
            drop_a = max(drop_a, float(lab.PA[j - 1] - lab.PA[j]))
        if lab.PB[j] > lab.PB[j - 1]:
            n_b += 1
            rise_b = max(rise_b, float(lab.PB[j] - lab.PB[j - 1]))
    return {"PA_nondecreasing": n_a == 0, "PB_nonincreasing": n_b == 0,
            "violations_A": n_a, "violations_B": n_b, "max_drop_A": drop_a, "max_rise_B": rise_b}


def _z_intervals_a(lab: OutcomeLab, rival: float) -> list[tuple[float, float]]:
    return [(rival * lo / (1 - lo), math.inf if hi >= 1 else rival * hi / (1 - hi))
            for lo, hi in zip(lab.lo.tolist(), lab.hi.tolist())]


def _z_intervals_b(lab: OutcomeLab, rival: float) -> list[tuple[float, float]]:
    return [(rival * (1 - hi) / hi, math.inf if lo == 0 else rival * (1 - lo) / lo)
            for lo, hi in zip(lab.lo.tolist(), lab.hi.tolist())]


def _myerson_side(own: Sequence[float], intervals, value: float, truthful_piece: int) -> dict:
    """Port of ``myersonSide``: x(z) = own[j] on report intervals (zl, zh]."""
    order = sorted(range(len(own)), key=lambda i: intervals[i][0])
    integral, truthful, best = 0.0, None, -math.inf
    for j in order:
        zl, zh = intervals[j]
        if j == truthful_piece:
            utility = own[j] * (value - zl) + integral
            truthful = {"utility": utility, "pay": own[j] * value - utility}
        if not zh > zl:
            continue
        best = max(best, own[j] * (value - zl) + integral)
        integral += own[j] * (zh - zl)
        if not math.isfinite(integral):
            break
    return {**truthful, "regret": max(0.0, best - truthful["utility"])}


def proportional_myerson(lab: OutcomeLab, va: float, vb: float, monotone: bool = True) -> dict:
    """Port of ``proportionalMyerson``: lambda = V_A/(V_A+V_B), Myerson payments.

    With ``monotone`` (the default here), pieces are mapped onto ``monotone_chain`` so
    x_A is nondecreasing and x_B nonincreasing in the reports, and the payments are
    exactly truthful; the generation runs at ``implemented_lambda`` in the mapped piece.
    ``regret`` is the exact gain from the best misreport against a truthful rival.
    """
    va, vb = _check_values(va, vb, positive=True)
    lam = va / (va + vb)
    j = piece_index(lab, lam)
    k = monotone_chain(lab)["map"] if monotone else None
    pa = [float(lab.PA[i]) for i in k] if k else lab.PA.tolist()
    pb = [float(lab.PB[i]) for i in k] if k else lab.PB.tolist()
    side_a = _myerson_side(pa, _z_intervals_a(lab, vb), va, j)
    side_b = _myerson_side(pb, _z_intervals_b(lab, va), vb, j)
    used = k[j] if k else j
    return _result(
        "proportional" if monotone else "proportional-raw", va, vb, pa[j], pb[j],
        side_a["pay"], side_b["pay"],
        **{"lambda": lam,
           "implemented_lambda": _closest_in_piece(lab, used, lam) if k else lam,
           "piece": used, "piece_interval": [float(lab.lo[used]), float(lab.hi[used])],
           "regret": {"A": side_a["regret"], "B": side_b["regret"]},
           "monotonicity": monotonicity(lab),
           "remapped_measure": monotone_chain(lab)["remapped_measure"] if k else 0.0},
    )


def leaf_vcg(lab: OutcomeLab, va: float, vb: float, ref: float | None = None,
             compare_lambda: float | None = None) -> dict:
    """Port of ``vcgExPost``: per seed, the leaf maximizing V_A x_A + V_B x_B (ties to
    the leaf containing ``ref``, then the nearest), with per-seed Clarke payments.
    Averages are over the seeds; ``per_seed`` lists each choice."""
    va, vb = _check_values(va, vb)
    ref = _default_ref(va, vb) if ref is None else float(ref)
    compare_lambda = ref if compare_lambda is None else float(compare_lambda)
    tot_pa = tot_pb = tot_pay_a = tot_pay_b = moved = 0.0
    changed = 0
    per_seed = []

    def dist(leaf):
        if leaf.lo < ref <= leaf.hi:
            return 0.0
        return min(abs(ref - leaf.lo), abs(ref - leaf.hi))

    for seed in lab.seeds:
        best, max_a, max_b = seed.leaves[0], 0.0, 0.0
        for leaf in seed.leaves:
            w, wb = va * leaf.xa + vb * leaf.xb, va * best.xa + vb * best.xb
            if w > wb + EPS or (abs(w - wb) <= EPS and dist(leaf) < dist(best)):
                best = leaf
            max_a = max(max_a, va * leaf.xa)
            max_b = max(max_b, vb * leaf.xb)
        pay_a, pay_b = max_b - vb * best.xb, max_a - va * best.xa
        tot_pa += best.xa
        tot_pb += best.xb
        tot_pay_a += pay_a
        tot_pay_b += pay_b
        if best.index != leaf_index_at(seed, compare_lambda):
            changed += 1
        moved += dist(best)
        per_seed.append({"seed": seed.seed, "leaf": best.index, "leaf_id": best.leaf_id,
                         "interval": [best.lo, best.hi], "A": best.xa, "B": best.xb,
                         "payments": {"A": pay_a, "B": pay_b}})
    n = lab.n
    return _result("leaf-vcg", va, vb, tot_pa / n, tot_pb / n, tot_pay_a / n, tot_pay_b / n,
                   **{"lambda": None, "ref": ref,
                      "output_changed_vs_ref": changed / n, "mean_lambda_distance": moved / n,
                      "per_seed": per_seed})


def run_mechanism(lab: OutcomeLab, name: str, va: float, vb: float) -> dict:
    if name == "vcg":
        return vcg_ex_ante(lab, va, vb)
    if name == "proportional":
        return proportional_myerson(lab, va, vb, monotone=True)
    if name == "proportional-raw":
        return proportional_myerson(lab, va, vb, monotone=False)
    if name == "leaf-vcg":
        return leaf_vcg(lab, va, vb)
    raise ValueError(f"Unknown mechanism {name!r}.")


# --------------------------------------------------------------------------- CLI


def _parse_pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, rest = item.rpartition("=") if what == "value" else item.partition("=")
        if not sep or not name:
            raise SystemExit(f"--{what} expects NAME=..., got {item!r}.")
        out[name] = rest
    return out


def _fmt(x, width=9, digits=4) -> str:
    if x is None:
        return "per seed".rjust(width)
    return f"{x:{width}.{digits}f}"


def report(atlas_set: AtlasSet, lab: OutcomeLab, values: Mapping[str, float],
           outcomes: Mapping[str, str], results: Sequence[dict], curves: bool = False) -> str:
    a, b = atlas_set.bidders
    leaves = sum(len(x["leaves"]) for x in atlas_set.atlases)
    title = atlas_set.atlases[0].get("title")
    lines = [
        f"MODE: {atlas_set.mode} -- {mode_label(atlas_set.mode)}",
        f"Outcome auction on {lab.n} stored {SCHEMA} atlas(es), kind {atlas_set.kind}"
        + (f", {title!r}" if title else ""),
        f"Seeds ({lab.n}): {', '.join(map(str, atlas_set.seeds))}; leaves: {leaves}; "
        f"pieces: {lab.m}",
        f"  A = {a}: value {values[a]:g} per occurrence of {outcomes[a]!r}",
        f"  B = {b}: value {values[b]:g} per occurrence of {outcomes[b]!r}",
        NONCLAIMS,
    ]
    if lab.n == 1:
        lines.append("Only 1 seed: P_i are 0/1 indicators for that seed, not probabilities "
                     "over prompts or tapes.")
    lines.append("")
    header = (f"{'mechanism':<14}{'lambda':>9}{'P(A)':>9}{'P(B)':>9}{'pay A':>9}{'pay B':>9}"
              f"{'revenue':>9}{'welfare':>9}{'util A':>9}{'util B':>9}")
    lines += [header, "-" * len(header)]
    for r in results:
        lam = r.get("implemented_lambda", r.get("lambda"))
        lines.append(
            f"{r['mechanism']:<14}{_fmt(lam)}{_fmt(r['allocation']['A'])}"
            f"{_fmt(r['allocation']['B'])}{_fmt(r['payments']['A'])}{_fmt(r['payments']['B'])}"
            f"{_fmt(r['revenue'])}{_fmt(r['welfare'])}{_fmt(r['utility']['A'])}"
            f"{_fmt(r['utility']['B'])}"
        )
    for r in results:
        if r["mechanism"] == "vcg":
            lo, hi = r["piece_interval"]
            lines.append(f"vcg: piece {r['piece']} = ({lo:.6g}, {hi:.6g}], generate at "
                         f"lambda {r['lambda']:.6g} on a seed drawn from these {lab.n}.")
        elif r["mechanism"].startswith("proportional"):
            mono = r["monotonicity"]
            lines.append(
                f"{r['mechanism']}: reference lambda {r['lambda']:.6g}, implemented "
                f"{r['implemented_lambda']:.6g}; P_A drops {mono['violations_A']}, P_B rises "
                f"{mono['violations_B']}; chain remaps lambda-measure "
                f"{r['remapped_measure']:.4g}; misreport regret A {r['regret']['A']:.3g}, "
                f"B {r['regret']['B']:.3g}."
            )
        elif r["mechanism"] == "leaf-vcg":
            lines.append(f"leaf-vcg: output differs from lambda {r['ref']:.6g} in "
                         f"{100 * r['output_changed_vs_ref']:.1f}% of seeds.")
            shown = r["per_seed"][:20]
            for s in shown:
                lo, hi = s["interval"]
                lines.append(f"  seed {s['seed']}: leaf id {s['leaf_id']} ({lo:.6g}, {hi:.6g}] "
                             f"A={s['A']} B={s['B']} pay A {s['payments']['A']:.4g} "
                             f"pay B {s['payments']['B']:.4g}")
            if len(r["per_seed"]) > len(shown):
                lines.append(f"  ... {len(r['per_seed']) - len(shown)} more seeds in --json.")
    if curves:
        lines += ["", f"{'lo':>10}{'hi':>10}{'P_A':>8}{'P_B':>8}{'K_A':>10}{'K_B':>10}"]
        for j in range(lab.m):
            lines.append(f"{lab.lo[j]:10.6f}{lab.hi[j]:10.6f}{lab.PA[j]:8.4f}{lab.PB[j]:8.4f}"
                         f"{lab.KA[j]:10.4f}{lab.KB[j]:10.4f}")
    return "\n".join(lines)


def _sources(atlas_set: AtlasSet) -> list[dict]:
    out = []
    for doc, path in zip(atlas_set.atlases, atlas_set.paths):
        manifest = doc.get("manifest") or {}
        entry = {
            "path": path, "seed": doc.get("seed"), "mode": doc.get("mode"),
            "model": manifest.get("model_id", manifest.get("model")),
            "model_revision": manifest.get("model_revision", manifest.get("revision")),
            "tokenizer": manifest.get("tokenizer", manifest.get("tokenizers")),
            "verification_status": doc.get("verification_status"),
        }
        if path:
            entry["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        out.append(entry)
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        prog="python -m token_auction_lab.outcome_auction",
        description="Outcome auctions (VCG, proportional Myerson, leaf VCG) on stored "
        "bid-atlas-v1 files. Simulated credits.",
    )
    parser.add_argument("atlases", nargs="+", help="atlas files or directories (one prompt)")
    parser.add_argument("--value", action="append", default=[], metavar="NAME=V",
                        help="reported value of NAME's outcome; give one per bidder")
    parser.add_argument("--outcome", action="append", default=[], metavar="NAME=KEY",
                        help="stored outcome key for NAME (default mentions:NAME); "
                        "KEY>=x thresholds a numeric outcome")
    parser.add_argument("--mechanism", default="all", choices=[*MECHANISMS, "all"])
    parser.add_argument("--curves", action="store_true", help="print the step functions")
    parser.add_argument("--json", type=Path, default=None, metavar="OUT")
    args = parser.parse_args(argv)

    try:
        atlas_set = load_atlases(args.atlases)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Cannot load atlases: {exc}") from exc
    bidders = atlas_set.bidders
    raw_values = _parse_pairs(args.value, "value")
    raw_outcomes = _parse_pairs(args.outcome, "outcome")
    for name in [*raw_values, *raw_outcomes]:
        if name not in bidders:
            raise SystemExit(f"Unknown bidder {name!r}; the atlases' bidders are {bidders}.")
    missing = [name for name in bidders if name not in raw_values]
    if missing:
        raise SystemExit(f"Missing --value for {missing}.")
    try:
        values = {name: float(raw_values[name]) for name in bidders}
    except ValueError as exc:
        raise SystemExit(f"Bad --value: {exc}") from exc
    outcomes = {name: raw_outcomes.get(name, f"mentions:{name}") for name in bidders}
    try:
        lab = prepare_lab(atlas_set.atlases, outcome_predicate(outcomes[bidders[0]]),
                          outcome_predicate(outcomes[bidders[1]]))
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"Outcome predicate failed: {exc.args[0]}") from exc
    names = MECHANISMS if args.mechanism == "all" else (args.mechanism,)
    va, vb = values[bidders[0]], values[bidders[1]]
    results = [run_mechanism(lab, name, va, vb) for name in names]
    print(report(atlas_set, lab, values, outcomes, results, curves=args.curves))
    out = {
        "schema": "outcome-auction-v1",
        "mode": atlas_set.mode,
        "mode_label": mode_label(atlas_set.mode),
        "kind": atlas_set.kind,
        "credit_unit": "simulated",
        "nonclaims": NONCLAIMS,
        "bidders": {"A": bidders[0], "B": bidders[1]},
        "values": {"A": va, "B": vb},
        "outcomes": {"A": outcomes[bidders[0]], "B": outcomes[bidders[1]]},
        "seeds": atlas_set.seeds,
        "sources": _sources(atlas_set),
        "skipped_files": atlas_set.skipped,
        "pieces": lab.m,
        "curves": lab.curves(),
        "results": results,
        "generator": "token_auction_lab.outcome_auction",
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=_json_default) + "\n")
    return out


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


if __name__ == "__main__":
    main(sys.argv[1:])
