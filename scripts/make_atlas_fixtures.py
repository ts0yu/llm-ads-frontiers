"""Write SYNTHETIC FIXTURES for the Atlas Explorer: python -m scripts.make_atlas_fixtures

Nothing here runs a language or image model. The "models" are hand-written toy
corpora (a weighted prefix tree of sentences) and a toy Gaussian step schedule.
The interval search, contest uniforms and branch records follow MOONSHOTS.md
sections 1 and 3, using the reference sampler in auction.py for two bidders, so
the leaf structure is internally consistent and checkable. The text, images and
CLIP-like scores are placeholders, labeled mode "synthetic_fixture".

Pillow is needed only for regenerating the placeholder images.
Outputs: walkthrough/atlas_data/{index.json, fixture-*.json, fixture-image-cafe/*.jpg}
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from auction import RandomTape, TwoBidderAuction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "walkthrough" / "atlas_data"
EOS = 2
TOKEN_RE = re.compile(r" ?[A-Za-z0-9']+| ?[^\sA-Za-z0-9']")
FIXTURE_NOTE = (
    "SYNTHETIC FIXTURE for UI development. Toy weighted-corpus model, not a language "
    "model; token IDs come from a toy word tokenizer. Charges are simulated credits."
)


# --------------------------------------------------------------------------- toy model
STYLE = {  # stylistic word preferences per brand; anything missing weighs 1
    "Bayview Hotel": {"stunning": 3, "sweeping": 2, "wonderful": 2.5, "Exciting": 2,
                      "right in": 2.5, "easiest": 2, "Book now at": 2},
    "Lantern House": {"peaceful": 2.5, "unwind": 2.5, "simplest": 2.5, "comfortable": 2,
                      "Hello and welcome": 2, "near": 2, "with luggage": 2},
    "Harbor Loft": {"practical": 2.5, "close to": 2, "most convenient": 2, "Great": 2},
}
GROUP_RE = re.compile(r"\{([^{}]*)\}")


def expand(template: str):
    """All expansions of {x|y} groups, with the chosen alternatives."""
    match = GROUP_RE.search(template)
    if not match:
        return [(template, [])]
    out = []
    for alt in match.group(1).split("|"):
        rest = template[:match.start()] + alt + template[match.end():]
        out += [(text, [alt] + chosen) for text, chosen in expand(rest)]
    return out


class SlotModel:
    """Toy sentence-slot model: one weight column per bidder, never an LLM.

    The response is one sentence from each slot in order, then EOS. Within a slot,
    p_i(t | prefix) is proportional to the bidder-i weight of the slot sentences
    that extend the partial sentence with t. A sentence's weight depends on its
    style words and on which brands earlier sentences mentioned. All weights are
    positive, so any prefix the sampler can produce stays in every bidder's support.
    """

    def __init__(self, brands: list[str], slots, style_power: float = 1.0):
        self.brands, self.pieces, self.ids = brands, {EOS: "<eos>"}, {}
        self.slots = []
        for alternatives in slots:
            expanded = []
            for template, base, mentions in alternatives:
                for text, chosen in expand(template):
                    pieces = TOKEN_RE.findall(text)
                    assert "".join(pieces) == text, text
                    style = [math.prod(STYLE.get(b, {}).get(c, 1) ** style_power
                                       for c in chosen)
                             for b in brands]
                    expanded.append(([self.token_id(t) for t in pieces],
                                     [x * y for x, y in zip(base, style)], set(mentions)))
            for i, (seq, _, _) in enumerate(expanded):
                for j, (other, _, _) in enumerate(expanded):
                    assert i == j or other[:len(seq)] != seq, "slot sentence is a prefix"
            self.slots.append(expanded)
        self.vocab_size = 1000 + len(self.ids)
        self.cache: dict[tuple[int, ...], np.ndarray] = {}

    def token_id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = 1000 + len(self.ids)
            self.pieces[self.ids[piece]] = piece
        return self.ids[piece]

    def decode(self, ids) -> str:
        return "".join(self.pieces[i] for i in ids if i != EOS)

    def weight(self, base, mentions, mentioned):
        # Each advertiser mildly dislikes a sentence that praises only rival brands.
        return [w * (.55 if mentions and brand not in mentions | mentioned else 1)
                for w, brand in zip(base, self.brands)]

    def distributions(self, prefix) -> np.ndarray:
        key = tuple(prefix)
        if key in self.cache:
            return self.cache[key]
        dist = np.zeros((len(self.brands), self.vocab_size))
        pos, mentioned = 0, set()
        for alternatives in self.slots:
            rest = key[pos:]
            done = [a for a in alternatives if tuple(a[0]) == rest[:len(a[0])]]
            if done:
                pos += len(done[0][0])
                mentioned |= done[0][2]
                continue
            for seq, base, mentions in alternatives:
                if len(seq) > len(rest) and tuple(seq[:len(rest)]) == rest:
                    dist[:, seq[len(rest)]] += self.weight(base, mentions, mentioned)
            break
        else:
            assert pos == len(key), "tokens after the final slot"
            dist[:, EOS] = 1.0
        totals = dist.sum(axis=1, keepdims=True)
        assert (totals > 0).all(), "prefix left the toy grammar"
        self.cache[key] = dist / totals
        return self.cache[key]


def hotel_model(brands: list[str], url: str, style_power: float = 1.0) -> SlotModel:
    """Plausible hotel-ad sentences. Base weight column i favours brands[i]."""
    n = len(brands)

    def own(i, strong=6.0, weak=0.35):
        return [strong if j == i else weak for j in range(n)]

    even = [1.0] * n
    facts = {
        "Bayview Hotel": [
            " Stay at Bayview Hotel, {right in|in} the Fort Mason area, with "
            "{beautiful|stunning|sweeping} views of the bay.",
            " Stay at Bayview Hotel, rated 4.9 stars on Google and {close to|near} Fort Mason.",
            " For a hotel {close to|near} Fort Mason, consider Bayview Hotel and its "
            "{beautiful|stunning} views of the bay.",
        ],
        "Lantern House": [
            " Stay at Lantern House, a hotel in the Fort Mason area with a "
            "{quiet|peaceful} courtyard.",
            " Stay at Lantern House in the Fort Mason area, where you can {rest|unwind} "
            "in a quiet courtyard.",
            " For a hotel {close to|near} Fort Mason, consider Lantern House and its "
            "{quiet|peaceful} courtyard.",
        ],
        "Harbor Loft": [
            " Stay at Harbor Loft, a small hotel near Fort Mason with a rooftop reading room.",
            " For a hotel {close to|near} Fort Mason, consider Harbor Loft and its rooftop "
            "reading room.",
        ],
    }
    openers = [("{Welcome|Hello and welcome} to San Francisco!", even, ()),
               ("{Great|Exciting} choice for a conference trip!", even, ()),
               ("Welcome to the city for your Fort Mason conference!", even, ())]
    stays = [(s, own(i), (b,)) for i, b in enumerate(brands) for s in facts[b]]
    pair = [1.4 if j < 2 else .3 for j in range(n)]
    stays += [(f" Both {brands[0]} and {brands[1]} are in the Fort Mason area"
               "{.|, so pick the one that fits your style.}", pair, tuple(brands[:2])),
              (f" Both {brands[1]} and {brands[0]} are in the Fort Mason area.", pair,
               tuple(brands[:2]))]
    rides = [(" From SFO, an Uber is the {simplest|easiest} way to get there{| with luggage}.",
              even, ()),
             (" From SFO, a rideshare like Uber is the {easiest|most convenient} option on "
              "arrival day.", even, ()),
             (" An Uber from SFO is a {practical|comfortable} choice after your flight.",
              even, ())]
    closers = [(" Enjoy the conference!", even, ()),
               (" Have a {great|wonderful} trip!", even, ())]
    if "Lantern House" in brands:
        rides.append((" Lantern House offers luggage storage, so you can drop your bags "
                      "before the conference.", own(brands.index("Lantern House"), 3.0, .2),
                      ("Lantern House",)))
    if "Bayview Hotel" in brands:
        closers.append((f" {{Book at|Book now at}} {url} and enjoy the bay views.",
                        own(brands.index("Bayview Hotel"), 4.0, .2), ("Bayview Hotel",)))
    if "Harbor Loft" in brands:
        closers.append((" Unwind in the Harbor Loft reading room after a long day.",
                        own(brands.index("Harbor Loft"), 4.0, .2), ("Harbor Loft",)))
    return SlotModel(brands, [openers, stays, rides, closers], style_power)


# ------------------------------------------------------------------- two-bidder search
def interval_search(step_fn, seed: int, max_steps: int):
    """MOONSHOTS.md section 1. step_fn(state, tape) -> dict with branch info.

    step_fn returns {"shared": bool, "token": t (shared), "a": t, "b": t (contested),
    "shared_mass": C, "extra": {...}} and never sees bids.
    """
    rng = np.random.default_rng(seed)
    root = {"id": 0, "lo": 0.0, "hi": 1.0, "parent": None, "split_step": None,
            "tokens": [], "steps": [], "stop": None}
    branches, splits = [root], []
    for k in range(max_steps):
        tape = RandomTape.draw(rng)  # step k's tape is the k-th draw, regardless of bids
        live = [b for b in branches if b["stop"] is None]
        if not live:
            break
        for br in live:
            r = step_fn(br["tokens"], tape, k)
            base = {"step": k, "u": tape.contest, "shared_mass": r["shared_mass"], **r["extra"]}
            if r["shared"]:
                take(br, r["token"], {**base, "branch": "shared"})
                continue
            u = tape.contest
            if u <= br["lo"]:
                take(br, r["a"], {**base, "branch": "A"})
            elif u >= br["hi"]:
                take(br, r["b"], {**base, "branch": "B"})
            else:
                child = deepcopy(br)
                child.update(id=len(branches), lo=u, parent=br["id"], split_step=k)
                splits.append({"step": k, "u": u, "parent": br["id"], "child": child["id"],
                               "b_token": r["b"], "a_token": r["a"], "prefix": list(br["tokens"])})
                br["hi"] = u
                take(br, r["b"], {**base, "branch": "B"})
                take(child, r["a"], {**base, "branch": "A"})
                branches.append(child)
    for br in branches:
        br["stop"] = br["stop"] or "length"
    return sorted(branches, key=lambda b: b["lo"]), splits


def take(branch, token, record):
    branch["steps"].append(record)
    if token is not None:
        branch["tokens"].append(token)
        if token == EOS:
            branch["stop"] = "eos"


def text_step(model: SlotModel):
    def step(prefix, tape, k):
        pa, pb = model.distributions(prefix)
        auction = TwoBidderAuction(pa, pb)
        alloc = auction.resolve(1.0, 1.0, tape)  # only branch and candidates are read
        if alloc.branch == "shared":
            return {"shared": True, "token": alloc.token_id, "shared_mass": auction.shared_mass,
                    "extra": {}}
        return {"shared": False, "a": alloc.candidate_a, "b": alloc.candidate_b,
                "shared_mass": auction.shared_mass, "extra": {}}
    return step


def direct_text_run(model: SlotModel, seed: int, a: float, b: float, max_steps: int):
    rng, tokens = np.random.default_rng(seed), []
    for _ in range(max_steps):
        pa, pb = model.distributions(tokens)
        tokens.append(TwoBidderAuction(pa, pb).resolve(a, b, RandomTape.draw(rng)).token_id)
        if tokens[-1] == EOS:
            break
    return tokens


def text_outcomes(text: str, keys: list[str]) -> dict:
    return {key: key.split(":", 1)[1] in text for key in keys}


def build_text_atlas(seed: int, max_steps: int = 96):
    brands = ["Bayview Hotel", "Lantern House"]
    model = hotel_model(brands, "bayviewhotel.com", style_power=3)
    leaves, splits = interval_search(text_step(model), seed, max_steps)
    keys = [f"mentions:{b}" for b in brands] + ["mentions:bayviewhotel.com"]
    out_leaves = []
    for br in leaves:
        text = model.decode(br["tokens"])
        for s in br["steps"]:
            s["token_id"] = (br["tokens"][s["step"]] if s["step"] < len(br["tokens"]) else None)
        out_leaves.append({
            "id": br["id"], "lo": br["lo"], "hi": br["hi"], "parent": br["parent"],
            "split_step": br["split_step"], "token_ids": br["tokens"], "text": text,
            "image": None, "stop_reason": br["stop"], "steps": br["steps"],
            "outcomes": text_outcomes(text, keys),
        })
    out_splits = [{
        "step": s["step"], "u": s["u"], "parent": s["parent"], "child": s["child"],
        "b_token": s["b_token"], "a_token": s["a_token"],
        "b_token_text": model.pieces[s["b_token"]], "a_token_text": model.pieces[s["a_token"]],
        "prefix_text": model.decode(s["prefix"]),
    } for s in splits]
    verification = []
    for leaf in out_leaves:
        for lam in sorted({(leaf["lo"] + leaf["hi"]) / 2, leaf["hi"]}):
            tokens = direct_text_run(model, seed, lam, 1 - lam, max_steps)
            verification.append({"leaf": leaf["id"], "lambda": lam,
                                 "matches_direct_run": tokens == leaf["token_ids"]})
    atlas = {
        "schema": "bid-atlas-v1", "kind": "text", "mode": "synthetic_fixture",
        "title": "Fort Mason conference stay (toy corpus)",
        "bidders": brands, "seed": seed, "parameter": "lambda = bid_A / (bid_A + bid_B)",
        "max_steps": max_steps,
        "manifest": manifest(seed, {
            "model": "toy weighted-corpus model (scripts/make_atlas_fixtures.py), not an LLM",
            "revision": corpus_hash(model), "tokenizer": "toy word tokenizer; fake token IDs",
            "prompts": {"user": "I'm visiting Fort Mason in San Fransisco for a conference. "
                                "I get into SFO airport on Oct. 11th. Where should I stay? "
                                "How should I get there? Uber?"},
            "temperature": None, "sampler": "auction.py TwoBidderAuction, full distributions",
            "eos_token_id": EOS,
        }),
        "leaves": out_leaves, "splits": out_splits, "verification": verification,
    }
    return atlas


# ------------------------------------------------------------------------ image atlas
IMAGE_BRANDS = ["Tidewater Coffee", "Ember Roasters"]  # A = teal café, B = red café


def image_step(steps: int):
    def step(history, tape, k):
        # Toy Gaussian conditioning gap delta = |mu_B - mu_A| / s, larger at early steps.
        h = int(hashlib.sha256(bytes(history)).hexdigest()[:8], 16) / 0xFFFFFFFF
        delta = 3.2 * math.exp(-k / 6.5) * (0.75 + 0.5 * h)
        shared_mass = math.erfc(delta / (2 * math.sqrt(2)))  # C = 2*Phi(-delta/2)
        extra = {"delta": delta}
        if tape.branch < shared_mass:
            return {"shared": True, "token": 0, "shared_mass": shared_mass, "extra": extra}
        return {"shared": False, "a": 1, "b": 3, "shared_mass": shared_mass, "extra": extra}
    return step


def build_image_atlas(seed: int, steps: int = 24):
    leaves, splits = interval_search(image_step(steps), seed, steps)
    directory = "fixture-image-cafe"
    (OUT / directory).mkdir(parents=True, exist_ok=True)
    for old in (OUT / directory).glob("*.jpg"):
        old.unlink()
    rng = np.random.default_rng(seed + 1000)
    out_leaves = []
    for index, br in enumerate(leaves):
        score = sum((1 if s["branch"] == "A" else -1) * s["delta"]
                    for s in br["steps"] if s["branch"] != "shared")
        mix = 1 / (1 + math.exp(-0.9 * score))  # 1 = all bidder A (teal), 0 = all B (red)
        detail = int(hashlib.sha256(bytes(br["tokens"])).hexdigest()[:8], 16)
        path = f"{directory}/leaf-{index:02d}.jpg"
        render_cafe(OUT / path, mix, detail, index, br)
        clip_a = 0.19 + 0.09 * mix + float(rng.normal(0, 0.012))
        clip_b = 0.19 + 0.09 * (1 - mix) + float(rng.normal(0, 0.012))
        for s in br["steps"]:
            s.pop("token_id", None)
        out_leaves.append({
            "id": br["id"], "lo": br["lo"], "hi": br["hi"], "parent": br["parent"],
            "split_step": br["split_step"], "token_ids": [], "text": None, "image": path,
            "stop_reason": "steps", "steps": br["steps"],
            "outcomes": {
                f"clip:{IMAGE_BRANDS[0]}": round(clip_a, 4),
                f"clip:{IMAGE_BRANDS[1]}": round(clip_b, 4),
                f"clip_top:{IMAGE_BRANDS[0]}": clip_a > clip_b,
            },
        })
    out_splits = [{"step": s["step"], "u": s["u"], "parent": s["parent"], "child": s["child"],
                   "b_token": None, "a_token": None, "prefix_text": None} for s in splits]
    return {
        "schema": "bid-atlas-v1", "kind": "image", "mode": "synthetic_fixture",
        "title": "Rainy-street café storefront (placeholder images)",
        "bidders": IMAGE_BRANDS, "seed": seed, "parameter": "lambda = bid_A / (bid_A + bid_B)",
        "max_steps": steps,
        "manifest": manifest(seed, {
            "model": "toy Gaussian step schedule + Pillow placeholder renderer, not a "
                     "diffusion model",
            "revision": None, "tokenizer": None, "sampler": "simulated Euler-ancestral steps",
            "prompts": {"user": "A photo of a cozy café storefront on a rainy evening street.",
                        "A": "…with a teal Tidewater Coffee sign.",
                        "B": "…with a red Ember Roasters sign."},
            "outcome_note": "clip:* values are SYNTHETIC numbers imitating CLIP zero-shot "
                            "scores, a noisy heuristic even when real.",
            "image_size": [640, 400], "pillow": pillow_version(),
        }),
        "leaves": out_leaves, "splits": out_splits, "verification": [],
    }


def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def font(size, bold=True):
    from PIL import ImageFont

    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names + ["/usr/share/fonts/truetype/dejavu/" + n for n in names]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_cafe(path: Path, mix: float, detail: int, index: int, leaf: dict):
    from PIL import Image, ImageDraw, ImageFilter

    w, h = 640, 400
    red, teal = (186, 52, 44), (24, 132, 128)
    img = Image.new("RGB", (w, h))
    px = ImageDraw.Draw(img)
    sky_top = lerp((34, 30, 52), (40, 56, 92), mix)
    sky_low = lerp((230, 140, 110), (150, 120, 140), mix)
    for y in range(h):
        px.line([(0, y), (w, y)], fill=lerp(sky_top, sky_low, y / h))
    # Neighbouring buildings and the storefront.
    px.rectangle([0, 70, 110, 330], fill=(54, 58, 70))
    px.rectangle([530, 50, 640, 330], fill=(60, 54, 66))
    facade = lerp(red, teal, mix)
    px.rectangle([110, 60, 530, 330], fill=facade)
    px.rectangle([110, 60, 530, 82], fill=lerp(facade, (20, 20, 24), .35))
    sign_w = 300
    px.rounded_rectangle([320 - sign_w // 2, 96, 320 + sign_w // 2, 146], 10,
                         fill=(250, 244, 230), outline=lerp(facade, (0, 0, 0), .4), width=3)
    name = IMAGE_BRANDS[0] if mix >= .5 else IMAGE_BRANDS[1]
    if .38 < mix < .62:
        name = "TIDEWATER · EMBER"
    f = font(26)
    tw = px.textlength(name.upper(), font=f)
    px.text((320 - tw / 2, 106), name.upper(), font=f, fill=lerp(facade, (0, 0, 0), .45))
    stripes = 6 + detail % 5
    sw = 420 / stripes
    for i in range(stripes):
        color = (245, 240, 232) if i % 2 else lerp(facade, (255, 255, 255), .15)
        px.polygon([(110 + i * sw, 160), (110 + (i + 1) * sw, 160),
                    (100 + (i + 1) * sw * 1.05, 196), (100 + i * sw * 1.05, 196)], fill=color)
    glow = lerp((255, 196, 120), (255, 226, 170), (detail >> 4) % 7 / 6)
    px.rectangle([140, 205, 370, 315], fill=glow)
    for x in (217, 294):
        px.line([(x, 205), (x, 315)], fill=lerp(facade, (0, 0, 0), .5), width=4)
    px.rectangle([400, 205, 490, 330], fill=lerp(facade, (0, 0, 0), .55))
    px.ellipse([470, 262, 478, 270], fill=(230, 200, 120))
    for i in range((detail >> 8) % 4 + 1):
        cx = 160 + i * 52
        px.rounded_rectangle([cx, 280, cx + 22, 300], 4, fill=(250, 250, 248))
        px.arc([cx + 16, 284, cx + 30, 296], -90, 90, fill=(250, 250, 248), width=3)
    px.rectangle([0, 330, w, h], fill=(36, 38, 46))
    for y in range(336, h, 9):
        px.line([(0, y), (w, y)], fill=lerp((36, 38, 46), facade, .18 * (h - y) / 70))
    rain = np.random.default_rng(detail)
    for _ in range(140):
        x, y = rain.uniform(0, w), rain.uniform(0, h)
        px.line([(x, y), (x - 3, y + 12)], fill=(200, 210, 230), width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    px = ImageDraw.Draw(img)
    px.rectangle([0, 0, w, 30], fill=(250, 204, 21))
    label = font(15)
    px.text((12, 6), "SYNTHETIC FIXTURE · placeholder image, not model output", font=label,
            fill=(28, 25, 23))
    small = font(13, bold=False)
    tag = (f"answer {index + 1} (leaf id {leaf['id']}) · "
           f"λ∈({leaf['lo']:.3f}, {leaf['hi']:.3f}] · mix {mix:.2f}")
    px.rectangle([w - 12 - px.textlength(tag, font=small) - 12, h - 32, w - 10, h - 10],
                 fill=(15, 15, 18))
    px.text((w - 16 - px.textlength(tag, font=small), h - 29), tag, font=small,
            fill=(235, 235, 235))
    img.save(path, "JPEG", quality=84, optimize=True)


# ---------------------------------------------------------------------- simplex atlas
def clip(poly, f):
    """Sutherland-Hodgman clip of a barycentric polygon to the half-space f(p) >= 0."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        fp, fq = f(p), f(q)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append([a + t * (b - a) for a, b in zip(p, q)])
    return out


def area(poly):
    """Area in the (lambda_0, lambda_1) chart; the whole simplex has area 1/2."""
    return abs(sum(p[0] * q[1] - q[0] * p[1]
                   for p, q in zip(poly, poly[1:] + poly[:1]))) / 2 if len(poly) >= 3 else 0.0


def race_region(i, E):
    """Winner region {lambda : lambda_i/E_i >= lambda_j/E_j for all j}."""
    def regions(poly):
        for j in range(len(E)):
            if j != i:
                poly = clip(poly, lambda p, j=j: p[i] * E[j] - p[j] * E[i])
                if not poly:
                    break
        return poly
    return regions


def race_step(ps, uniforms, E):
    """MOONSHOTS.md section 3, N bidders: common mass, equal residual mass, Exp race."""
    common = ps.min(axis=0)
    shared_mass = float(min(1.0, common.sum()))
    if uniforms[0] < shared_mass or shared_mass >= 1 - 1e-15:
        cdf = np.cumsum(common / common.sum())
        cdf[-1] = 1
        return shared_mass, int(np.searchsorted(cdf, uniforms[1], side="right")), None
    cands = []
    for i, p in enumerate(ps):
        r = np.clip(p - common, 0, None)
        cdf = np.cumsum(r / r.sum())
        cdf[-1] = 1
        cands.append(int(np.searchsorted(cdf, uniforms[1 + i], side="right")))
    return shared_mass, None, cands


def build_simplex_atlas(seed: int, max_steps: int = 96, min_area: float = 1e-12):
    brands = ["Bayview Hotel", "Lantern House", "Harbor Loft"]
    model = hotel_model(brands, "bayviewhotel.com")
    n = len(brands)
    rng = np.random.default_rng(seed)
    root = {"id": 0, "polygon": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "parent": None, "split_step": None, "tokens": [], "steps": [], "stop": None}
    cells, splits = [root], []
    for k in range(max_steps):
        uniforms, E = rng.random(1 + n), rng.exponential(size=n)
        E = [float(max(e, np.nextafter(0, 1))) for e in E]
        live = [c for c in cells if c["stop"] is None]
        if not live:
            break
        for cell in live:
            mass, token, cands = race_step(model.distributions(cell["tokens"]), uniforms, E)
            base = {"step": k, "shared_mass": mass}
            if cands is None:
                take(cell, token, {**base, "branch": "shared", "winner": None, "E": None,
                                   "token_id": token})
                continue
            pieces = [(i, race_region(i, E)(cell["polygon"])) for i in range(n)]
            pieces = [(i, p) for i, p in pieces if area(p) > min_area]
            pieces.sort(key=lambda ip: -area(ip[1]))
            prefix = list(cell["tokens"])
            members = []
            for rank, (i, poly) in enumerate(pieces):
                target = cell if rank == 0 else deepcopy(cell)
                if rank:
                    target.update(id=len(cells), parent=cell["id"], split_step=k)
                    target["tokens"], target["steps"] = list(prefix), deepcopy(cell["steps"][:k])
                target["polygon"] = poly
                take(target, cands[i], {**base, "branch": "contested", "winner": i, "E": E,
                                        "token_id": cands[i]})
                members.append({"cell": target["id"], "winner": i, "token": cands[i],
                                "token_text": model.pieces[cands[i]]})
                if rank:
                    cells.append(target)
            if len(pieces) > 1:
                total = sum(E)
                splits.append({"step": k, "E": E, "apex": [e / total for e in E],
                               "parent": cell["id"], "pieces": members,
                               "prefix_text": model.decode(prefix)})
    for c in cells:
        c["stop"] = c["stop"] or "length"
    keys = [f"mentions:{b}" for b in brands]
    leaves = []
    for c in cells:
        text = model.decode(c["tokens"])
        leaves.append({"id": c["id"], "polygon": c["polygon"], "parent": c["parent"],
                       "split_step": c["split_step"], "token_ids": c["tokens"], "text": text,
                       "image": None, "stop_reason": c["stop"], "steps": c["steps"],
                       "outcomes": text_outcomes(text, keys)})
    verification = []
    for leaf in leaves:
        centroid = np.mean(leaf["polygon"], axis=0)
        tokens = direct_race_run(model, seed, centroid, max_steps)
        verification.append({"leaf": leaf["id"], "lambda": [float(x) for x in centroid],
                             "matches_direct_run": tokens == leaf["token_ids"]})
    return {
        "schema": "simplex-atlas-v1", "kind": "text", "mode": "synthetic_fixture",
        "title": "Three hotels, one seed (toy corpus)",
        "bidders": brands, "seed": seed,
        "parameter": "lambda_i = bid_i / sum(bids), barycentric",
        "max_steps": max_steps,
        "manifest": manifest(seed, {
            "model": "toy weighted-corpus model (scripts/make_atlas_fixtures.py), not an LLM",
            "revision": corpus_hash(model), "tokenizer": "toy word tokenizer; fake token IDs",
            "prompts": {"user": "I'm visiting Fort Mason in San Fransisco for a conference. "
                                "Where should I stay? How should I get there? Uber?"},
            "sampler": "exponential race, winner argmin E_i/b_i (MOONSHOTS.md section 3)",
            "min_cell_area": min_area, "eos_token_id": EOS,
        }),
        "leaves": leaves, "splits": splits, "verification": verification,
    }


def direct_race_run(model, seed, bids, max_steps):
    rng, tokens = np.random.default_rng(seed), []
    for _ in range(max_steps):
        uniforms, E = rng.random(1 + len(bids)), rng.exponential(size=len(bids))
        _, token, cands = race_step(model.distributions(tokens), uniforms, E)
        if cands is not None:
            token = cands[int(np.argmin([e / b if b > 0 else math.inf for e, b in zip(E, bids)]))]
        tokens.append(token)
        if token == EOS:
            break
    return tokens


# ------------------------------------------------------------------------------ write
def corpus_hash(model):
    blob = json.dumps([model.brands, [[(s, w) for s, w, _ in slot] for slot in model.slots]])
    blob = blob.encode()
    return "toy-corpus-sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def pillow_version():
    try:
        import PIL
        return PIL.__version__
    except ImportError:
        return None


def manifest(seed, extra):
    return {"label": FIXTURE_NOTE, "generator": "scripts/make_atlas_fixtures.py",
            "seed": seed, "python": platform.python_version(), "numpy": np.__version__,
            "simulated_credits": True, **extra}


def write(name, atlas):
    (OUT / name).write_text(json.dumps(atlas, indent=1, ensure_ascii=False) + "\n")
    count = len(atlas["leaves"])
    ok = all(v["matches_direct_run"] for v in atlas["verification"])
    print(f"wrote {name}: {count} leaves, {len(atlas['splits'])} splits, verified={ok}")
    return {"id": name.removesuffix(".json"), "title": atlas["title"], "schema": atlas["schema"],
            "kind": atlas["kind"], "mode": atlas["mode"], "path": name,
            "bidders": atlas["bidders"], "leaves": count, "seed": atlas["seed"]}


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    text_seed, image_seed, simplex_seed = 110, 30, 91
    if len(argv) > 1 and argv[1] == "--search":
        for s in range(60):
            print(s, len(build_text_atlas(s)["leaves"]),
                  len(interval_search(image_step(24), s, 24)[0]),
                  len(build_simplex_atlas(s)["leaves"]))
        return
    entries = [
        write("fixture-text-hotels.json", build_text_atlas(text_seed)),
        write("fixture-image-cafe.json", build_image_atlas(image_seed)),
        write("fixture-simplex-hotels.json", build_simplex_atlas(simplex_seed)),
    ]
    index = {"schema": "atlas-index-v1", "note": "Paths are relative to this file. Every "
             "entry here is a SYNTHETIC FIXTURE unless its mode says otherwise.",
             "atlases": entries}
    (OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
