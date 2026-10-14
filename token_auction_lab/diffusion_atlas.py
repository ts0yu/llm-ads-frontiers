"""Exact bid atlases for an Euler-ancestral diffusion sampler with two advertiser prompts.

Each Euler-ancestral step draws ``x' ~ N(mean_c(x), sigma_up^2 I)``. The covariance
comes from the noise schedule and does not depend on the conditioning, so with one
guided epsilon per advertiser the step is a pair of equal-covariance Gaussians and
``GaussianTwoBidderAuction`` gives the exact stable-sampling step (MOONSHOTS.md
section 3). Bids never enter the prompts. With bids (1, 0) the loop reproduces
``EulerAncestralDiscreteScheduler.step`` given the same noise tensor (tested).

Per step i (sigma_i -> sigma_{i+1}), mirroring the scheduler's float32 ops:

    sigma_up   = sqrt(sigma_{i+1}^2 (sigma_i^2 - sigma_{i+1}^2) / sigma_i^2)
    sigma_down = sqrt(sigma_{i+1}^2 - sigma_up^2)
    eps_c      = eps_uncond + g * (eps_c_raw - eps_uncond)  (uncond, A, B: one UNet call)
    mean_c     = x + eps_c * (sigma_down - sigma_i)          (epsilon prediction)

The final step has sigma_up = 0: point masses (contested unless the means agree).

Random-number order for a seed: ``rng = numpy.random.default_rng(seed)``; first
``rng.standard_normal(latent_numel)`` is the initial latent (times
``scheduler.init_noise_sigma``); then for each step k in order,
``GaussianTape.draw(rng, latent_numel)`` (four uniforms, then a standard normal
vector). Tapes do not depend on bids, so the generation is a piecewise-constant
function of lambda = a/(a+b), and the atlas below enumerates it exactly.

Atlas: interval search over lambda (MOONSHOTS.md section 1). The only forked state
is the latent tensor. At most 1 + T leaves. Labels: ``live_model`` for real model
inference, ``synthetic_model_test`` for tiny random models. CLIP brand scores are
a noisy zero-shot heuristic, not ground truth. Simulated credits only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from token_auction_lab.gaussian_auction import GaussianTape, GaussianTwoBidderAuction

SCHEMA = "bid-atlas-v1"
SCENES_PATH = Path(__file__).with_name("image_scenes.json")
RNG_ORDER = ("numpy.random.default_rng(seed): standard_normal(latent_numel) initial latent "
             "(x init_noise_sigma), then per step GaussianTape.draw(rng, latent_numel) = "
             "RandomTape.draw (4 uniforms: branch, token_a, token_b, contest) followed by "
             "standard_normal(latent_numel)")


# ---------------------------------------------------------------------------
# Core sampler (accepts precomputed embeddings; no tokenizer needed).
# ---------------------------------------------------------------------------

@dataclass
class Conditioning:
    """Precomputed conditioning, batched in the order [uncond, A, B]."""

    prompt_embeds: torch.Tensor
    added_cond_kwargs: dict | None = None

    def __post_init__(self):
        if self.prompt_embeds.shape[0] != 3:
            raise ValueError("prompt_embeds must be batched as [uncond, A, B].")
        for key, value in (self.added_cond_kwargs or {}).items():
            if value.shape[0] != 3:
                raise ValueError(f"added_cond_kwargs[{key!r}] must have batch 3.")


@dataclass
class StepContext:
    step: int
    mean_a: torch.Tensor  # float32, as the scheduler computes it
    mean_b: torch.Tensor
    sigma_up: torch.Tensor  # 0-dim float32 (CPU), the scheduler's own value
    auction: GaussianTwoBidderAuction


@dataclass
class StepRecord:
    step: int
    branch: str
    u: float
    shared_mass: float
    disagreement: float
    delta: float | None  # None = infinite (point masses at the final step)
    sigma_up: float

    def to_json(self) -> dict:
        return {"step": self.step, "branch": self.branch, "u": self.u,
                "shared_mass": self.shared_mass, "disagreement": self.disagreement,
                "delta": self.delta, "sigma_up": self.sigma_up}


@dataclass
class DirectResult:
    latent: torch.Tensor
    steps: list[StepRecord]
    payment_a: float
    payment_b: float
    latents: list[torch.Tensor] = field(default_factory=list)  # after each step
    noises: list[torch.Tensor] = field(default_factory=list)  # noise tensor used per step


@dataclass
class Leaf:
    id: int
    lo: float
    hi: float
    parent: int | None
    split_step: int | None
    latent: torch.Tensor
    start: int
    steps: list[StepRecord] = field(default_factory=list)
    done: bool = False


@dataclass
class AtlasResult:
    leaves: list[Leaf]  # sorted by lo
    splits: list[dict]
    complete: bool
    pending: list[dict]
    unet_calls: int
    seconds: float


def _record(ctx: StepContext, branch: str, u: float) -> StepRecord:
    a = ctx.auction
    return StepRecord(ctx.step, branch, float(u), a.shared_mass, a.disagreement,
                      a.delta if math.isfinite(a.delta) else None, float(ctx.sigma_up))


class EulerAncestralBidSampler:
    """Two-advertiser Euler-ancestral sampling with an exact per-step Gaussian auction."""

    def __init__(self, unet, scheduler, conditioning: Conditioning, *, guidance: float,
                 steps: int, latent_shape: tuple[int, int, int], device=None, dtype=None):
        from diffusers import EulerAncestralDiscreteScheduler

        if not isinstance(scheduler, EulerAncestralDiscreteScheduler):
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        if scheduler.config.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"Unsupported prediction_type {scheduler.config.prediction_type}.")
        if getattr(unet.config, "time_cond_proj_dim", None) is not None:
            raise ValueError("Guidance-embedding UNets (time_cond_proj_dim) are not supported.")
        if not math.isfinite(guidance) or steps < 1:
            raise ValueError("guidance must be finite and steps >= 1.")
        self.unet, self.scheduler = unet, scheduler
        self.device = torch.device(device) if device is not None else unet.device
        self.dtype = dtype or unet.dtype
        self.guidance = float(guidance)
        self.steps = int(steps)
        self.latent_shape = (1, *latent_shape)
        self.numel = int(np.prod(latent_shape))
        scheduler.set_timesteps(self.steps, device=self.device)
        self.timesteps = scheduler.timesteps
        self.sigmas = scheduler.sigmas  # float32 on CPU, final entry 0
        self.prediction_type = scheduler.config.prediction_type
        self.cond = Conditioning(
            conditioning.prompt_embeds.to(self.device, self.dtype),
            {k: v.to(self.device, self.dtype)
             for k, v in (conditioning.added_cond_kwargs or {}).items()} or None,
        )
        self.unet_calls = 0

    # -- randomness -------------------------------------------------------------
    def tapes(self, seed: int) -> tuple[torch.Tensor, list[GaussianTape]]:
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(self.numel)
        tapes = [GaussianTape.draw(rng, self.numel) for _ in range(self.steps)]
        latent = torch.from_numpy(z0.reshape(self.latent_shape)).to(self.device, self.dtype)
        return latent * self.scheduler.init_noise_sigma, tapes

    # -- one step ---------------------------------------------------------------
    @torch.inference_mode()
    def guided_eps(self, latent: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigmas[k]
        model_input = torch.cat([latent] * 3)
        model_input = model_input / ((sigma**2 + 1) ** 0.5)  # scale_model_input
        out = self.unet(model_input, self.timesteps[k],
                        encoder_hidden_states=self.cond.prompt_embeds,
                        added_cond_kwargs=self.cond.added_cond_kwargs, return_dict=False)[0]
        self.unet_calls += 1
        uncond, raw_a, raw_b = out.chunk(3)
        g = self.guidance
        return uncond + g * (raw_a - uncond), uncond + g * (raw_b - uncond)

    def _mean(self, sample32: torch.Tensor, model_output: torch.Tensor, k: int):
        """The scheduler's ``prev_sample`` before noise, with identical float32 ops."""
        sigma = self.sigmas[k]
        if self.prediction_type == "epsilon":
            pred_original = sample32 - sigma * model_output
        else:
            pred_original = (model_output * (-sigma / (sigma**2 + 1) ** 0.5)
                             + (sample32 / (sigma**2 + 1)))
        sigma_from, sigma_to = self.sigmas[k], self.sigmas[k + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        derivative = (sample32 - pred_original) / sigma
        dt = sigma_down - sigma
        return sample32 + derivative * dt, sigma_up

    @torch.inference_mode()
    def step_context(self, latent: torch.Tensor, k: int) -> StepContext:
        eps_a, eps_b = self.guided_eps(latent, k)
        sample32 = latent.to(torch.float32)
        mean_a, sigma_up = self._mean(sample32, eps_a, k)
        mean_b, _ = self._mean(sample32, eps_b, k)
        auction = GaussianTwoBidderAuction(
            mean_a.detach().double().cpu().numpy().reshape(-1),
            mean_b.detach().double().cpu().numpy().reshape(-1),
            float(sigma_up),
        )
        return StepContext(k, mean_a, mean_b, sigma_up, auction)

    def noise_tensor(self, offset: np.ndarray) -> torch.Tensor:
        # The scheduler draws its noise in model_output's dtype; so do we.
        return torch.from_numpy(np.asarray(offset).reshape(self.latent_shape)).to(
            self.device, self.dtype)

    def emit(self, ctx: StepContext, frame: str, offset: np.ndarray,
             noises: list | None = None) -> torch.Tensor:
        mean = ctx.mean_a if frame == "A" else ctx.mean_b
        noise = self.noise_tensor(offset)
        if noises is not None:
            noises.append(noise)
        return (mean + noise * ctx.sigma_up).to(self.dtype)

    # -- direct (non-tree) run --------------------------------------------------
    def run_direct(self, seed: int, bid_a: float, bid_b: float, *,
                   keep_trajectory: bool = False) -> DirectResult:
        latent, tapes = self.tapes(seed)
        records, latents, noises = [], [], []
        pay_a = pay_b = 0.0
        for k in range(self.steps):
            ctx = self.step_context(latent, k)
            alloc = ctx.auction.resolve(bid_a, bid_b, tapes[k])
            latent = self.emit(ctx, alloc.frame, alloc.offset, noises if keep_trajectory else None)
            pay_a += alloc.payment_a
            pay_b += alloc.payment_b
            records.append(_record(ctx, alloc.branch, tapes[k].contest))
            if keep_trajectory:
                latents.append(latent)
        return DirectResult(latent, records, pay_a, pay_b, latents, noises)

    # -- exact atlas ------------------------------------------------------------
    def atlas(self, seed: int, *, on_leaf: Callable[[Leaf], None] | None = None,
              deadline: float | None = None) -> AtlasResult:
        """Interval search over lambda in (0, 1]. Depth-first; leaves finish one by one."""
        start_time = time.perf_counter()
        calls0 = self.unet_calls
        latent0, tapes = self.tapes(seed)
        stack = [Leaf(0, 0.0, 1.0, None, None, latent0, 0)]
        leaves: list[Leaf] = []
        splits: list[dict] = []
        next_id = 1
        complete = True
        while stack:
            if deadline is not None and time.perf_counter() > deadline:
                complete = False
                break
            leaf = stack.pop()
            for k in range(leaf.start, self.steps):
                ctx = self.step_context(leaf.latent, k)
                tape = tapes[k]
                draw = ctx.auction.draw(tape)
                u = tape.contest
                if draw.shared:
                    leaf.latent = self.emit(ctx, "A", draw.offset_shared)
                    leaf.steps.append(_record(ctx, "shared", u))
                elif u <= leaf.lo:  # lambda > lo >= u: A for the whole interval
                    leaf.latent = self.emit(ctx, "A", draw.offset_a)
                    leaf.steps.append(_record(ctx, "A", u))
                elif u >= leaf.hi:  # lambda <= hi <= u: B for the whole interval
                    leaf.latent = self.emit(ctx, "B", draw.offset_b)
                    leaf.steps.append(_record(ctx, "B", u))
                else:  # split: parent keeps (lo, u] (B), child takes (u, hi] (A)
                    child = Leaf(next_id, u, leaf.hi, leaf.id, k,
                                 self.emit(ctx, "A", draw.offset_a), k + 1,
                                 leaf.steps + [_record(ctx, "A", u)])
                    splits.append({"step": k, "u": u, "parent": leaf.id, "child": next_id,
                                   "delta": ctx.auction.delta
                                   if math.isfinite(ctx.auction.delta) else None,
                                   "disagreement": ctx.auction.disagreement})
                    next_id += 1
                    stack.append(child)
                    leaf.hi = u
                    leaf.latent = self.emit(ctx, "B", draw.offset_b)
                    leaf.steps.append(_record(ctx, "B", u))
            leaf.done = True
            leaves.append(leaf)
            if on_leaf is not None:
                on_leaf(leaf)
        pending = [{"id": p.id, "lo": p.lo, "hi": p.hi, "start_step": p.start} for p in stack]
        if complete and len(leaves) > self.steps + 1:
            raise AssertionError("Atlas exceeded the 1 + T leaf bound.")
        return AtlasResult(sorted(leaves, key=lambda x: x.lo), splits, complete, pending,
                           self.unet_calls - calls0, time.perf_counter() - start_time)


def leaf_charges(steps: list[StepRecord] | list[dict], bid_a: float, bid_b: float):
    """Exact simulated charges for a bid pair whose lambda lies inside the leaf.

    Same formulas and summation order as ``GaussianTwoBidderAuction.resolve``:
    A-won steps charge A ``b*u/(1-u)``, B-won steps charge B ``a*(1-u)/u``.
    """
    pay_a = pay_b = 0.0
    for rec in steps:
        branch = rec.branch if isinstance(rec, StepRecord) else rec["branch"]
        u = rec.u if isinstance(rec, StepRecord) else rec["u"]
        if branch == "A":
            pay_a += bid_b * (u / (1.0 - u))
        elif branch == "B":
            pay_b += bid_a * ((1.0 - u) / u)
    return pay_a, pay_b


def latent_sha256(latent: torch.Tensor) -> str:
    return hashlib.sha256(latent.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_leaves(sampler: EulerAncestralBidSampler, seed: int, result: AtlasResult,
                  count: int, deadline: float | None = None) -> list[dict]:
    """Re-run direct (non-tree) sampling at leaf midpoints; compare final latents."""
    leaves = result.leaves
    if count <= 0 or not leaves:
        return []
    order = [leaves[0], leaves[-1]] if len(leaves) > 1 else [leaves[0]]
    order += sorted(leaves[1:-1], key=lambda x: (-(x.hi - x.lo), x.lo))
    report = []
    for leaf in order[:count]:
        if deadline is not None and time.perf_counter() > deadline:
            break
        lam = 0.5 * (leaf.lo + leaf.hi)
        direct = sampler.run_direct(seed, lam, 1.0 - lam)
        diff = (direct.latent.float() - leaf.latent.float()).abs().max().item()
        branches_match = [r.branch for r in direct.steps] == [r.branch for r in leaf.steps]
        pay_a, pay_b = leaf_charges(leaf.steps, lam, 1.0 - lam)
        report.append({
            "leaf": leaf.id, "lambda": lam, "bids": [lam, 1.0 - lam],
            "matches_direct_run": bool(torch.equal(direct.latent, leaf.latent)),
            "max_abs_latent_diff": diff, "branches_match": branches_match,
            "payments_direct": [direct.payment_a, direct.payment_b],
            "payments_from_atlas": [pay_a, pay_b],
        })
    return report


# ---------------------------------------------------------------------------
# Atlas document.
# ---------------------------------------------------------------------------

def atlas_document(result: AtlasResult, *, seed: int, bidders: list[str], mode: str,
                   manifest: dict, steps: int, images: dict[int, str] | None = None,
                   outcomes: dict[int, dict] | None = None,
                   verification: list[dict] | None = None, title: str | None = None) -> dict:
    images, outcomes = images or {}, outcomes or {}
    return {
        "schema": SCHEMA,
        "kind": "image",
        "mode": mode,
        "title": title,
        "bidders": list(bidders),
        "seed": seed,
        "parameter": "lambda = bid_A / (bid_A + bid_B)",
        "max_steps": steps,
        "complete": result.complete,
        "pending": result.pending,
        "manifest": manifest,
        "stats": {"leaves": len(result.leaves), "splits": len(result.splits),
                  "unet_calls": result.unet_calls, "seconds": round(result.seconds, 3),
                  "bound_1_plus_T": steps + 1},
        "leaves": [
            {
                "id": leaf.id, "lo": leaf.lo, "hi": leaf.hi, "parent": leaf.parent,
                "split_step": leaf.split_step, "image": images.get(leaf.id),
                "stop_reason": "steps", "latent_sha256": latent_sha256(leaf.latent),
                "contested_steps": sum(r.branch != "shared" for r in leaf.steps),
                "steps": [r.to_json() for r in leaf.steps],
                "outcomes": outcomes.get(leaf.id, {}),
            }
            for leaf in result.leaves
        ],
        "splits": result.splits,
        "verification": verification or [],
    }


# ---------------------------------------------------------------------------
# Pipeline helpers (live mode).
# ---------------------------------------------------------------------------

def load_scenes(path: Path = SCENES_PATH) -> dict:
    data = json.loads(Path(path).read_text())
    for name, scene in data["scenes"].items():
        if len(scene["advertisers"]) != 2:
            raise ValueError(f"Scene {name!r} must have exactly two advertisers.")
    return data


def scene_prompts(scenes: dict, scene: str) -> dict:
    spec = scenes["scenes"][scene]
    shared = spec["shared"]
    ads = spec["advertisers"]
    return {
        "scene": scene,
        "title": spec.get("title", scene),
        "shared": shared,
        "bidders": [ad["name"] for ad in ads],
        "prompt_a": f"{shared}, {ads[0]['brief']}",
        "prompt_b": f"{shared}, {ads[1]['brief']}",
        "negative": spec.get("negative_prompt", scenes.get("negative_prompt", "")),
        "clip_labels": spec.get("clip_labels", {}),
    }


def pipeline_kind(pipe) -> str:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    if isinstance(pipe, StableDiffusionXLPipeline):
        return "sdxl"
    if isinstance(pipe, StableDiffusionPipeline):
        return "sd"
    raise ValueError(f"Unsupported pipeline {type(pipe).__name__}; need SD 1.x/2.x or SDXL.")


@torch.inference_mode()
def encode_conditioning(pipe, kind: str, prompts: dict, height: int, width: int,
                        device) -> Conditioning:
    """Encode [negative, A, B] with the pipeline's own text encoders."""
    neg = prompts["negative"]
    if kind == "sdxl":
        pe_a, neg_a, pool_a, neg_pool_a = pipe.encode_prompt(
            prompt=prompts["prompt_a"], device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=neg)
        pe_b, neg_b, pool_b, neg_pool_b = pipe.encode_prompt(
            prompt=prompts["prompt_b"], device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=neg)
        if not (torch.equal(neg_a, neg_b) and torch.equal(neg_pool_a, neg_pool_b)):
            raise RuntimeError("Negative-prompt encoding is not deterministic.")
        time_ids = torch.tensor([[height, width, 0, 0, height, width]] * 3, dtype=pe_a.dtype)
        expected = pipe.unet.add_embedding.linear_1.in_features
        passed = pipe.unet.config.addition_time_embed_dim * 6 + pool_a.shape[-1]
        if expected != passed:
            raise ValueError(f"UNet expects added embedding {expected}, got {passed}.")
        cond = Conditioning(torch.cat([neg_a, pe_a, pe_b]),
                            {"text_embeds": torch.cat([neg_pool_a, pool_a, pool_b]),
                             "time_ids": time_ids.to(device)})
    else:
        pe_a, neg_a = pipe.encode_prompt(prompts["prompt_a"], device, 1, True, neg)
        pe_b, neg_b = pipe.encode_prompt(prompts["prompt_b"], device, 1, True, neg)
        if not torch.equal(neg_a, neg_b):
            raise RuntimeError("Negative-prompt encoding is not deterministic.")
        cond = Conditioning(torch.cat([neg_a, pe_a, pe_b]))
    return cond


def prompt_token_lengths(pipe, prompts: dict) -> dict:
    lengths = {}
    for name in ("tokenizer", "tokenizer_2"):
        tok = getattr(pipe, name, None)
        if tok is None:
            continue
        for key in ("prompt_a", "prompt_b", "negative"):
            n = len(tok(prompts[key]).input_ids)
            lengths[f"{name}:{key}"] = {"tokens": n, "truncated": n > tok.model_max_length}
    return lengths


class LatentDecoder:
    """VAE decode -> PIL, upcasting an fp16 VAE when its config asks (as the pipelines do)."""

    def __init__(self, vae, safety=None):
        self.vae = vae
        self.safety = safety  # optional (pipe.run_safety_checker, device, dtype)
        if vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False):
            vae.to(dtype=torch.float32)

    @torch.inference_mode()
    def __call__(self, latent: torch.Tensor):
        from PIL import Image

        cfg = self.vae.config
        param = next(iter(self.vae.post_quant_conv.parameters()))
        latent = latent.to(param.device, param.dtype)
        mean, std = getattr(cfg, "latents_mean", None), getattr(cfg, "latents_std", None)
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(1, -1, 1, 1).to(latent.device, latent.dtype)
            std_t = torch.tensor(std).view(1, -1, 1, 1).to(latent.device, latent.dtype)
            latent = latent * std_t / cfg.scaling_factor + mean_t
        else:
            latent = latent / cfg.scaling_factor
        image = self.vae.decode(latent, return_dict=False)[0]
        nsfw = None
        if self.safety is not None:
            checker, device, dtype = self.safety
            image, flags = checker(image, device, dtype)
            nsfw = bool(flags[0]) if flags is not None else None
        array = ((image[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
        return Image.fromarray(array.round().astype(np.uint8)), nsfw


class ClipScorer:
    """Zero-shot CLIP label probabilities: a NOISY HEURISTIC, not ground truth."""

    def __init__(self, model_id: str, device, revision: str | None = None,
                 local_files_only: bool = False):
        from transformers import CLIPModel, CLIPProcessor

        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id, revision=revision,
                                               local_files_only=local_files_only).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision,
                                                       local_files_only=local_files_only)
        self.device = device
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image, labels: dict[str, str]) -> dict:
        names = list(labels)
        inputs = self.processor(text=[labels[n] for n in names], images=image,
                                return_tensors="pt", padding=True).to(self.device)
        probs = self.model(**inputs).logits_per_image.softmax(-1)[0].tolist()
        result = {f"clip:{n}": float(p) for n, p in zip(names, probs)}
        top = names[int(np.argmax(probs))]
        result["clip_argmax"] = top
        # Boolean per label, so the explorer can draw a band (still a noisy heuristic).
        result.update({f"clip_top:{n}": n == top for n in names})
        return result


def resolved_revision(pipe, requested: str | None) -> str | None:
    """Best effort: the snapshot hash in the local cache path, else the request."""
    path = str(getattr(pipe.unet.config, "_name_or_path", "") or "")
    parts = Path(path).parts
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        return parts[parts.index("snapshots") + 1]
    return requested


def environment_manifest() -> dict:
    import importlib.metadata as md

    versions = {}
    for pkg in ("torch", "diffusers", "transformers", "huggingface_hub", "numpy", "scipy",
                "pillow", "accelerate", "safetensors"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def write_atlas(sampler: EulerAncestralBidSampler, *, seed: int, prompts: dict, out_dir: Path,
                stem: str, mode: str, manifest: dict, decoder=None, clip=None,
                verify: int = 0, jpeg_quality: int = 88, jpeg_size: int | None = None,
                deadline: float | None = None, log=print) -> dict:
    """Run one atlas, writing each leaf's JPEG as it finishes, then the atlas JSON."""
    image_dir = out_dir / stem
    images, outcomes, safety = {}, {}, {}

    def on_leaf(leaf: Leaf):
        if decoder is None:
            return
        image, nsfw = decoder(leaf.latent)
        if jpeg_size and max(image.size) != jpeg_size:
            scale = jpeg_size / max(image.size)
            image = image.resize((max(1, round(image.width * scale)),
                                  max(1, round(image.height * scale))))
        image_dir.mkdir(parents=True, exist_ok=True)
        rel = f"{stem}/leaf_{leaf.id:02d}.jpg"
        image.save(out_dir / rel, quality=jpeg_quality)
        images[leaf.id] = rel
        if nsfw is not None:
            safety[leaf.id] = nsfw
        if clip is not None and prompts.get("clip_labels"):
            outcomes[leaf.id] = clip(image, prompts["clip_labels"])
        log(f"  leaf {leaf.id} ({leaf.lo:.4f}, {leaf.hi:.4f}] -> {rel}")

    result = sampler.atlas(seed, on_leaf=on_leaf, deadline=deadline)
    if safety:
        for leaf_id, flag in safety.items():
            outcomes.setdefault(leaf_id, {})["safety_checker_nsfw"] = flag
    verification = verify_leaves(sampler, seed, result, verify, deadline) if result.complete \
        else []
    doc = atlas_document(result, seed=seed, bidders=prompts["bidders"], mode=mode,
                         manifest=manifest, steps=sampler.steps, images=images,
                         outcomes=outcomes, verification=verification,
                         title=prompts.get("title"))
    if clip is not None:
        doc["outcome_note"] = ("clip:* values are CLIP zero-shot label probabilities: a noisy "
                               "heuristic for brand visibility, not ground truth.")
    (out_dir / f"{stem}.atlas.json").write_text(json.dumps(doc, indent=1, default=str))
    return doc


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--revision", default=None)
    p.add_argument("--scene", default="cafe")
    p.add_argument("--scenes-file", type=Path, default=SCENES_PATH)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--size", type=int, default=1024, help="square image size in pixels")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", type=int, default=2, help="leaves to re-run directly")
    p.add_argument("--time-budget-seconds", type=float, default=None)
    p.add_argument("--clip", default=None, help="optional CLIP model id for outcome scores")
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--jpeg-size", type=int, default=768, help="max JPEG side; 0 = native")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-truncation", action="store_true",
                   help="run even if a prompt exceeds the text encoder's token window")
    return p


def main(argv: list[str] | None = None, *, mode: str = "live_model") -> int:
    """``mode`` stays live_model for real pipelines; tiny random pipelines pass
    synthetic_model_test (scripts/make_model_test_atlases.py)."""
    args = build_parser().parse_args(argv)
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    t0 = time.perf_counter()
    deadline = t0 + args.time_budget_seconds if args.time_budget_seconds else None
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else (
        torch.float16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    scenes = load_scenes(args.scenes_file)
    prompts = scene_prompts(scenes, args.scene)
    load = {"torch_dtype": dtype, "revision": args.revision,
            "local_files_only": args.local_files_only}
    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.model, variant="fp16" if dtype == torch.float16 else None, **load)
    except (OSError, ValueError):
        pipe = DiffusionPipeline.from_pretrained(args.model, **load)
    kind = pipeline_kind(pipe)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.size % pipe.vae_scale_factor:
        raise SystemExit(f"--size must be a multiple of {pipe.vae_scale_factor}.")
    latent_hw = args.size // pipe.vae_scale_factor
    token_lengths = prompt_token_lengths(pipe, prompts)
    truncated = sorted(k for k, v in token_lengths.items() if v["truncated"])
    if truncated and not args.allow_truncation:
        raise SystemExit(f"Prompts exceed the text-encoder window ({', '.join(truncated)}); a "
                         "truncated brief can make the advertisers' prompts identical. Shorten "
                         "the scene or pass --allow-truncation.")
    cond = encode_conditioning(pipe, kind, prompts, args.size, args.size, device)
    sampler = EulerAncestralBidSampler(
        pipe.unet, pipe.scheduler, cond, guidance=args.guidance, steps=args.steps,
        latent_shape=(pipe.unet.config.in_channels, latent_hw, latent_hw), device=device,
        dtype=dtype)
    safety = None
    if kind == "sd" and getattr(pipe, "safety_checker", None) is not None:
        safety = (pipe.run_safety_checker, device, dtype)
    decoder = LatentDecoder(pipe.vae, safety)
    clip = ClipScorer(args.clip, device, local_files_only=args.local_files_only) \
        if args.clip else None
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model, "revision": args.revision,
        "resolved_revision": resolved_revision(pipe, args.revision),
        "pipeline": type(pipe).__name__, "pipeline_kind": kind,
        "tokenizers": [type(getattr(pipe, n)).__name__ for n in ("tokenizer", "tokenizer_2")
                       if getattr(pipe, n, None) is not None],
        "scheduler": "EulerAncestralDiscreteScheduler",
        "scheduler_config": dict(pipe.scheduler.config),
        "prompts": {k: prompts[k] for k in ("scene", "shared", "prompt_a", "prompt_b",
                                             "negative")},
        "prompt_token_lengths": token_lengths,
        "bids_in_prompts": False,
        "guidance": args.guidance, "steps": args.steps,
        "height": args.size, "width": args.size, "dtype": str(dtype).replace("torch.", ""),
        "device": str(device), "rng_order": RNG_ORDER,
        "clip_model": args.clip, "jpeg_quality": args.jpeg_quality,
        "jpeg_size": args.jpeg_size or None, "watermark": "not applied",
        "environment": environment_manifest(),
        "credits": "simulated",
    }
    index_path = args.output / "index.json"
    index = {"schema": "bid-atlas-index-v1", "mode": mode, "kind": "image",
             "created": datetime.now(timezone.utc).isoformat(), "atlases": [], "skipped": []}
    for seed in args.seeds:
        if deadline is not None and time.perf_counter() > deadline:
            index["skipped"].append({"seed": seed, "reason": "time budget"})
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stem = f"{args.scene}_seed{seed}"
        print(f"[{args.scene} seed {seed}] building atlas ...", flush=True)
        doc = write_atlas(sampler, seed=seed, prompts=prompts, out_dir=args.output, stem=stem,
                          mode=mode, manifest={**manifest, "seed": seed},
                          decoder=decoder, clip=clip, verify=args.verify,
                          jpeg_quality=args.jpeg_quality, jpeg_size=args.jpeg_size or None,
                          deadline=deadline)
        entry = {"scene": args.scene, "seed": seed, "atlas": f"{stem}.atlas.json",
                 "leaves": len(doc["leaves"]), "complete": doc["complete"],
                 "seconds": doc["stats"]["seconds"], "unet_calls": doc["stats"]["unet_calls"],
                 "verification": [{k: v[k] for k in ("leaf", "matches_direct_run",
                                                     "max_abs_latent_diff")}
                                  for v in doc["verification"]]}
        if device.type == "cuda":
            entry["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        index["atlases"].append(entry)
        index["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        index_path.write_text(json.dumps(index, indent=1, default=str))
        print(f"[{args.scene} seed {seed}] {entry}", flush=True)
    index["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    index_path.write_text(json.dumps(index, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
