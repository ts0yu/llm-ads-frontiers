"""Diffusion bid atlas on tiny RANDOM models (synthetic_model_test): not quality evidence."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, UNet2DConditionModel

from token_auction_lab import diffusion_atlas as da

STEPS = 8
LATENT = (4, 8, 8)
_THREADS = torch.get_num_threads()


def setUpModule():
    # Tiny convolutions are ~100x slower with intra-op thread contention on shared CPUs.
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_THREADS)


def tiny_unet(sdxl=False):
    torch.manual_seed(0)
    extra = {}
    if sdxl:
        extra = dict(addition_embed_type="text_time", addition_time_embed_dim=4,
                     projection_class_embeddings_input_dim=4 * 6 + 10)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=12,
        attention_head_dim=2, norm_num_groups=4, **extra)
    return unet.eval()


def tiny_scheduler(prediction_type="epsilon"):
    # SDXL-base scheduler settings, with the ancestral sampler.
    return EulerAncestralDiscreteScheduler(
        beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
        timestep_spacing="leading", steps_offset=1, prediction_type=prediction_type)


def conditioning(sdxl=False, b_scale=1.0):
    g = torch.Generator().manual_seed(1)
    uncond, a, noise = (torch.randn(1, 5, 12, generator=g) for _ in range(3))
    b = a + b_scale * noise  # b_scale -> 0 makes B's prompt nearly equal to A's
    added = None
    if sdxl:
        pooled = torch.randn(3, 10, generator=g)
        added = {"text_embeds": pooled,
                 "time_ids": torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0]] * 3)}
    return da.Conditioning(torch.cat([uncond, a, b]), added)


def sampler(sdxl=False, b_scale=1.0, guidance=5.0, prediction_type="epsilon", steps=STEPS):
    return da.EulerAncestralBidSampler(
        tiny_unet(sdxl), tiny_scheduler(prediction_type), conditioning(sdxl, b_scale),
        guidance=guidance, steps=steps, latent_shape=LATENT, device="cpu",
        dtype=torch.float32)


class SchedulerEquivalenceTests(unittest.TestCase):
    def _reference(self, s, seed, noises):
        """Plain diffusers loop for prompt A (guided), noise injected into step()."""
        sched = tiny_scheduler(s.prediction_type)
        sched.set_timesteps(s.steps, device="cpu")
        latent, _ = s.tapes(seed)
        queue = list(noises)
        out = []
        with patch("diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor",
                   side_effect=lambda *a, **k: queue.pop(0)), torch.inference_mode():
            for t in sched.timesteps:
                model_input = sched.scale_model_input(torch.cat([latent] * 3), t)
                eps = s.unet(model_input, t, encoder_hidden_states=s.cond.prompt_embeds,
                             added_cond_kwargs=s.cond.added_cond_kwargs, return_dict=False)[0]
                uncond, raw_a, _ = eps.chunk(3)
                guided = uncond + s.guidance * (raw_a - uncond)
                latent = sched.step(guided, t, latent).prev_sample
                out.append(latent)
        self.assertEqual(queue, [])
        return out

    def test_bids_one_zero_reproduce_scheduler_step_exactly(self):
        for kwargs in ({}, {"sdxl": True}, {"prediction_type": "v_prediction"}):
            s = sampler(**kwargs)
            direct = s.run_direct(3, 1.0, 0.0, keep_trajectory=True)
            self.assertTrue(all(r.branch in ("shared", "A") for r in direct.steps))
            reference = self._reference(s, 3, direct.noises)
            for k, (mine, ref) in enumerate(zip(direct.latents, reference)):
                self.assertTrue(torch.equal(mine, ref), msg=f"{kwargs} step {k}")

    def test_injected_noise_is_standard_normal_scale(self):
        s = sampler()
        direct = s.run_direct(3, 1.0, 0.0, keep_trajectory=True)
        noise = torch.cat([n.reshape(-1) for n in direct.noises[:-1]])
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(noise.std().item(), 1.0, delta=0.1)

    def test_sigma_schedule_and_final_point_mass(self):
        s = sampler()
        direct = s.run_direct(0, 1.0, 1.0)
        last = direct.steps[-1]
        self.assertEqual(last.sigma_up, 0.0)
        self.assertIsNone(last.delta)  # point masses with different means
        self.assertEqual((last.shared_mass, last.disagreement), (0.0, 1.0))
        for k, rec in enumerate(direct.steps[:-1]):
            hi, lo = float(s.sigmas[k]), float(s.sigmas[k + 1])
            self.assertAlmostEqual(rec.sigma_up, math.sqrt(lo**2 * (hi**2 - lo**2) / hi**2),
                                   places=5)


class AtlasTests(unittest.TestCase):
    def check_atlas(self, s, seed):
        result = s.atlas(seed)
        leaves = result.leaves
        self.assertTrue(result.complete)
        self.assertLessEqual(len(leaves), s.steps + 1)
        self.assertEqual(len(result.splits), len(leaves) - 1)
        self.assertEqual(leaves[0].lo, 0.0)
        self.assertEqual(leaves[-1].hi, 1.0)
        for left, right in zip(leaves, leaves[1:]):
            self.assertEqual(left.hi, right.lo)
            self.assertLess(left.lo, left.hi)
        for leaf in leaves:
            self.assertEqual(len(leaf.steps), s.steps)
            for lam in (0.5 * (leaf.lo + leaf.hi), leaf.hi):
                direct = s.run_direct(seed, lam, 1.0 - lam)
                self.assertTrue(torch.equal(direct.latent, leaf.latent), msg=f"leaf {leaf.id}")
                self.assertEqual([r.branch for r in direct.steps],
                                 [r.branch for r in leaf.steps])
                self.assertEqual((direct.payment_a, direct.payment_b),
                                 da.leaf_charges(leaf.steps, lam, 1.0 - lam))
            # Arbitrary scale S = a + b inside the leaf: MOONSHOTS.md formulas.
            lam = 0.5 * (leaf.lo + leaf.hi)
            total = 3.0
            pa, pb = da.leaf_charges(leaf.steps, total * lam, total * (1 - lam))
            ea = sum(total * (1 - lam) * r.u / (1 - r.u) for r in leaf.steps if r.branch == "A")
            eb = sum(total * lam * (1 - r.u) / r.u for r in leaf.steps if r.branch == "B")
            self.assertAlmostEqual(pa, ea, places=9)
            self.assertAlmostEqual(pb, eb, places=9)
        # Extremes: lambda = 1 is bids (1, 0); lambda -> 0+ is bids (0, 1).
        self.assertTrue(torch.equal(s.run_direct(seed, 1.0, 0.0).latent, leaves[-1].latent))
        self.assertTrue(torch.equal(s.run_direct(seed, 0.0, 1.0).latent, leaves[0].latent))
        return result

    def test_atlas_partitions_and_matches_direct_runs(self):
        s = sampler()
        result = self.check_atlas(s, seed=11)
        self.assertGreater(len(result.leaves), 1)  # random prompts disagree: splits happen
        # Same seed, same atlas (determinism).
        again = s.atlas(11)
        self.assertEqual([(x.lo, x.hi) for x in again.leaves],
                         [(x.lo, x.hi) for x in result.leaves])

    def test_atlas_with_sdxl_conditioning(self):
        self.check_atlas(sampler(sdxl=True), seed=5)

    def test_nearly_identical_prompts_share_steps(self):
        s = sampler(b_scale=1e-4, guidance=1.5)
        result = self.check_atlas(s, seed=2)
        branches = [r.branch for r in result.leaves[0].steps]
        self.assertIn("shared", branches)
        deltas = [r.delta for r in result.leaves[0].steps[:-1]]
        self.assertTrue(any(d < 1.0 for d in deltas))

    def test_identical_prompts_give_one_leaf(self):
        s = sampler(b_scale=0.0)
        result = s.atlas(4)
        self.assertEqual(len(result.leaves), 1)
        self.assertTrue(all(r.branch == "shared" for r in result.leaves[0].steps))
        self.assertEqual(da.leaf_charges(result.leaves[0].steps, 1, 1), (0.0, 0.0))

    def test_deadline_marks_partial_atlas(self):
        s = sampler()
        result = s.atlas(11, deadline=0.0)
        self.assertFalse(result.complete)
        self.assertEqual(result.leaves, [])
        self.assertEqual(result.pending[0]["lo"], 0.0)


class DocumentTests(unittest.TestCase):
    def test_write_atlas_document_and_images(self):
        torch.manual_seed(0)
        vae = AutoencoderKL(block_out_channels=(8,), in_channels=3, out_channels=3,
                            latent_channels=4, norm_num_groups=4,
                            down_block_types=("DownEncoderBlock2D",),
                            up_block_types=("UpDecoderBlock2D",)).eval()
        s = sampler()
        prompts = da.scene_prompts(da.load_scenes(), "cafe")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            doc = da.write_atlas(s, seed=11, prompts=prompts, out_dir=out, stem="cafe_seed11",
                                 mode="synthetic_model_test", manifest={"model": "tiny-random"},
                                 decoder=da.LatentDecoder(vae), verify=3, jpeg_size=16,
                                 log=lambda *_: None)
            saved = json.loads((out / "cafe_seed11.atlas.json").read_text())
            self.assertEqual(saved, json.loads(json.dumps(doc)))
            self.assertEqual((saved["schema"], saved["kind"], saved["mode"]),
                             ("bid-atlas-v1", "image", "synthetic_model_test"))
            self.assertEqual(saved["bidders"], prompts["bidders"])
            self.assertTrue(saved["complete"])
            self.assertLessEqual(len(saved["leaves"]), STEPS + 1)
            for leaf in saved["leaves"]:
                self.assertTrue((out / leaf["image"]).exists())
                self.assertEqual(len(leaf["steps"]), STEPS)
                self.assertEqual(set(leaf["steps"][0]), {"step", "branch", "u", "shared_mass",
                                                         "disagreement", "delta", "sigma_up"})
            self.assertEqual(len(saved["verification"]), min(3, len(saved["leaves"])))
            for v in saved["verification"]:
                self.assertTrue(v["matches_direct_run"])
                self.assertEqual(v["max_abs_latent_diff"], 0.0)
                self.assertEqual(v["payments_direct"], v["payments_from_atlas"])
            ids = {leaf["id"] for leaf in saved["leaves"]}
            for split in saved["splits"]:
                self.assertIn(split["child"], ids)

    def test_clip_scorer_formats_probabilities(self):
        class Out:
            logits_per_image = torch.tensor([[2.0, 0.0, -1.0]])

        class Inputs(dict):
            def to(self, device):
                return self

        scorer = object.__new__(da.ClipScorer)
        scorer.device = "cpu"
        scorer.processor = lambda **kw: Inputs()
        scorer.model = lambda **kw: Out()
        result = scorer(None, {"A": "a", "B": "b", "neutral": "n"})
        self.assertEqual(result["clip_argmax"], "A")
        self.assertAlmostEqual(sum(v for k, v in result.items() if k.startswith("clip:")), 1.0)


class SceneTests(unittest.TestCase):
    def test_scenes_are_two_advertiser_and_keep_names_out_of_prompts(self):
        scenes = da.load_scenes()
        self.assertGreaterEqual(len(scenes["scenes"]), 2)
        for name in scenes["scenes"]:
            p = da.scene_prompts(scenes, name)
            self.assertEqual(len(p["bidders"]), 2)
            self.assertTrue(p["prompt_a"].startswith(p["shared"]))
            self.assertTrue(p["prompt_b"].startswith(p["shared"]))
            self.assertNotEqual(p["prompt_a"], p["prompt_b"])
            for bidder in p["bidders"]:
                for key in ("prompt_a", "prompt_b", "negative"):
                    self.assertNotIn(bidder.lower(), p[key].lower())
                self.assertIn(bidder, p["clip_labels"])
            # Stay well inside CLIP's 77-token window (rough: ~1.3 tokens per word).
            for key in ("prompt_a", "prompt_b"):
                self.assertLess(len(p[key].split()) * 1.3, 75, msg=f"{name}:{key}")

    def test_cli_parser_defaults(self):
        args = da.build_parser().parse_args(["--output", "x", "--seeds", "1", "2"])
        self.assertEqual((args.steps, args.guidance, args.size, args.verify),
                         (30, 6.0, 1024, 2))
        self.assertEqual(args.model, "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(args.seeds, [1, 2])


def tiny_pipeline_dir(root: Path, sdxl: bool) -> Path:
    """Save a tiny random SD / SDXL pipeline with a character-level CLIP tokenizer."""
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )
    from transformers.models.clip.tokenization_clip import bytes_to_unicode

    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)}
    vocab.update({c + "</w>": len(chars) + i for i, c in enumerate(chars)})
    vocab["<|startoftext|>"], vocab["<|endoftext|>"] = len(vocab), len(vocab) + 1
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"),
                              model_max_length=77)
    torch.manual_seed(0)
    text_cfg = dict(vocab_size=len(vocab), intermediate_size=16, num_attention_heads=2,
                    num_hidden_layers=2, max_position_embeddings=77,
                    bos_token_id=vocab["<|startoftext|>"], eos_token_id=vocab["<|endoftext|>"])
    vae = AutoencoderKL(block_out_channels=(8,), in_channels=3, out_channels=3,
                        latent_channels=4, norm_num_groups=4,
                        down_block_types=("DownEncoderBlock2D",),
                        up_block_types=("UpDecoderBlock2D",))
    if sdxl:
        enc1 = CLIPTextModel(CLIPTextConfig(hidden_size=6, **text_cfg))
        enc2 = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=6, projection_dim=10,
                                                          **text_cfg))
        pipe = StableDiffusionXLPipeline(vae, enc1, enc2, tokenizer, tokenizer, tiny_unet(True),
                                         tiny_scheduler())
    else:
        enc = CLIPTextModel(CLIPTextConfig(hidden_size=12, **text_cfg))
        pipe = StableDiffusionPipeline(vae, enc, tokenizer, tiny_unet(False), tiny_scheduler(),
                                       safety_checker=None, feature_extractor=None,
                                       requires_safety_checker=False)
    out = root / ("sdxl" if sdxl else "sd")
    pipe.save_pretrained(out)
    return out


class CliTests(unittest.TestCase):
    def test_cli_end_to_end_with_tiny_saved_pipelines(self):
        for sdxl in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = tiny_pipeline_dir(root, sdxl)
                out = root / "atlas"
                # The test tokenizer is character-level (77 characters), so use short prompts.
                scenes = root / "scenes.json"
                scenes.write_text(json.dumps({"negative_prompt": "blur", "scenes": {"kiosk": {
                    "shared": "a kiosk", "advertisers": [
                        {"name": "Yellowco", "brief": "yellow cans"},
                        {"name": "Blueco", "brief": "blue cans"}]}}}))
                code = da.main(["--model", str(model), "--local-files-only", "--device", "cpu",
                                "--scenes-file", str(scenes), "--scene", "kiosk",
                                "--seeds", "1", "2", "--steps", "4",
                                "--guidance", "6", "--size", "8", "--verify", "2",
                                "--jpeg-size", "0", "--output", str(out)])
                self.assertEqual(code, 0)
                index = json.loads((out / "index.json").read_text())
                self.assertEqual([a["seed"] for a in index["atlases"]], [1, 2])
                for entry in index["atlases"]:
                    doc = json.loads((out / entry["atlas"]).read_text())
                    # main() always labels real pipeline inference; here the weights are
                    # random, which the test (not the CLI) knows.
                    self.assertEqual(doc["mode"], "live_model")
                    man = doc["manifest"]
                    self.assertEqual(man["pipeline_kind"], "sdxl" if sdxl else "sd")
                    self.assertEqual(man["seed"], entry["seed"])
                    self.assertFalse(man["bids_in_prompts"])
                    self.assertIn("packages", man["environment"])
                    self.assertTrue(all(v["matches_direct_run"] for v in doc["verification"]))
                    self.assertEqual(len(doc["verification"]), min(2, len(doc["leaves"])))
                    self.assertGreater(len(doc["leaves"]), 1)  # final step is always contested
                    self.assertFalse(any(v["truncated"]
                                         for v in man["prompt_token_lengths"].values()))
                    for leaf in doc["leaves"]:
                        self.assertTrue((out / leaf["image"]).exists())

    def test_cli_refuses_truncated_briefs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = tiny_pipeline_dir(root, False)
            with self.assertRaises(SystemExit) as caught:
                da.main(["--model", str(model), "--local-files-only", "--device", "cpu",
                         "--scene", "cafe", "--steps", "2", "--size", "8",
                         "--output", str(root / "atlas")])
            self.assertIn("truncated", str(caught.exception))

    def test_cli_time_budget_skips_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = tiny_pipeline_dir(root, False)
            out = root / "atlas"
            da.main(["--model", str(model), "--local-files-only", "--device", "cpu",
                     "--seeds", "1", "2", "--steps", "3", "--size", "8", "--output", str(out),
                     "--time-budget-seconds", "1e-9", "--allow-truncation"])
            index = json.loads((out / "index.json").read_text())
            self.assertEqual(index["atlases"], [])
            self.assertEqual([s["seed"] for s in index["skipped"]], [1, 2])


class TapeOrderTests(unittest.TestCase):
    def test_tape_order_is_initial_latent_then_steps(self):
        s = sampler()
        latent, tapes = s.tapes(9)
        rng = np.random.default_rng(9)
        z0 = rng.standard_normal(s.numel)
        expected = torch.from_numpy(z0.reshape(1, *LATENT)).float() * s.scheduler.init_noise_sigma
        self.assertTrue(torch.equal(latent, expected))
        from token_auction_lab.gaussian_auction import GaussianTape

        first = GaussianTape.draw(rng, s.numel)
        self.assertEqual(first.uniforms, tapes[0].uniforms)
        np.testing.assert_array_equal(first.z, tapes[0].z)
        self.assertEqual(len(tapes), STEPS)


if __name__ == "__main__":
    unittest.main()
