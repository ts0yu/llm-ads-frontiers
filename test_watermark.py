"""Keyed sponsor watermark (token_auction_lab/watermark.py).

Synthetic checks only: keyed draws are exact samples, the auction's marginal and
payments are unchanged, detection works and has the stated false-positive rate on a
random toy world, and watermarked atlases on a tiny random Qwen3 are exact. No claim
about real text, robustness to editing, or real advertisers.
"""

import math
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401

    from auction import RandomTape, TwoBidderAuction
    from test_atlas import PROMPTS, tiny_model
    from test_span_hold import PieceTokenizer
    from token_auction_lab import span_hold
    from token_auction_lab import watermark as wm
    from token_auction_lab.atlas import build_atlas, check_partition, verify, verify_point
except ImportError:
    torch = None

BIDDERS = ["Lantern House", "Bayview Hotel"]


@unittest.skipIf(torch is None, "Install the inference extra to run watermark tests.")
class KeyedDrawTests(unittest.TestCase):
    def setUp(self):
        self.keys = wm.demo_keys(BIDDERS)

    def test_uniforms_are_deterministic_keyed_and_in_the_open_interval(self):
        a = wm.keyed_uniforms(self.keys.a, [1, 2, 3], 1000)
        self.assertTrue(np.array_equal(a, wm.keyed_uniforms(self.keys.a, [1, 2, 3], 1000)))
        self.assertFalse(np.array_equal(a, wm.keyed_uniforms(self.keys.b, [1, 2, 3], 1000)))
        self.assertFalse(np.array_equal(a, wm.keyed_uniforms(self.keys.a, [1, 2, 4], 1000)))
        # Only the last CONTEXT_TOKENS tokens matter.
        self.assertTrue(np.array_equal(
            wm.keyed_uniforms(self.keys.a, [9, 9, 1, 2, 3, 4], 50),
            wm.keyed_uniforms(self.keys.a, [7, 1, 2, 3, 4], 50)))
        self.assertTrue(((a > 0) & (a < 1)).all())
        self.assertEqual(wm.keyed_uniform(self.keys.a, [1, 2, 3], 17), a[17])

    def test_exponential_race_is_an_exact_draw(self):
        q = np.array([0.5, 0.25, 0.15, 0.1, 0.0])
        n = 40000
        counts = np.bincount([wm.keyed_draw(q, self.keys.platform, [k, k >> 8]) for k in range(n)],
                             minlength=len(q))
        self.assertEqual(counts[-1], 0)  # zero-probability tokens are never drawn
        for i in range(4):
            sd = math.sqrt(n * q[i] * (1 - q[i]))
            self.assertLess(abs(counts[i] - n * q[i]), 5 * sd, (i, counts))

    def test_auction_marginal_and_payments_are_the_reference(self):
        rng = np.random.default_rng(3)
        p_a = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
        p_b = np.array([0.1, 0.0, 0.2, 0.1, 0.35, 0.25])
        auction = TwoBidderAuction(p_a, p_b)
        lam, n = 0.3, 30000
        counts = np.zeros(len(p_a))
        for k in range(n):
            tape = RandomTape.draw(rng)
            reference = auction.resolve(lam, 1 - lam, tape)
            keyed, role = wm.resolve(auction, lam, 1 - lam, tape, self.keys, [k, k >> 7])
            self.assertEqual((keyed.branch, keyed.payment_a, keyed.payment_b),
                             (reference.branch, reference.payment_a, reference.payment_b))
            self.assertEqual(role, {"shared": "platform"}.get(keyed.branch, keyed.branch))
            counts[keyed.token_id] += 1
        mixture = lam * p_a + (1 - lam) * p_b
        for i, m in enumerate(mixture):
            sd = math.sqrt(n * m * (1 - m)) or 1.0
            self.assertLess(abs(counts[i] - n * m), 5 * sd, (i, counts, mixture))

    def test_floor_keying_changes_attribution_not_the_distribution(self):
        floor_keys = wm.demo_keys(BIDDERS, "floor")
        self.assertEqual(wm.shared_role(floor_keys, None), "platform")
        self.assertEqual(wm.shared_role(floor_keys, "B"), "B")
        self.assertEqual(wm.shared_role(self.keys, "B"), "platform")
        with self.assertRaises(ValueError):
            wm.WatermarkKeys(b"p", b"a", b"b", shared="sometimes")
        rng = np.random.default_rng(5)
        p_a = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
        p_b = np.array([0.35, 0.05, 0.2, 0.1, 0.3])
        auction = TwoBidderAuction(p_a, p_b)
        lam, n = 0.6, 30000
        counts = np.zeros(len(p_a))
        for k in range(n):
            tape = RandomTape.draw(rng)
            reference = auction.resolve(lam, 1 - lam, tape)
            keyed, role = wm.resolve(auction, lam, 1 - lam, tape, floor_keys, [k, 1], "A")
            self.assertEqual((keyed.branch, keyed.payment_a, keyed.payment_b),
                             (reference.branch, reference.payment_a, reference.payment_b))
            self.assertEqual(role, "A" if keyed.branch in ("shared", "A") else "B")
            counts[keyed.token_id] += 1
        mixture = lam * p_a + (1 - lam) * p_b
        for i, m in enumerate(mixture):
            sd = math.sqrt(n * m * (1 - m)) or 1.0
            self.assertLess(abs(counts[i] - n * m), 5 * sd, (i, counts, mixture))
        self.assertEqual(floor_keys.manifest()["shared_tokens"], "floor")

    def test_gamma_tail_and_manifest_hide_the_keys(self):
        self.assertAlmostEqual(wm.gamma_upper_tail(1, 2.0), math.exp(-2.0))
        self.assertAlmostEqual(wm.gamma_upper_tail(2, 3.0), math.exp(-3.0) * 4.0)
        self.assertAlmostEqual(wm.gamma_upper_tail(50, 50.0), 0.4812, places=3)
        manifest = self.keys.manifest()
        text = repr(manifest)
        for role in wm.ROLES:
            self.assertNotIn(self.keys.key(role).hex(), text)
        self.assertEqual(manifest["key_source"], wm.DEMO_SOURCE)
        other = wm.WatermarkKeys(b"x" * 32, b"y" * 32, b"z" * 32)
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            wm.keys_for_manifest({"watermark": manifest}, BIDDERS, other)
        with self.assertRaisesRegex(ValueError, "private keys"):
            wm.keys_for_manifest({"watermark": other.manifest()}, BIDDERS)


# ------------------------------------------------------------------ random toy world

VOCAB = 60
EOS = VOCAB - 1


class RandomWorldTokenizer:
    eos_token_id = EOS

    def decode(self, ids, **kwargs):
        return "".join(f" w{i}" for i in ids if i != EOS)


class RandomWorldDecoder:
    """Two contexts with different, context-dependent, fairly high-entropy policies."""

    def __init__(self):
        self.generated = []

    def distributions(self, temperature):
        out = []
        for side in (1, 2):
            seed = hash((side, tuple(self.generated[-3:]))) % (2**32)
            p = np.random.default_rng(seed).dirichlet(np.full(VOCAB, 0.4))
            p[EOS] = 0.002
            out.append(p / p.sum())
        return out

    def advance(self, token_id):
        self.generated.append(token_id)


def world_text(lam, seed, keys, max_tokens=80):
    return span_hold.run(
        RandomWorldDecoder(), RandomWorldTokenizer(), lam, 1 - lam,
        tapes=span_hold.draw_tapes(seed, max_tokens),
        holds=span_hold.hold_uniforms(seed, max_tokens), stops={EOS},
        temperature=1.0, max_tokens=max_tokens, hold=False, watermark=keys,
    )


@unittest.skipIf(torch is None, "Install the inference extra to run watermark tests.")
class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.keys = wm.demo_keys(BIDDERS)

    def test_detects_who_won_and_records_the_truth(self):
        """Detection rates over seeds (alpha = 0.01). Power is statistical, not certain."""
        hits = {lam: {"A": 0, "B": 0} for lam in (1.0, 0.5, 0.0)}
        seeds = range(10)
        for seed in seeds:
            for lam in hits:
                text = world_text(lam, seed, self.keys)
                found = wm.detect(text["token_ids"], self.keys)
                for role in ("A", "B"):
                    hits[lam][role] += found[role]["p_value"] < 0.01
                roles = [s["key"] for s in text["steps"]]
                self.assertEqual(len(roles), len(text["token_ids"]))
                self.assertTrue(set(roles) <= set(wm.ROLES))
                if lam == 1.0:
                    self.assertNotIn("B", roles)
        self.assertGreaterEqual(hits[1.0]["A"], 9, hits)
        self.assertGreaterEqual(hits[0.0]["B"], 9, hits)
        self.assertLessEqual(hits[1.0]["B"] + hits[0.0]["A"], 2, hits)  # false alarms
        self.assertGreaterEqual(min(hits[0.5].values()), 7, hits)  # both sponsors, mixed

    def test_unwatermarked_text_has_the_stated_false_positive_rate(self):
        hits = {role: 0 for role in wm.ROLES}
        n = 300
        for seed in range(n):
            text = world_text(0.5, 1000 + seed, None)
            result = wm.detect(text["token_ids"], self.keys)
            for role in wm.ROLES:
                hits[role] += result[role]["p_value"] < 0.01
        for role in wm.ROLES:
            # Expect 3 of 300 at alpha = 0.01; allow binomial slack.
            self.assertLessEqual(hits[role], 10, hits)

    def test_wrong_keys_see_nothing(self):
        other = wm.WatermarkKeys(b"p" * 32, b"a" * 32, b"b" * 32)
        found = wm.detect(world_text(1.0, 7, self.keys)["token_ids"], other)
        self.assertTrue(all(found[role]["p_value"] > 1e-3 for role in wm.ROLES))


# ------------------------------------------------------------------ exact atlas, tiny model

@unittest.skipIf(torch is None, "Install the inference extra to run watermark tests.")
class WatermarkAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tokenizer = PieceTokenizer()
        cls.keys = wm.demo_keys(BIDDERS)

    def test_watermarked_atlases_are_exact_with_and_without_hold(self):
        for span in (False, True):
            for seed in range(3):
                atlas = build_atlas(self.model, self.tokenizer, PROMPTS, BIDDERS, seed=seed,
                                    temperature=1.0, max_tokens=14, mode="synthetic_model_test",
                                    span_hold=span, watermark=self.keys)
                check_partition(atlas)
                self.assertEqual(atlas["manifest"]["watermark"]["scheme"], wm.SCHEME)
                self.assertEqual("span_hold" in atlas["manifest"], span)
                for leaf in atlas["leaves"]:
                    self.assertTrue(all(s["key"] in wm.ROLES for s in leaf["steps"]))
                    lam = (leaf["lo"] + leaf["hi"]) / 2
                    record = verify_point(atlas, leaf, lam, 1 - lam, self.model,
                                          self.tokenizer, workdir=None, kind="midpoint")
                    self.assertTrue(record["matches_direct_run"], (span, seed, leaf["id"]))
                    self.assertIn("watermark", record["sampler"])
                self.assertTrue(all(r["matches_direct_run"]
                                    for r in verify(atlas, self.model, self.tokenizer, n=2)))


    def test_floor_variant_atlases_are_exact_and_follow_the_last_contest_winner(self):
        floor_keys = wm.demo_keys(BIDDERS, "floor")
        for span in (False, True):
            atlas = build_atlas(self.model, self.tokenizer, PROMPTS, BIDDERS, seed=2,
                                temperature=1.0, max_tokens=14, mode="synthetic_model_test",
                                span_hold=span, watermark=floor_keys)
            self.assertEqual(atlas["manifest"]["watermark"]["shared_tokens"], "floor")
            for leaf in atlas["leaves"]:
                floor = None
                for step in leaf["steps"]:
                    if step["branch"] == "shared":
                        self.assertEqual(step["key"], floor or "platform")
                    if step["branch"] in ("A", "B"):
                        floor = step["branch"]
                lam = (leaf["lo"] + leaf["hi"]) / 2
                record = verify_point(atlas, leaf, lam, 1 - lam, self.model, self.tokenizer,
                                      workdir=None, kind="midpoint")
                self.assertTrue(record["matches_direct_run"], (span, leaf["id"]))

    def test_replay_attribution_recovers_every_keyed_token(self):
        from itertools import product

        from token_auction_lab import watermark_replay as replay

        probs = [0.2, 0.5, 0.05, 0.9]
        for k in range(6):
            brute = sum(math.prod(p if b else 1 - p for p, b in zip(probs, bits))
                        for bits in product((0, 1), repeat=4) if sum(bits) >= k)
            self.assertAlmostEqual(replay.poisson_binomial_upper_tail(probs, k), brute)
        for shared in ("platform", "floor"):
            keys = wm.demo_keys(BIDDERS, shared)
            for span in (False, True):
                atlas = build_atlas(self.model, self.tokenizer, PROMPTS, BIDDERS, seed=3,
                                    temperature=1.0, max_tokens=14, mode="synthetic_model_test",
                                    span_hold=span, watermark=keys)
                for leaf in atlas["leaves"]:
                    found = replay.attribute(self.model, [ids for _, ids in PROMPTS],
                                             leaf["token_ids"], keys, temperature=1.0,
                                             span_hold=span)
                    seen, first = set(), {}
                    for t, tok in enumerate(leaf["token_ids"]):
                        pair = (tuple(leaf["token_ids"][max(0, t - wm.CONTEXT_TOKENS):t]), tok)
                        if pair not in seen:
                            seen.add(pair)
                            first[t] = True
                    for t, step in enumerate(leaf["steps"]):
                        if t in first:
                            self.assertIn(t, found["roles"][step["key"]]["matched_steps"],
                                          (shared, span, leaf["id"], t, step["key"]))
                    for role in wm.ROLES:
                        p = found["roles"][role]["p_value"]
                        self.assertTrue(0.0 <= p <= 1.0)

    def test_evaluation_script_reports_shares_false_alarms_and_edits(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "watermark_eval", Path(__file__).resolve().parent / "scripts" / "watermark_eval.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        common = dict(temperature=1.0, max_tokens=14, mode="synthetic_model_test")
        marked = build_atlas(self.model, self.tokenizer, PROMPTS, BIDDERS, seed=1,
                             watermark=self.keys, **common)
        plain = build_atlas(self.model, self.tokenizer, PROMPTS, BIDDERS, seed=1, **common)
        report = evaluation.evaluate([("marked", marked)], [("plain", plain)], alpha=0.01)
        row = report["atlases"][0]
        for role in wm.ROLES:
            self.assertGreaterEqual(row["drew_share"][role], 0.0)
            self.assertLessEqual(row["detected_share"][role], 1.0 + 1e-12)
        self.assertEqual(len(report["probes"]), len(evaluation.PROBE_LAMBDAS))
        self.assertIn("substitute:0.1", report["summary"]["robustness_detection_rate"])
        self.assertEqual(report["summary"]["null_atlases"], 1)
        rng = np.random.default_rng(0)
        self.assertEqual(evaluation.edit([1, 2, 3, 4], "truncate", 2, rng), [1, 2])
        self.assertEqual(len(evaluation.edit(list(range(100)), "substitute", 0.5, rng)), 100)


if __name__ == "__main__":
    unittest.main()
