"""Exact bid-atlas tests. Model-loop tests use random tiny Qwen3 weights.

These are synthetic_model_test checks of the interval search, forking, tape
sequence and payments against direct run_round calls. They are not evidence of
model quality, latency, or truthful bidding.
"""

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from token_auction_lab.atlas import (
        analyze,
        build_atlas,
        check_partition,
        draw_tapes,
        fork_decoder,
        leaf_for,
        main,
        midpoint_bids,
        payments,
        verify,
        verify_point,
    )
    from token_auction_lab.decoder import SharedPrefixDecoder
    from token_auction_lab.runner import RoundConfig, run_round
except ImportError:
    torch = None


class TestTokenizer:
    eos_token_id = 16

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if not (kwargs.get("skip_special_tokens") and i == 16))


class ChatTokenizer(TestTokenizer):
    """Synthetic chat tokenizer for CLI plumbing tests; IDs come from a text hash."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def encode(self, text, add_special_tokens):
        return [b % 16 for b in hashlib.sha256(text.encode()).digest()[:12]]


PROMPTS = [("A", [1, 2, 3]), ("B", [9, 8, 7, 6])]
BIDDERS = ["3", "5"]


def tiny_model():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=17,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        eos_token_id=16,
        # Larger random weights give peaked, disagreeing distributions, so that
        # many steps are contested and the atlas has several leaves.
        initializer_range=0.5,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def synthetic_atlas(values_by_key, bounds):
    """Hand-built fixture: leaves with given (lo, hi] bounds and outcome values."""
    leaves = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        leaves.append(
            {
                "id": i,
                "lo": lo,
                "hi": hi,
                "parent": None,
                "split_step": None,
                "token_ids": [i],
                "text": str(i),
                "stop_reason": "length",
                "steps": [],
                "outcomes": {key: values[i] for key, values in values_by_key.items()},
            }
        )
    return {
        "schema": "bid-atlas-v1",
        "kind": "text",
        "mode": "synthetic_fixture",
        "bidders": ["Brand A", "Brand B"],
        "seed": 0,
        "max_steps": 10,
        "leaves": leaves,
        "splits": [{"step": i} for i in range(len(leaves) - 1)],
        "verification": [],
    }


class AnalyzeFixtureTests(unittest.TestCase):
    def test_monotonicity_classification_measure_and_critical_lambda(self):
        bounds = [0.0, 0.2, 0.5, 0.7, 1.0]
        atlas = synthetic_atlas(
            {
                "mentions:Brand A": [False, False, True, True],
                "mentions:Brand B": [True, True, True, False],
                "mentions:Other": [True, False, True, False],
                "mentions:Never": [False, False, False, False],
            },
            bounds,
        )
        result = analyze(atlas)
        a = result["outcomes"]["mentions:Brand A"]
        self.assertEqual(a["classification"], "monotone_increasing")
        self.assertEqual(a["switches"], 1)
        self.assertEqual(a["critical_lambda"], 0.5)
        self.assertEqual(a["true_intervals"], [[0.5, 1.0]])
        self.assertAlmostEqual(a["measure"], 0.5)
        self.assertTrue(a["consistent_with_expected"])
        b = result["outcomes"]["mentions:Brand B"]
        self.assertEqual(b["classification"], "monotone_decreasing")
        self.assertEqual(b["critical_lambda"], 0.7)
        self.assertEqual(b["true_intervals"], [[0.0, 0.7]])
        self.assertAlmostEqual(b["measure"], 0.7)
        self.assertTrue(b["consistent_with_expected"])
        other = result["outcomes"]["mentions:Other"]
        self.assertEqual(other["classification"], "non_monotone")
        self.assertEqual(other["switches"], 3)
        self.assertIsNone(other["critical_lambda"])
        self.assertFalse(other["monotone_increasing"] or other["monotone_decreasing"])
        self.assertAlmostEqual(other["measure"], 0.2 + 0.2)
        self.assertIsNone(other["expected_direction"])
        never = result["outcomes"]["mentions:Never"]
        self.assertEqual(never["classification"], "constant")
        self.assertTrue(never["monotone_increasing"] and never["monotone_decreasing"])
        self.assertEqual(never["measure"], 0.0)
        self.assertEqual(result["leaves"], 4)
        self.assertEqual(result["bound"], 11)
        self.assertTrue(result["within_bound"])

    def test_wrong_direction_is_reported_not_hidden(self):
        atlas = synthetic_atlas(
            {
                "mentions:Brand A": [True, False],
                "mentions:Brand B": [False, True],
            },
            [0.0, 0.4, 1.0],
        )
        result = analyze(atlas)["outcomes"]
        self.assertEqual(result["mentions:Brand A"]["classification"], "monotone_decreasing")
        self.assertFalse(result["mentions:Brand A"]["consistent_with_expected"])
        self.assertEqual(result["mentions:Brand B"]["classification"], "monotone_increasing")
        self.assertFalse(result["mentions:Brand B"]["consistent_with_expected"])
        self.assertEqual(result["mentions:Brand B"]["critical_lambda"], 0.4)

    def test_partition_checks_reject_gaps_and_excess_leaves(self):
        atlas = synthetic_atlas({"mentions:Brand A": [True, False]}, [0.0, 0.4, 1.0])
        check_partition(atlas)
        atlas["leaves"][1]["lo"] = 0.5
        with self.assertRaises(AssertionError):
            check_partition(atlas)
        atlas = synthetic_atlas({"mentions:Brand A": [True] * 12}, list(np.linspace(0, 1, 13)))
        with self.assertRaises(AssertionError):
            check_partition(atlas)


@unittest.skipIf(torch is None, "Install the inference extra to run model-loop tests.")
class AtlasModelTests(unittest.TestCase):
    max_tokens = 24

    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tokenizer = TestTokenizer()
        # Seed 1 with EOS enabled includes both eos and length leaves.
        cls.eos_atlas = build_atlas(
            cls.model,
            cls.tokenizer,
            PROMPTS,
            BIDDERS,
            seed=1,
            temperature=1.5,
            max_tokens=cls.max_tokens,
            metadata={"model_id": "random-tiny-qwen3-test"},
            mode="synthetic_model_test",
        )

    def setUp(self):
        self.model.generation_config.eos_token_id = 16
        self.tokenizer.eos_token_id = 16

    def disable_eos(self):
        self.model.generation_config.eos_token_id = None
        self.tokenizer.eos_token_id = None
        self.addCleanup(self.setUp)

    def build(self, seed, **kwargs):
        return build_atlas(
            self.model,
            self.tokenizer,
            PROMPTS,
            BIDDERS,
            seed=seed,
            temperature=kwargs.pop("temperature", 1.5),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            mode="synthetic_model_test",
            **kwargs,
        )

    def assert_partition(self, atlas):
        leaves = atlas["leaves"]
        self.assertEqual(leaves[0]["lo"], 0.0)
        self.assertEqual(leaves[-1]["hi"], 1.0)
        for left, right in zip(leaves, leaves[1:]):
            self.assertEqual(left["hi"], right["lo"])
        self.assertTrue(all(leaf["lo"] < leaf["hi"] for leaf in leaves))
        self.assertLessEqual(len(leaves), 1 + atlas["max_steps"])
        self.assertEqual(len(leaves), 1 + len(atlas["splits"]))
        self.assertEqual(len({s["step"] for s in atlas["splits"]}), len(atlas["splits"]))

    def test_tapes_equal_the_runner_sequence(self):
        self.disable_eos()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_round(
                self.model,
                self.tokenizer,
                PROMPTS,
                RoundConfig(0.3, 0.7, seed=11, max_tokens=12, checkpoint_steps=()),
                Path(tmp) / "round",
                {},
                mode="synthetic_model_test",
            )
            events = [
                json.loads(line)
                for line in (Path(tmp) / "round/events.jsonl").read_text().splitlines()
            ][:-1]
        self.assertEqual(result["tokens"], 12)
        self.assertEqual([e["tape"] for e in events], [asdict(t) for t in draw_tapes(11, 12)])
        # A longer precomputation has the same prefix: tapes do not depend on length.
        self.assertEqual(draw_tapes(11, 20)[:12], draw_tapes(11, 12))

    def test_fork_copies_state_without_aliasing(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7]])
        decoder.advance(9)
        before = [
            (
                list(c.token_ids),
                c.logits.clone(),
                c.cache.get_seq_length(),
                c.attention_mask.shape[1],
            )
            for c in decoder.contexts
        ]
        fork = fork_decoder(decoder)
        self.assertIs(fork.model, decoder.model)
        for original, copy in zip(decoder.contexts, fork.contexts):
            self.assertIsNot(original.cache, copy.cache)
            self.assertIsNot(original.token_ids, copy.token_ids)
            self.assertNotEqual(original.logits.data_ptr(), copy.logits.data_ptr())
            torch.testing.assert_close(original.logits, copy.logits, rtol=0, atol=0)
        for token in (12, 0, 5):
            fork.advance(token)
        self.assertEqual(decoder.generated, [9])
        for context, (ids, logits, length, mask) in zip(decoder.contexts, before):
            self.assertEqual(context.token_ids, ids)
            self.assertEqual(context.cache.get_seq_length(), length)
            self.assertEqual(context.attention_mask.shape[1], mask)
            torch.testing.assert_close(context.logits, logits, rtol=0, atol=0)
        # The original continues exactly like a fresh decoder that never forked.
        decoder.advance(3)
        fresh = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7]])
        fresh.advance(9)
        fresh.advance(3)
        for a, b in zip(decoder.contexts, fresh.contexts):
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        # The fork's caches still match an uncached full-prefix forward.
        self.assertLess(max(fork.check_cache(atol=2e-5, rtol=1e-4)), 2e-5)

    def test_schema_labels_and_split_semantics(self):
        atlas = self.eos_atlas
        self.assertEqual(atlas["schema"], "bid-atlas-v1")
        self.assertEqual(atlas["kind"], "text")
        self.assertEqual(atlas["mode"], "synthetic_model_test")
        self.assertEqual(atlas["bidders"], BIDDERS)
        self.assertEqual(atlas["parameter"], "lambda = bid_A / (bid_A + bid_B)")
        manifest = atlas["manifest"]
        self.assertEqual(manifest["model_id"], "random-tiny-qwen3-test")
        self.assertEqual([p["token_ids"] for p in manifest["prompts"]], [p[1] for p in PROMPTS])
        self.assertEqual(
            (manifest["seed"], manifest["temperature"], manifest["max_tokens"]),
            (1, 1.5, self.max_tokens),
        )
        json.dumps(atlas, allow_nan=False)
        by_id = {leaf["id"]: leaf for leaf in atlas["leaves"]}
        for split in atlas["splits"]:
            parent, child = by_id[split["parent"]], by_id[split["child"]]
            self.assertEqual(child["parent"], parent["id"])
            self.assertEqual(child["split_step"], split["step"])
            self.assertEqual(child["lo"], split["u"])
            self.assertEqual(parent["steps"][split["step"]]["token_id"], split["b_token"])
            self.assertEqual(child["steps"][split["step"]]["token_id"], split["a_token"])
            self.assertEqual(
                parent["token_ids"][: split["step"]], child["token_ids"][: split["step"]]
            )
        stops = {leaf["stop_reason"] for leaf in atlas["leaves"]}
        self.assertEqual(stops, {"eos", "length"})
        for leaf in atlas["leaves"]:
            self.assertEqual(
                leaf["text"],
                self.tokenizer.decode(
                    leaf["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
            )
            self.assertEqual(set(leaf["outcomes"]), {"mentions:3", "mentions:5"})
            self.assertEqual(leaf["outcomes"]["mentions:3"], "3" in leaf["text"])
            if leaf["stop_reason"] == "eos":
                self.assertEqual(leaf["token_ids"][-1], 16)
            else:
                self.assertEqual(len(leaf["token_ids"]), self.max_tokens)
            for step in leaf["steps"]:
                self.assertEqual(
                    set(step),
                    {
                        "step",
                        "branch",
                        "u",
                        "shared_mass",
                        "disagreement",
                        "token_id",
                        "candidate_a",
                        "candidate_b",
                    },
                )
                if step["branch"] == "A":
                    self.assertLessEqual(step["u"], leaf["lo"])
                elif step["branch"] == "B":
                    self.assertGreaterEqual(step["u"], leaf["hi"])

    def check_every_leaf(self, atlas, seed):
        self.assert_partition(atlas)
        self.assertGreaterEqual(len(atlas["leaves"]), 4, "Fixture should produce many leaves.")
        rng = np.random.default_rng(seed)
        with tempfile.TemporaryDirectory() as tmp:
            for leaf in atlas["leaves"]:
                lo, hi = leaf["lo"], leaf["hi"]
                interior = lo + (hi - lo) * float(rng.uniform(0.25, 0.75))
                scale = float(rng.uniform(0.5, 4.0))
                points = [
                    (*midpoint_bids(leaf), "midpoint"),
                    (scale * interior, scale * (1.0 - interior), "interior"),
                ]
                for bid_a, bid_b, kind in points:
                    record = verify_point(
                        atlas,
                        leaf,
                        bid_a,
                        bid_b,
                        self.model,
                        self.tokenizer,
                        workdir=Path(tmp),
                        kind=kind,
                    )
                    self.assertTrue(record["token_ids_match"], record)
                    self.assertTrue(record["steps_match"], record)
                    self.assertTrue(record["matches_direct_run"], record)
                    # Same expressions and summation order: equality, not only 1e-9.
                    self.assertEqual(record["payment_a_direct"], record["payment_a_atlas"])
                    self.assertEqual(record["payment_b_direct"], record["payment_b_atlas"])

    def test_every_leaf_matches_direct_runs_with_eos(self):
        self.check_every_leaf(self.eos_atlas, 101)

    def test_every_leaf_matches_direct_runs_without_eos(self):
        self.disable_eos()
        atlas = self.build(3)
        self.assertEqual({leaf["stop_reason"] for leaf in atlas["leaves"]}, {"length"})
        self.check_every_leaf(atlas, 103)

    def test_extreme_leaves_match_endpoint_bids_and_verify_records(self):
        atlas = self.build(1)
        self.assertIs(leaf_for(atlas, 1.0), atlas["leaves"][-1])
        self.assertIs(leaf_for(atlas, 0.0), atlas["leaves"][0])
        records = verify(atlas, self.model, self.tokenizer, 2)
        self.assertEqual(records, atlas["verification"])
        self.assertTrue(all(r["matches_direct_run"] for r in records), records)
        midpoints = [r["leaf"] for r in records if r["kind"] == "midpoint"]
        self.assertEqual(len(midpoints), 4)
        self.assertIn(atlas["leaves"][0]["id"], midpoints)
        self.assertIn(atlas["leaves"][-1]["id"], midpoints)
        endpoints = [r for r in records if r["kind"] == "endpoint"]
        self.assertEqual([(r["bid_a"], r["bid_b"]) for r in endpoints], [(1.0, 0.0), (0.0, 1.0)])
        self.assertEqual(
            [r["leaf"] for r in endpoints], [atlas["leaves"][-1]["id"], atlas["leaves"][0]["id"]]
        )
        for record in endpoints:
            # A sole bidder pays zero; the zero-bid side is never charged.
            self.assertEqual(record["payment_a_direct"] + record["payment_b_direct"], 0.0)

    def test_replay_fork_strategy_gives_the_same_atlas(self):
        replay = self.build(1, fork="replay")
        copied = self.build(1)
        self.assertEqual(replay["leaves"], copied["leaves"])
        self.assertEqual(replay["splits"], copied["splits"])
        self.assertEqual(copied["leaves"], self.eos_atlas["leaves"])

    def test_payments_formula_and_leaf_membership(self):
        atlas = self.eos_atlas
        contested = [
            leaf for leaf in atlas["leaves"] if any(s["branch"] != "shared" for s in leaf["steps"])
        ]
        self.assertTrue(contested)
        for leaf in contested:
            lam = (leaf["lo"] + leaf["hi"]) / 2
            total = 3.0
            result = payments(leaf, total * lam, total * (1 - lam))
            expected_a = math.fsum(
                total * (1 - lam) * s["u"] / (1 - s["u"])
                for s in leaf["steps"]
                if s["branch"] == "A"
            )
            expected_b = math.fsum(
                total * lam * (1 - s["u"]) / s["u"] for s in leaf["steps"] if s["branch"] == "B"
            )
            self.assertAlmostEqual(result["payment_a"], expected_a, places=9)
            self.assertAlmostEqual(result["payment_b"], expected_b, places=9)
            outside = leaf["hi"] + 1e-6 if leaf["hi"] < 1 else leaf["lo"] / 2
            with self.assertRaises(ValueError):
                payments(leaf, outside, 1 - outside)
        with self.assertRaises(ValueError):
            payments(atlas["leaves"][0], 0.0, 0.0)

    def test_analysis_on_model_atlas_is_consistent(self):
        result = analyze(self.eos_atlas)
        self.assertEqual(result["leaves"], len(self.eos_atlas["leaves"]))
        self.assertTrue(result["within_bound"])
        self.assertTrue(result["split_steps_distinct"])
        for key, outcome in result["outcomes"].items():
            measure = math.fsum(
                leaf["hi"] - leaf["lo"]
                for leaf in self.eos_atlas["leaves"]
                if leaf["outcomes"][key]
            )
            self.assertAlmostEqual(outcome["measure"], measure, places=12)

    def run_cli(self, argv):
        tokenizer = ChatTokenizer()
        metadata = {"model_id": "random-tiny-qwen3-test", "model_revision": None}
        with (
            patch(
                "token_auction_lab.atlas._load",
                return_value=(self.model, tokenizer, dict(metadata)),
            ),
            patch("builtins.print"),
        ):
            main(argv, mode="synthetic_model_test")

    def test_cli_writes_each_atlas_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "atlas"
            self.run_cli(
                [
                    "--seeds",
                    "1",
                    "2",
                    "--max-tokens",
                    "8",
                    "--temperature",
                    "1.5",
                    "--verify",
                    "1",
                    "--output",
                    str(out),
                ]
            )
            index = json.loads((out / "index.json").read_text())
            self.assertEqual([a["seed"] for a in index["atlases"]], [1, 2])
            for summary in index["atlases"]:
                atlas = json.loads((out / summary["file"]).read_text())
                self.assertEqual(atlas["schema"], "bid-atlas-v1")
                self.assertEqual(atlas["mode"], "synthetic_model_test")
                self.assertEqual(atlas["verification_status"], "all_match")
                self.assertEqual(atlas["manifest"]["scenario"], "transit-hotel")
                self.assertEqual(atlas["bidders"], ["HarborHop", "Lantern House"])
                self.assertIn("analysis", atlas)
                self.assertTrue(summary["verification"]["all_match"])
            self.run_cli(
                [
                    "--live-defaults",
                    "--seeds",
                    "4",
                    "--max-tokens",
                    "6",
                    "--no-verify",
                    "--output",
                    str(out),
                ]
            )
            atlas = json.loads((out / "live-defaults-seed4.json").read_text())
            self.assertEqual(atlas["bidders"], ["Bayview Hotel", "Lantern House"])
            self.assertEqual(atlas["verification_status"], "not_requested")
            inputs = atlas["manifest"]["live_inputs_without_bids"]
            self.assertNotIn("bid", json.dumps(inputs["advertisers"]))
            self.assertNotIn("bid", atlas["manifest"]["prompts"][0]["text"].lower())

    def test_cli_time_budget_skips_new_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "atlas"
            self.run_cli(
                [
                    "--seeds",
                    "1",
                    "2",
                    "--max-tokens",
                    "4",
                    "--output",
                    str(out),
                    "--time-budget-seconds",
                    "0",
                ]
            )
            index = json.loads((out / "index.json").read_text())
            self.assertEqual(index["atlases"], [])
            self.assertEqual(index["skipped_seeds"], [1, 2])


if __name__ == "__main__":
    unittest.main()
