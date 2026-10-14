"""Model-loop tests use random tiny Qwen weights: they are not quality evidence."""

import json
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auction import Allocation, RandomTape, TwoBidderAuction
from token_auction_lab.prompts import messages, scenarios

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from token_auction_lab.decoder import SharedPrefixDecoder, probabilities
    from token_auction_lab.runner import RoundConfig, run_round, top_tokens
except ImportError:
    torch = None


class PromptTests(unittest.TestCase):
    def test_same_user_task_and_private_facts_in_each_context(self):
        for scenario in scenarios().values():
            a, b = (messages(scenario, i) for i in (0, 1))
            self.assertEqual(a[-1], b[-1])
            self.assertNotEqual(a[0], b[0])
            for side, ad in enumerate(scenario["advertisers"]):
                system = (a, b)[side][0]["content"]
                self.assertIn(ad["brand"], system)
                self.assertNotIn(scenario["advertisers"][1 - side]["brand"], system)
                for fact in ad["facts"]:
                    self.assertIn(fact, system)
            self.assertNotIn("bid", a[0]["content"].lower())
            self.assertNotIn("bid", b[0]["content"].lower())

    def test_private_brief_edits_leave_the_other_prompt_unchanged(self):
        for scenario in scenarios().values():
            for side in (0, 1):
                edited = deepcopy(scenario)
                edited["advertisers"][side] = {
                    "brand": "PRIVATE_BRAND_SENTINEL",
                    "facts": ["PRIVATE_FACT_SENTINEL"],
                }
                own = messages(edited, side)
                other = messages(edited, 1 - side)
                self.assertIn("PRIVATE_BRAND_SENTINEL", own[0]["content"])
                self.assertIn("PRIVATE_FACT_SENTINEL", own[0]["content"])
                self.assertEqual(other, messages(scenario, 1 - side))


class TestTokenizer:
    eos_token_id = 16

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if not (kwargs.get("skip_special_tokens") and i == 16))


@unittest.skipIf(torch is None, "Install the inference extra to run model-loop tests.")
class DecoderTests(unittest.TestCase):
    def setUp(self):
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
            max_position_embeddings=128,
            eos_token_id=16,
        )
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()

    def test_cache_matches_uncached_and_suffixes_remain_identical(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7, 8]])
        self.assertIsNot(decoder.contexts[0].cache, decoder.contexts[1].cache)
        for token in [9, 1, 12, 0, 16]:
            decoder.advance(token)
            errors = decoder.check_cache(atol=2e-5, rtol=1e-4)
            self.assertLess(max(errors), 2e-5)
        self.assertEqual(decoder.generated, [9, 1, 12, 0, 16])
        for context in decoder.contexts:
            self.assertEqual(context.token_ids[len(context.prefix) :], decoder.generated)
            self.assertEqual(context.attention_mask.shape[1], len(context.token_ids))
            self.assertEqual(context.cache.get_seq_length(), len(context.token_ids))
        self.assertNotEqual(decoder.contexts[0].prefix, decoder.contexts[1].prefix)

    def test_full_float64_distributions_are_accepted_by_reference(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2], [3, 4, 5]])
        a, b = decoder.distributions(0.8)
        self.assertEqual(a.dtype, np.float64)
        self.assertEqual(a.shape, (17,))
        self.assertAlmostEqual(float(a.sum()), 1, places=14)
        self.assertGreater(TwoBidderAuction(a, b).disagreement, 0)
        for invalid in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                probabilities(decoder.contexts[0].logits, invalid)

    def test_production_sdpa_cache_matches_uncached(self):
        self.model.config._attn_implementation = "sdpa"
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7, 8]])
        for token in [9, 1, 12, 16]:
            decoder.advance(token)
            self.assertLess(max(decoder.check_cache(atol=2e-5, rtol=1e-4)), 2e-5)

    def test_cache_probability_guard_detects_corruption(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5]])
        decoder.advance(9)
        with torch.inference_mode():
            decoder.contexts[0].logits[0] += 5
        # Even when a caller allows a loose logit tolerance, a substantially
        # changed sampling distribution must fail the independent TV guard.
        with self.assertRaisesRegex(AssertionError, "probability TV"):
            decoder.check_cache(atol=10, rtol=0, max_probability_tv=1e-4)

    def test_top_table_is_union_and_is_not_renormalized(self):
        a = np.array([0.8, 0.1, 0.1])
        b = np.array([0.1, 0.1, 0.8])
        q = (a + b) / 2
        rows = top_tokens(TestTokenizer(), a, b, q, 1)
        self.assertEqual({row["token_id"] for row in rows}, {0, 2})
        self.assertAlmostEqual(sum(row["q"] for row in rows), 0.9)

    def test_round_logs_reference_payments_tape_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-round"
            result = run_round(
                self.model,
                TestTokenizer(),
                [("A", [1, 2]), ("B", [3, 4, 5])],
                RoundConfig(max_tokens=4, checkpoint_steps=(0,)),
                path,
                {"model_id": "random-tiny-qwen3-test"},
                mode="synthetic_model_test",
            )
            events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
            token_events = events[:-1]
            self.assertEqual(len(token_events), result["tokens"])
            self.assertTrue(result["shared_suffix_verified"])
            self.assertTrue(all(event["mode"] == "synthetic_model_test" for event in events))
            from auction import RandomTape

            with np.load(path / "checkpoint-000.npz", allow_pickle=False) as checkpoint:
                auction = TwoBidderAuction(checkpoint["p_a"], checkpoint["p_b"])
                first = token_events[0]
                replay = auction.resolve(1, 1, RandomTape(**first["tape"]))
                self.assertEqual(replay.token_id, first["token_id"])
                self.assertEqual(replay.payment_a, first["payment_a"])
            self.assertAlmostEqual(sum(e["payment_a"] for e in token_events), result["payment_a"])

    def test_eos_is_fed_to_both_contexts_and_stops_generation(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "token_auction_lab.runner.TwoBidderAuction.resolve",
                return_value=Allocation(16, "shared", 0, 0, None, None),
            ),
        ):
            result = run_round(
                self.model,
                TestTokenizer(),
                [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=4),
                Path(tmp) / "eos",
                {},
                mode="synthetic_model_test",
            )
            self.assertEqual(result["token_ids"], [16])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertTrue(result["shared_suffix_verified"])
            self.assertEqual(len(result["cache_max_absolute_errors"]), 2)

    def test_length_cap_and_baseline_without_auction(self):
        self.model.generation_config.eos_token_id = None
        tokenizer = TestTokenizer()
        tokenizer.eos_token_id = None
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "token_auction_lab.runner.TwoBidderAuction.resolve",
                side_effect=AssertionError("The no-ad path must not run an auction"),
            ),
        ):
            result = run_round(
                self.model,
                tokenizer,
                [("No ad", [1, 2])],
                RoundConfig(0, 0, baseline=True, max_tokens=3),
                Path(tmp) / "baseline",
                {},
                mode="synthetic_model_test",
            )
            self.assertEqual(result["tokens"], 3)
            self.assertEqual(result["stop_reason"], "length")
            self.assertEqual(result["payment_a"] + result["payment_b"], 0)

    def test_config_rejects_zero_pair_and_unbounded_lengths(self):
        for kwargs in [
            {"bid_a": 0, "bid_b": 0},
            {"max_tokens": 4097},
            {"max_tokens": 0},
            {"max_tokens": 1.5},
            {"max_tokens": True},
            {"bid_a": -1},
            {"bid_b": float("nan")},
        ]:
            with self.assertRaises(ValueError):
                RoundConfig(**kwargs)
        self.assertTrue(RoundConfig(0, 0, baseline=True).baseline)
        self.assertEqual(RoundConfig(max_tokens=97).max_tokens, 97)
        self.assertEqual(RoundConfig(max_tokens=4096).max_tokens, 4096)

    def test_generation_passes_96_then_eos_reaches_all_contexts(self):
        events = []
        with (tempfile.TemporaryDirectory() as tmp, patch(
            "token_auction_lab.runner.TwoBidderAuction.resolve",
            side_effect=[Allocation(5, "shared", 0, 0, None, None)] * 100
                + [Allocation(16, "shared", 0, 0, None, None)],
        )):
            result = run_round(self.model, TestTokenizer(), [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=128), Path(tmp)/"long", {},
                mode="synthetic_model_test", on_event=events.append,
                no_ad_prompt=("User only", [4, 5, 6]))
            self.assertEqual(result["token_ids"], [5] * 100 + [16])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(len([e for e in events if e["type"] == "token"]), 101)
            self.assertTrue(result["shared_suffix_verified"])
            self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])
            self.assertEqual(len(result["cache_max_absolute_errors"]), 2)

    def test_user_stop_preserves_selected_tokens_charges_and_caches(self):
        tokens = []
        with tempfile.TemporaryDirectory() as tmp, patch(
            "token_auction_lab.runner.eos_ids", return_value=set()
        ):
            result = run_round(self.model, TestTokenizer(), [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=128), Path(tmp)/"stopped", {},
                mode="synthetic_model_test", no_ad_prompt=("User only", [4, 5]),
                on_event=lambda e: tokens.append(e) if e["type"] == "token" else None,
                should_stop=lambda: len(tokens) == 3)
            self.assertEqual(result["stop_reason"], "user_stop")
            self.assertEqual(result["token_ids"], [e["token_id"] for e in tokens])
            self.assertEqual(result["tokens"], 3)
            for side in ("a", "b"):
                self.assertEqual(result[f"payment_{side}"], sum(e[f"payment_{side}"] for e in tokens))
            self.assertEqual(len(result["cache_max_absolute_errors"]), 2)
            self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])
            self.assertEqual(json.loads((Path(tmp)/"stopped/result.json").read_text()), result)

    def test_stop_before_first_token_has_valid_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_round(self.model, TestTokenizer(), [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=128), Path(tmp)/"empty", {},
                mode="synthetic_model_test", should_stop=lambda: True)
            self.assertEqual(result["stop_reason"], "user_stop")
            self.assertEqual(result["token_ids"], [])
            self.assertEqual(result["text"], "")
            self.assertEqual(result["payment_a"] + result["payment_b"], 0)
            self.assertIsNone(result["seconds_per_token"])
            self.assertIsNone(result["time_to_first_token_seconds"])

    def test_baseline_ends_on_eos_with_large_safety_limit(self):
        p = np.zeros(17)
        p[16] = 1
        with (tempfile.TemporaryDirectory() as tmp,
            patch.object(SharedPrefixDecoder, "distributions", return_value=[p]),
            patch("token_auction_lab.runner.TwoBidderAuction", side_effect=AssertionError("No auction"))):
            result = run_round(self.model, TestTokenizer(), [("User", [1, 2])],
                RoundConfig(0, 0, baseline=True, max_tokens=4096), Path(tmp)/"baseline", {},
                mode="synthetic_model_test")
            self.assertEqual(result["token_ids"], [16])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(result["payment_a"] + result["payment_b"], 0)
            self.assertEqual(len(result["cache_max_absolute_errors"]), 1)

    def test_continuation_limit_returns_to_auction_and_carried_eos_reaches_all_caches(self):
        original = SharedPrefixDecoder.distributions
        events = []

        def distributions(decoder, temperature):
            if len(decoder.contexts) == 1:
                return original(decoder, temperature)
            a, b = np.zeros(17), np.zeros(17)
            step = len(decoder.generated)
            if step == 0:
                a[2], b[3] = 1, 1
            elif step < 5:
                a[5], a[0], b[6], b[0] = .98, .02, .99, .01
            else:
                a[5], b[16], b[0] = 1, .98, .02
            return [a, b]

        with (tempfile.TemporaryDirectory() as tmp,
            patch.object(SharedPrefixDecoder, "distributions", distributions),
            patch("token_auction_lab.runner.RandomTape.draw", side_effect=[
                RandomTape(.5, .5, .5, .25), RandomTape(.5, .5, .5, .9),
            ]) as draws):
            result = run_round(self.model, TestTokenizer(), [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=12, continuation_tokens=3), Path(tmp)/"carry", {},
                mode="synthetic_model_test", no_ad_prompt=("User", [4]),
                on_event=lambda e: events.append(e) if e["type"] == "token" else None)
            self.assertEqual(result["token_ids"], [2, 5, 5, 5, 6, 16])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(result["carried_tokens"], 4)
            self.assertEqual(draws.call_count, 2)
            self.assertAlmostEqual(result["payment_a"], 1/3)
            self.assertAlmostEqual(result["payment_b"], 1/9)
            self.assertTrue(result["shared_suffix_verified"])
            self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])
            for step in (1, 2, 3, 5):
                event = events[step]
                self.assertEqual(event["selection_rule"], "winner_continuation")
                self.assertIsNone(event["tape"])
                self.assertEqual(event["payment_a"] + event["payment_b"], 0)
                self.assertEqual(event["selected_probabilities"]["p_output"], 1)
                self.assertLess(event["selected_probabilities"]["q"], 1)
            self.assertEqual(events[4]["selection_rule"], "auction")
            manifest = json.loads((Path(tmp)/"carry/manifest.json").read_text())
            self.assertEqual(manifest["sampler"], "experimental_winner_continuation")
            self.assertEqual(manifest["config"]["continuation_threshold"], .98)

    def test_confidence_miss_auctions_the_same_step(self):
        events = []
        p = np.zeros(17)
        p[5], p[6] = .97, .03
        with (tempfile.TemporaryDirectory() as tmp,
            patch.object(SharedPrefixDecoder, "distributions", return_value=[p, p]),
            patch("token_auction_lab.runner.TwoBidderAuction.resolve",
                side_effect=[Allocation(5, "A", .2, 0, 5, 6),
                    Allocation(16, "B", 0, .4, 5, 16)]) as auctions):
            result = run_round(self.model, TestTokenizer(), [("A", [1]), ("B", [2])],
                RoundConfig(max_tokens=10, continuation_tokens=3), Path(tmp)/"miss", {},
                mode="synthetic_model_test", on_event=events.append)
            self.assertEqual(result["token_ids"], [5, 16])
            self.assertEqual(result["carried_tokens"], 0)
            self.assertEqual(auctions.call_count, 2)
            self.assertEqual(result["payment_a"], .2)
            self.assertEqual(result["payment_b"], .4)

    def test_live_runs_only_ad_response_and_keeps_stop_support(self):
        from token_auction_lab.live import InferenceEngine, LiveRequest, default_inputs

        class ChatTokenizer(TestTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)

            def encode(self, text, **kwargs):
                return [1, 2]

        for phase in ("ad", "no-ad"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                engine = InferenceEngine()
                engine.model = self.model
                engine.tokenizer = ChatTokenizer()
                engine.metadata = {"model_id": "synthetic-tiny-qwen"}
                stopped = []
                observed = []

                def emit(event):
                    observed.append(event)
                    if (phase == "ad" and event["type"] == "token"
                        or phase == "no-ad" and event["type"] == "no_ad_event"
                        and event["event"]["type"] == "token"):
                        stopped.append(True)

                with (patch("token_auction_lab.runner.eos_ids", return_value=set()),
                    patch("token_auction_lab.runner.run_round",
                        side_effect=lambda *a, **kw: run_round(*a, mode="synthetic_model_test", **kw))):
                    result = engine.run(LiveRequest.parse({**default_inputs(), "max_tokens": 4}),
                        Path(tmp)/"live", emit, lambda *args: None, should_stop=lambda: bool(stopped))
                if phase == "ad":
                    self.assertEqual(result["tokens"], 1)
                    self.assertEqual(result["stop_reason"], "user_stop")
                    self.assertFalse(any(e["type"] == "no_ad_event" for e in observed))
                    self.assertFalse((Path(tmp)/"live/no-ad").exists())
                else:
                    self.assertEqual(result["tokens"], 4)
                    self.assertEqual(result["stop_reason"], "length")
                self.assertNotIn("no_ad_result", result)
                self.assertFalse(any(e["type"] == "no_ad_event" for e in observed))
                self.assertFalse((Path(tmp)/"live/no-ad").exists())
                self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])

    def test_no_ad_observer_matches_full_prefix_without_changing_auction(self):
        self.model.generation_config.eos_token_id = None
        tokenizer = TestTokenizer()
        tokenizer.eos_token_id = None
        prompts = [("A", [1, 2]), ("B", [3, 4, 5])]
        reference_ids = [7, 8, 9, 10]
        captured = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = run_round(self.model, tokenizer, prompts, RoundConfig(max_tokens=4),
                root / "plain", {}, mode="synthetic_model_test")
            compared = run_round(self.model, tokenizer, prompts, RoundConfig(max_tokens=4),
                root / "compared", {}, mode="synthetic_model_test",
                no_ad_prompt=("User only", reference_ids),
                on_distributions=lambda step, vectors: captured.append(vectors))
            self.assertEqual(plain["token_ids"], compared["token_ids"])
            self.assertEqual(plain["payment_a"], compared["payment_a"])
            self.assertEqual(plain["payment_b"], compared["payment_b"])
            self.assertEqual(len(compared["cache_max_absolute_errors"]), 2)
            self.assertTrue(compared["no_ad_reference_validation"]["shared_suffix_verified"])
            original = [json.loads(line) for line in (root / "plain/events.jsonl").read_text().splitlines()][:-1]
            events = [json.loads(line) for line in (root / "compared/events.jsonl").read_text().splitlines()][:-1]
            for step, (vectors, event) in enumerate(zip(captured, events)):
                self.assertEqual(len(vectors), 3)
                self.assertEqual(event["tape"], original[step]["tape"])
                self.assertEqual(event["branch"], original[step]["branch"])
                self.assertEqual(event["candidate_a"], original[step]["candidate_a"])
                self.assertEqual(event["candidate_b"], original[step]["candidate_b"])
                ids = torch.tensor([reference_ids + compared["token_ids"][:step]])
                with torch.inference_mode():
                    logits = self.model(input_ids=ids, use_cache=False).logits[0, -1]
                expected = probabilities(logits, .8)
                np.testing.assert_allclose(vectors[2], expected, rtol=2e-6, atol=1e-8)
                selected = event["token_id"]
                self.assertAlmostEqual(event["selected_probabilities"]["p_no_ad"], vectors[2][selected])
                self.assertAlmostEqual(event["no_ad_comparison"]["tv_q"],
                    float(np.abs((vectors[0]+vectors[1])/2 - vectors[2]).sum()/2))
            with np.load(root / "compared/checkpoint-000.npz", allow_pickle=False) as checkpoint:
                np.testing.assert_array_equal(checkpoint["p_no_ad"], captured[0][2])
            manifest = json.loads((root / "compared/manifest.json").read_text())
            self.assertEqual(manifest["no_ad_reference"]["token_ids"], reference_ids)

    def test_top_table_includes_no_ad_candidates_without_renormalization(self):
        a = np.array([.8, .1, .1])
        b = np.array([.1, .1, .8])
        p_no_ad = np.array([.1, .8, .1])
        rows = top_tokens(TestTokenizer(), a, b, (a+b)/2, 1, p_no_ad)
        self.assertEqual({row["token_id"] for row in rows}, {0, 1, 2})
        middle = next(row for row in rows if row["token_id"] == 1)
        self.assertAlmostEqual(middle["p_no_ad"], .8)
        self.assertAlmostEqual(middle["delta_q_no_ad"], -.7)


if __name__ == "__main__":
    unittest.main()
