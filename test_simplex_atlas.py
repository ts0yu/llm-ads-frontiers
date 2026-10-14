"""Simplex atlas and n-context decoder. Random tiny Qwen weights: not quality evidence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.live import LiveRequest, default_inputs, live_messages
from token_auction_lab.prompts import messages, scenarios
from token_auction_lab.prompts_n import brands, messages_n, scenarios_n
from token_auction_lab.simplex_atlas import (
    clip,
    measure,
    partition,
    random_interior_point,
    simplex,
)

try:
    import copy

    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from token_auction_lab.decoder import SharedPrefixDecoder
    from token_auction_lab.multi_decoder import MultiPrefixDecoder
    from token_auction_lab.simplex_atlas import (
        build_atlas,
        cell_payments,
        direct_generate,
        two_bidder_tapes,
        verify,
        write_atlases,
    )
except ImportError:
    torch = None

EOS = 31


class TestTokenizer:
    eos_token_id = EOS

    def decode(self, ids, **kwargs):
        return " ".join(f"t{i}" for i in ids if not (kwargs.get("skip_special_tokens")
                                                       and i == EOS))


class PromptTests(unittest.TestCase):
    def test_two_advertiser_messages_match_demo_prompt_builders(self):
        for scenario in scenarios().values():
            for side in (0, 1):
                self.assertEqual(messages_n(scenario, side), messages(scenario, side))
        request = LiveRequest.parse(default_inputs())
        live = {"style": "live", "prompt": request.prompt, "instructions": request.instructions,
                "advertisers": [{"brand": a.brand, "facts": a.facts, "objective": a.objective}
                                for a in request.advertisers]}
        for side in (0, 1):
            self.assertEqual(messages_n(live, side), live_messages(request, side))

    def test_each_context_gets_shared_task_and_only_its_own_brief(self):
        for name, scenario in scenarios_n().items():
            self.assertEqual(len(scenario["advertisers"]), 3, name)
            rendered = [messages_n(scenario, i) for i in range(3)]
            self.assertEqual(len({m[-1]["content"] for m in rendered}), 1)
            self.assertEqual(len({m[0]["content"] for m in rendered}), 3)
            for i, ad in enumerate(scenario["advertisers"]):
                system = rendered[i][0]["content"]
                self.assertIn(ad["brand"], system)
                facts = ad["facts"] if isinstance(ad["facts"], list) else [ad["facts"]]
                for fact in facts:
                    self.assertIn(fact, system)
                for other in brands(scenario):
                    if other != ad["brand"]:
                        self.assertNotIn(other, system)
                self.assertNotIn("bid", system.lower())
                self.assertNotIn("bid", rendered[i][1]["content"].lower())
        live = scenarios_n()["fort-mason-three"]
        self.assertEqual(live["prompt"], default_inputs()["prompt"])
        self.assertEqual(brands(live)[:2], [a["brand"] for a in default_inputs()["advertisers"]])
        with self.assertRaises(ValueError):
            messages_n(live, 3)


class GeometryTests(unittest.TestCase):
    def test_partition_conserves_area_and_respects_winner_regions(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            exponentials = rng.exponential(size=3)
            pieces, slivers = partition(simplex(3), exponentials)
            self.assertEqual(sorted(w for w, _ in pieces), [0, 1, 2])
            self.assertAlmostEqual(sum(measure(p) for _, p in pieces) + sum(slivers), 0.5,
                                   places=14)
            for winner, piece in pieces:
                point = random_interior_point(piece, rng)
                self.assertEqual(int(np.argmax(point / exponentials)), winner)
                apex = exponentials / exponentials.sum()
                self.assertTrue(any(np.abs(v - apex).max() < 1e-12 for v in piece))

    def test_clip_touching_edge_gives_empty_piece(self):
        triangle = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self.assertLess(len(clip(triangle, np.array([0.0, 0, -1.0]), 1e-14)), 3)
        np.testing.assert_array_equal(clip(triangle, np.array([0.0, 0, 1.0]), 1e-14), triangle)

    def test_interval_partition_boundary_is_contest_uniform(self):
        for u in (1e-9, 0.25, 0.5, 0.9999):
            tape = RandomTape(0.5, 0.1, 0.2, u)
            from token_auction_lab.auction_n import RaceTape

            exponentials = RaceTape.from_two_bidder(tape).exponentials
            pieces, _ = partition(simplex(2), exponentials, min_area=0)
            by_winner = {w: p for w, p in pieces}
            self.assertAlmostEqual(by_winner[0][:, 0].min(), u, delta=1e-12)
            self.assertAlmostEqual(by_winner[1][:, 0].max(), u, delta=1e-12)


def _fork_shared(decoder):
    clone = object.__new__(SharedPrefixDecoder)
    clone.model, clone.device = decoder.model, decoder.device
    clone.generated = list(decoder.generated)
    clone.contexts = [copy.copy(c) for c in decoder.contexts]
    for c, original in zip(clone.contexts, decoder.contexts):
        c.token_ids, c.cache = list(original.token_ids), copy.deepcopy(original.cache)
    return clone


def interval_search(model, prefixes, *, seed, temperature, max_tokens, eos_ids):
    """MOONSHOTS.md section 1, written against the reference two-bidder auction."""
    rng = np.random.default_rng(seed)
    tapes = [RandomTape.draw(rng) for _ in range(max_tokens)]
    leaves, stack = [], [(0.0, 1.0, SharedPrefixDecoder(model, prefixes))]
    while stack:
        lo, hi, decoder = stack.pop()
        while True:
            step = len(decoder.generated)
            if step == max_tokens:
                leaves.append((lo, hi, list(decoder.generated)))
                break
            auction, tape = TwoBidderAuction(*decoder.distributions(temperature)), tapes[step]
            mid = 0.5 * (lo + hi)
            allocation = auction.resolve(mid, 1 - mid, tape)
            token = allocation.token_id
            if allocation.branch != "shared" and lo < tape.contest < hi:
                # (lo, u] keeps B's candidate; (u, hi] takes A's candidate.
                a_token = allocation.candidate_a
                if a_token in eos_ids:
                    leaves.append((tape.contest, hi, list(decoder.generated) + [a_token]))
                else:
                    other = _fork_shared(decoder)
                    other.advance(a_token)
                    stack.append((tape.contest, hi, other))
                hi, token = tape.contest, allocation.candidate_b
            if token in eos_ids:
                leaves.append((lo, hi, list(decoder.generated) + [token]))
                break
            decoder.advance(token)
    return sorted(leaves)


@unittest.skipIf(torch is None, "Install the inference extra to run model-loop tests.")
class AtlasTests(unittest.TestCase):
    PREFIXES = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]

    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256, eos_token_id=EOS,
        )
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()
        self.tokenizer = TestTokenizer()

    def atlas(self, **kwargs):
        options = dict(seed=7, temperature=0.1, max_tokens=12, eos_ids={EOS})
        options.update(kwargs)
        prefixes = options.pop("prefixes", self.PREFIXES)
        return build_atlas(self.model, self.tokenizer, prefixes, **options)

    def direct(self, bids, *, max_tokens=12, eos_ids=frozenset({EOS}), prefixes=None,
               temperature=0.1, **kw):
        return direct_generate(self.model, prefixes or self.PREFIXES, bids, seed=7,
                               temperature=temperature, max_tokens=max_tokens,
                               eos_ids=set(eos_ids),
                               **kw)

    def test_multi_decoder_with_two_prefixes_matches_shared_prefix_decoder(self):
        prefixes = [[1, 2, 3], [4, 5, 6, 7, 8]]
        reference, multi = SharedPrefixDecoder(self.model, prefixes), MultiPrefixDecoder(
            self.model, prefixes)
        for token in [9, 1, 12, 0, 30]:
            for a, b in zip(reference.contexts, multi.contexts):
                self.assertTrue(torch.equal(a.logits, b.logits))
            for a, b in zip(reference.distributions(0.8), multi.distributions(0.8)):
                np.testing.assert_array_equal(a, b)
            reference.advance(token)
            multi.advance(token)
        self.assertEqual(multi.generated, reference.generated)
        self.assertLess(max(multi.check_cache(atol=2e-5, rtol=1e-4)), 2e-5)

    def test_multi_decoder_fork_is_independent_and_cache_correct(self):
        decoder = MultiPrefixDecoder(self.model, self.PREFIXES)
        self.assertEqual(len({id(c.cache) for c in decoder.contexts}), 3)
        decoder.advance(5)
        fork = decoder.fork()
        fork.advance(11)
        decoder.advance(12)
        straight = MultiPrefixDecoder(self.model, self.PREFIXES)
        straight.advance(5)
        straight.advance(11)
        self.assertEqual((decoder.generated, fork.generated), ([5, 12], [5, 11]))
        for a, b in zip(fork.contexts, straight.contexts):
            self.assertTrue(torch.equal(a.logits, b.logits))
            self.assertIsNot(a.cache, b.cache)
        for d in (decoder, fork):
            self.assertLess(max(d.check_cache(atol=2e-5, rtol=1e-4)), 2e-5)
            for context in d.contexts:
                self.assertEqual(context.token_ids[len(context.prefix):], d.generated)
                self.assertEqual(context.cache.get_seq_length(), len(context.token_ids))
        with self.assertRaisesRegex(RuntimeError, "different generated"):
            fork.contexts[1].token_ids[-1] = 0
            fork.assert_shared_suffix()

    def test_areas_partition_the_simplex_and_every_cell_matches_direct_runs(self):
        atlas = self.atlas()
        stats = atlas["stats"]
        self.assertEqual(atlas["schema"], "simplex-atlas-v1")
        self.assertEqual(atlas["mode"], "synthetic_model_test")
        self.assertGreater(stats["cells"], 8)
        self.assertFalse(stats["truncated"])
        self.assertAlmostEqual(stats["area_total"] + stats["dropped_slivers"]["area"], 0.5,
                               delta=1e-9)
        self.assertAlmostEqual(sum(c["area"] for c in atlas["cells"]), 0.5, delta=1e-9)
        self.assertEqual(stats["cells_after_step"][-1], stats["cells"])
        rng = np.random.default_rng(0)
        for cell in atlas["cells"]:
            if cell["stop_reason"] == "eos":
                self.assertEqual(cell["token_ids"][-1], EOS)
            else:
                self.assertEqual((len(cell["token_ids"]), cell["stop_reason"]), (12, "length"))
            for point in (np.array(cell["centroid"]),
                          random_interior_point(np.array(cell["polygon"]), rng)):
                run = self.direct(point)
                self.assertEqual(run["token_ids"], cell["token_ids"], cell["id"])
                np.testing.assert_allclose(run["payments"], cell_payments(cell["steps"], point),
                                           rtol=1e-12, atol=1e-15)
                for pay, bid in zip(run["payments"], point):
                    self.assertLessEqual(pay, 12 * bid)
            self.assertTrue(cell["cache_check"]["max_absolute_errors"])
        json.dumps(atlas, allow_nan=False)
        records = verify(atlas, self.model, self.PREFIXES, sample=4, eos_ids={EOS})
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r["matches_direct_run"] for r in records))

    def test_corner_cells_equal_single_bidder_runs(self):
        atlas = self.atlas()
        for i in range(3):
            corner = np.eye(3)[i]
            owners = [c for c in atlas["cells"]
                      if any(np.abs(np.array(v) - corner).max() < 1e-12 for v in c["polygon"])]
            self.assertEqual(len(owners), 1)
            cell = owners[0]
            run = self.direct(corner)
            self.assertEqual(run["token_ids"], cell["token_ids"])
            self.assertEqual(run["payments"], [0.0, 0.0, 0.0])
            for record in cell["steps"]:
                self.assertIn(record["winner"], (None, i))
            near = 0.999 * corner + 0.0005 * (1 - corner)
            self.assertEqual(self.direct(near)["token_ids"], cell["token_ids"])

    def test_three_bidder_cells_can_exceed_the_two_bidder_bound(self):
        atlas = self.atlas(temperature=0.03, max_tokens=12)
        self.assertGreater(atlas["stats"]["cells"], 1 + 12)
        self.assertLessEqual(atlas["stats"]["distinct_outputs"], atlas["stats"]["cells"])

    def test_eos_stops_cells_and_matches_direct_runs(self):
        plain = self.atlas()
        stop = plain["cells"][0]["token_ids"][5]
        atlas = self.atlas(eos_ids={stop})
        eos_cells = [c for c in atlas["cells"] if c["stop_reason"] == "eos"]
        self.assertTrue(eos_cells)
        for cell in atlas["cells"]:
            run = self.direct(np.array(cell["centroid"]), eos_ids={stop})
            self.assertEqual(run["token_ids"], cell["token_ids"])
            self.assertEqual(run["stop_reason"], cell["stop_reason"])
            if cell["stop_reason"] == "eos":
                self.assertEqual(cell["token_ids"][-1], stop)
                self.assertNotIn(stop, cell["token_ids"][:-1])

    def test_max_cells_truncates_visibly_and_never_silently(self):
        full = self.atlas()
        atlas = self.atlas(max_cells=4)
        stats = atlas["stats"]
        self.assertLessEqual(stats["cells"], 4)
        self.assertLess(stats["cells"], full["stats"]["cells"])
        self.assertTrue(stats["truncated"])
        truncated = [c for c in atlas["cells"] if c["stop_reason"] == "truncated"]
        self.assertEqual(len(truncated), stats["truncated_cells"]["max_cells"])
        self.assertTrue(all(c["truncation"]["reason"] == "max_cells" for c in truncated))
        self.assertAlmostEqual(stats["area_total"], 0.5, delta=1e-9)
        records = verify(atlas, self.model, self.PREFIXES, sample=0, eos_ids={EOS},
                         cell_ids=[c["id"] for c in atlas["cells"]])
        self.assertTrue(all(r["matches_direct_run"] for r in records))
        self.assertTrue(any(r["truncated_prefix_only"] for r in records))
        budget = self.atlas(time_budget_seconds=0.0)
        self.assertEqual(budget["stats"]["truncated_cells"]["time_budget"], 1)
        self.assertEqual(budget["cells"][0]["truncation"], {"reason": "time_budget", "step": 0})

    def test_two_bidder_interval_atlas_matches_reference_interval_search(self):
        prefixes = self.PREFIXES[:2]
        atlas = self.atlas(prefixes=prefixes, tape_source="two-bidder-random-tape",
                           temperature=0.02, max_tokens=24)
        self.assertGreater(atlas["stats"]["splits"], 5)
        self.assertTrue(any(c["stop_reason"] == "eos" for c in atlas["cells"]))
        self.assertLessEqual(atlas["stats"]["cells"], 1 + 24)  # MOONSHOTS.md bound.
        self.assertAlmostEqual(atlas["stats"]["area_total"], 1.0, delta=1e-12)
        rng = np.random.default_rng(7)
        contests = [RandomTape.draw(rng).contest for _ in range(24)]
        for split in atlas["splits"]:
            self.assertAlmostEqual(split["apex"][0], contests[split["step"]], delta=1e-14)
        leaves = sorted(
            (min(v[0] for v in c["polygon"]), max(v[0] for v in c["polygon"]), c["token_ids"])
            for c in atlas["cells"]
        )
        reference = interval_search(self.model, prefixes, seed=7, temperature=0.02,
                                    max_tokens=24, eos_ids={EOS})
        self.assertEqual(len(leaves), len(reference))
        for (lo, hi, tokens), (rlo, rhi, rtokens) in zip(leaves, reference):
            self.assertAlmostEqual(lo, rlo, delta=1e-12)
            self.assertAlmostEqual(hi, rhi, delta=1e-12)
            self.assertEqual(tokens, rtokens)
            if 0.0 < lo:
                self.assertTrue(any(abs(lo - u) < 1e-12 for u in contests))
        # Each leaf also equals a direct run of the race at its midpoint.
        for lo, hi, tokens in leaves:
            mid = 0.5 * (lo + hi)
            run = self.direct((mid, 1 - mid), prefixes=prefixes, max_tokens=24,
                              temperature=0.02, tape_source=two_bidder_tapes)
            self.assertEqual(run["token_ids"], tokens)

    def test_write_atlases_writes_seed_files_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            index = write_atlases(
                self.model, self.tokenizer,
                [(f"prompt {i}", p) for i, p in enumerate(self.PREFIXES)],
                seeds=[7, 8], temperature=0.1, max_tokens=6, output=out,
                bidders=["Lantern House", "Tideglass Hotel", "Saltmarsh Inn"],
                eos_ids={EOS}, verify_cells=2, mode="synthetic_model_test",
                metadata={"model_id": "random-tiny-qwen3-test"},
            )
            self.assertEqual([a["seed"] for a in index["atlases"]], [7, 8])
            self.assertTrue(all(a["verification_ok"] for a in index["atlases"]))
            self.assertEqual(json.loads((out / "index.json").read_text()), index)
            for seed in (7, 8):
                atlas = json.loads((out / f"atlas-seed{seed}.json").read_text())
                self.assertEqual(atlas["seed"], seed)
                self.assertEqual(atlas["manifest"]["seed"], seed)
                self.assertEqual(atlas["manifest"]["prompts"][2]["token_ids"], [8, 9])
                self.assertIn("simplex_atlas.py", " ".join(atlas["manifest"]["source_sha256"]))
                self.assertEqual(len(atlas["verification"]), 2)
                cell = atlas["cells"][0]
                self.assertEqual(set(cell["outcomes"]), {
                    "mentions:Lantern House", "mentions:Tideglass Hotel",
                    "mentions:Saltmarsh Inn"})
                self.assertTrue({"step", "branch", "winner", "shared_mass"} <= set(
                    cell["steps"][0]))


class ChatTokenizer(TestTokenizer):
    """Deterministic stand-in: a short, prompt-dependent ID prefix per rendered text."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"

    def encode(self, text, add_special_tokens):
        import hashlib

        return [b % 30 for b in hashlib.sha256(text.encode()).digest()[:6]]


@unittest.skipIf(torch is None, "Install the inference extra to run model-loop tests.")
class CliTests(unittest.TestCase):
    setUp = AtlasTests.setUp

    def test_cli_writes_live_labelled_atlases_with_full_manifest(self):
        from unittest.mock import patch

        from token_auction_lab import simplex_atlas

        loaded = (ChatTokenizer(), self.model, "0" * 40, "cpu", "float32", 0.0)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "token_auction_lab.cli.load_model", return_value=loaded
        ), patch("builtins.print"):
            simplex_atlas.main([
                "--scenario", "three-hotels", "--seeds", "7", "8", "--max-tokens", "5",
                "--temperature", "0.1", "--revision", "0" * 40, "--local-files-only",
                "--output", tmp, "--verify", "2", "--max-cells", "50",
                "--time-budget-seconds", "60",
            ])
            index = json.loads((Path(tmp) / "index.json").read_text())
            self.assertEqual(index["mode"], "live_model")
            self.assertEqual(index["bidders"],
                             ["Lantern House", "Tideglass Hotel", "Saltmarsh Inn"])
            self.assertTrue(all(a["verification_ok"] for a in index["atlases"]))
            atlas = json.loads((Path(tmp) / "atlas-seed8.json").read_text())
            manifest = atlas["manifest"]
            self.assertEqual(atlas["label"], "LIVE MODEL OUTPUT")
            for key in ("model_revision", "tokenizer_revision", "dependencies", "prompts",
                        "temperature", "seed", "eos_token_ids", "scenario_definition"):
                self.assertIn(key, manifest)
            self.assertEqual(len({p["sha256"] for p in manifest["prompts"]}), 3)
            for i, prompt in enumerate(manifest["prompts"]):
                self.assertIn(index["bidders"][i], prompt["text"])
                for j, other in enumerate(index["bidders"]):
                    if j != i:
                        self.assertNotIn(other, prompt["text"])
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            simplex_atlas.main(["--local-files-only", "--revision", "main"])


if __name__ == "__main__":
    unittest.main()
