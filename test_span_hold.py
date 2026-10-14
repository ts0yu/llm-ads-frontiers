"""Span hold (token_auction_lab/span_hold.py): rule, toy two-sponsor world, exact atlas.

The toy world is a hand-written pair of next-token policies, not a language model;
the atlas tests use tiny random Qwen3 weights (synthetic_model_test). They check
the mechanism and its bookkeeping, not text quality or advertiser value.
"""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch  # noqa: F401

    from auction import TwoBidderAuction
    from test_atlas import PROMPTS, tiny_model
    from token_auction_lab import span_hold
    from token_auction_lab.atlas import (
        build_atlas,
        check_partition,
        leaf_for,
        payments,
        verify,
        verify_point,
    )
    from token_auction_lab.span_hold import SpanState, continues_span
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the inference extra to run span-hold tests.")
class SpanRuleTests(unittest.TestCase):
    def test_words_numbers_and_names_continue(self):
        self.assertTrue(continues_span(" Lan", "tern"))
        self.assertTrue(continues_span(" Lantern", " House"))  # capitalised name
        self.assertTrue(continues_span(" Bay Area", "’s"))
        self.assertTrue(continues_span(" Bay Area’s", " Lantern"))
        self.assertTrue(continues_span(" 4", "."))
        self.assertTrue(continues_span(" 4.", "9"))
        self.assertTrue(continues_span(" well", "-known"))

    def test_boundaries_end_a_span(self):
        self.assertFalse(continues_span(" Lantern", " is"))  # lowercase word
        self.assertFalse(continues_span(" the", " Bay"))  # not a name span
        self.assertFalse(continues_span(" Lantern", ","))
        self.assertFalse(continues_span(" Hotel", "."))  # sentence end, not a number
        self.assertFalse(continues_span(" Lantern", "\n"))
        self.assertFalse(continues_span(" Lantern", ""))  # EOS renders as nothing
        self.assertFalse(continues_span("", "tern"))
        self.assertFalse(continues_span(".", "The"))

    def test_state_owner_follows_contest_winner_and_releases(self):
        state = SpanState().after(" Lan", "B")
        self.assertEqual((state.span, state.owner, state.held), (" Lan", "B", 0))
        state = state.after("tern", "hold_B")
        self.assertEqual((state.span, state.owner, state.held), (" Lantern", "B", 1))
        self.assertEqual(state.released().owner, None)
        self.assertEqual(state.released().after(",", "shared"), SpanState(",", None, 0))
        # A contest won mid-span hands the span to the winner.
        self.assertEqual(SpanState(" Lantern").after(" Bay", "A").owner, "A")


# ------------------------------------------------------------------ toy two-sponsor world

PIECES = ["", "Stay", " at", " Lan", "tern", " House", " Bay", "view", " Hotel", ".", " Area"]
EOS = 0
# Each sponsor's next-token policy, keyed by the end of the shared text. A wants
# "Bayview Hotel"; B wants "Lantern House". Each context can finish the other's word,
# and after "Lantern" A prefers " Bay", which is how per-token contests blend names.
POLICY = {
    "A": {"": {1: 1}, "Stay": {2: 1}, " at": {6: 0.9, 9: 0.1}, " Bay": {7: 1},
          "view": {8: 1}, " Lan": {4: 1}, "tern": {6: 0.7, 5: 0.3}},
    "B": {"": {1: 1}, "Stay": {2: 1}, " at": {3: 0.9, 9: 0.1}, " Lan": {4: 1},
          "tern": {5: 1}, " Bay": {10: 0.7, 7: 0.3}, "view": {8: 0.6, 9: 0.4}},
}
EXACT = {"Stay at Bayview Hotel.", "Stay at Lantern House.", "Stay at."}


class ToyTokenizer:
    eos_token_id = EOS

    def decode(self, ids, **kwargs):
        return "".join(PIECES[i] for i in ids)


class ToyDecoder:
    def __init__(self):
        self.generated = []

    def _policy(self, who):
        text = "".join(PIECES[i] for i in self.generated)
        table = POLICY[who]
        key = max((k for k in table if text.endswith(k)), key=len, default=None)
        rule = table[key] if key is not None and text else table[""]
        if text.endswith((".", " Hotel", " House", " Area")):
            rule = {EOS: 1} if text.endswith(".") else {9: 1}
        p = np.zeros(len(PIECES))
        for token, weight in rule.items():
            p[token] = weight
        return p / p.sum()

    def distributions(self, temperature):
        return [self._policy("A"), self._policy("B")]

    def advance(self, token_id):
        self.generated.append(token_id)


def toy_run(bid_a, bid_b, seed, max_tokens=12):
    return span_hold.run(
        ToyDecoder(), ToyTokenizer(), bid_a, bid_b,
        tapes=span_hold.draw_tapes(seed, max_tokens),
        holds=span_hold.hold_uniforms(seed, max_tokens),
        stops={EOS}, temperature=1.0, max_tokens=max_tokens,
    )


def toy_reference(bid_a, bid_b, seed, max_tokens=12):
    """The reference per-token sampler (run_round's loop) on the toy world."""
    decoder, tapes = ToyDecoder(), span_hold.draw_tapes(seed, max_tokens)
    pay_a = pay_b = 0.0
    for step in range(max_tokens):
        step_auction = TwoBidderAuction(*decoder.distributions(1.0))
        allocation = step_auction.resolve(bid_a, bid_b, tapes[step])
        pay_a, pay_b = pay_a + allocation.payment_a, pay_b + allocation.payment_b
        decoder.advance(allocation.token_id)
        if allocation.token_id == EOS:
            break
    return decoder.generated, pay_a, pay_b


@unittest.skipIf(torch is None, "Install the inference extra to run span-hold tests.")
class ToyWorldTests(unittest.TestCase):
    LAMBDAS = [k / 40 for k in range(1, 40)]
    SEEDS = range(40)

    def test_without_holds_it_is_exactly_the_reference_sampler_and_blends(self):
        blends = set()
        with patch.object(span_hold, "MAX_HOLD", 0):
            for seed in self.SEEDS:
                for lam in self.LAMBDAS:
                    result = toy_run(lam, 1 - lam, seed)
                    tokens, pay_a, pay_b = toy_reference(lam, 1 - lam, seed)
                    self.assertEqual(result["token_ids"], tokens)
                    self.assertEqual((result["payment_a"], result["payment_b"]),
                                     (pay_a, pay_b))
                    self.assertFalse(any(s["branch"].startswith("hold") for s in result["steps"]))
                    if result["text"] not in EXACT:
                        blends.add(result["text"])
        # Per-token contests write names neither sponsor wants.
        self.assertIn("Stay at Lantern Bayview Hotel.", blends)
        self.assertIn("Stay at Bay Area.", blends)

    def test_span_hold_writes_only_exact_names_and_charges_only_contests(self):
        held = 0
        for seed in self.SEEDS:
            for lam in self.LAMBDAS:
                result = toy_run(lam, 1 - lam, seed)
                self.assertIn(result["text"], EXACT, (seed, lam))
                reference = TwoBidderAuction([1.0, 0.0], [0.0, 1.0])
                pay_a = pay_b = 0.0
                for step in result["steps"]:
                    if step["branch"].startswith("hold_"):
                        held += 1
                        self.assertEqual(step["span"]["owner"], step["branch"][-1])
                        continue
                    if step["branch"] == "shared":
                        continue
                    tape = span_hold.RandomTape(0.5, 0.5, 0.5, step["u"])
                    charge = reference.resolve(lam, 1 - lam, tape)
                    self.assertEqual(charge.branch, step["branch"])
                    pay_a, pay_b = pay_a + charge.payment_a, pay_b + charge.payment_b
                self.assertEqual((result["payment_a"], result["payment_b"]), (pay_a, pay_b))
        self.assertGreater(held, 0)

    def test_output_depends_on_bids_only_through_contest_uniforms(self):
        for seed in self.SEEDS:
            runs = [toy_run(lam, 1 - lam, seed) for lam in self.LAMBDAS]
            for lam, result in zip(self.LAMBDAS, runs):
                contested = [s for s in result["steps"] if s["branch"] in ("A", "B")]
                for step in contested:
                    self.assertEqual(step["branch"] == "A", step["u"] < lam)
            # Pieces of (0, 1] between consecutive contest uniforms give one output each.
            outputs = {tuple(r["token_ids"]) for r in runs}
            cuts = {s["u"] for r in runs for s in r["steps"] if s["branch"] in ("A", "B")}
            self.assertLessEqual(len(outputs), 1 + len(cuts))


# ------------------------------------------------------------------ exact atlas, tiny model

QWEN_PIECES = ["Stay", " at", " Lan", "tern", " House", " Bay", "view", " Hotel", ".",
               " the", " quiet", " 4", ".", "9", " stars", ",", ""]


class PieceTokenizer:
    eos_token_id = 16

    def decode(self, ids, **kwargs):
        return "".join(QWEN_PIECES[i] for i in ids
                       if not (kwargs.get("skip_special_tokens") and i == 16))


@unittest.skipIf(torch is None, "Install the inference extra to run span-hold tests.")
class SpanHoldAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tokenizer = PieceTokenizer()

    def atlas(self, seed, **kw):
        return build_atlas(self.model, self.tokenizer, PROMPTS, ["Lantern House", "Bayview Hotel"],
                           seed=seed, temperature=1.0, max_tokens=14,
                           mode="synthetic_model_test", **kw)

    def test_span_atlas_is_exact_and_every_leaf_matches_direct_runs(self):
        held = 0
        for seed in range(6):
            atlas = self.atlas(seed, span_hold=True)
            check_partition(atlas)
            self.assertEqual(atlas["manifest"]["span_hold"]["rule"], "span-hold-v1")
            self.assertLessEqual(len(atlas["leaves"]), 1 + 14)
            for leaf in atlas["leaves"]:
                held += sum(s["branch"].startswith("hold_") for s in leaf["steps"])
                for kind, lam in (("midpoint", (leaf["lo"] + leaf["hi"]) / 2),
                                  ("upper", leaf["hi"])):
                    record = verify_point(atlas, leaf, lam, 1 - lam, self.model, self.tokenizer,
                                          workdir=None, kind=kind)
                    self.assertTrue(record["matches_direct_run"], (seed, leaf["id"], record))
                    self.assertEqual(record["sampler"], "span_hold.generate")
            records = verify(atlas, self.model, self.tokenizer, n=2)
            self.assertTrue(all(r["matches_direct_run"] for r in records))
        self.assertGreater(held, 0, "no held steps: the test would not exercise the hold path")

    def test_without_holds_the_span_atlas_equals_the_reference_atlas(self):
        for seed in range(4):
            reference = self.atlas(seed)
            with patch.object(span_hold, "MAX_HOLD", 0):
                spanned = self.atlas(seed, span_hold=True)
            self.assertEqual([(leaf["lo"], leaf["hi"], leaf["token_ids"])
                              for leaf in reference["leaves"]],
                             [(leaf["lo"], leaf["hi"], leaf["token_ids"])
                              for leaf in spanned["leaves"]])
            for leaf in reference["leaves"]:
                lam = (leaf["lo"] + leaf["hi"]) / 2
                twin = leaf_for(spanned, lam)
                self.assertEqual(payments(leaf, lam, 1 - lam), payments(twin, lam, 1 - lam))
            self.assertNotIn("span_hold", reference["manifest"])
            self.assertEqual(reference["manifest"]["sampler"], "auction.TwoBidderAuction")

    def test_held_steps_are_free_and_rejected_without_the_sampler(self):
        atlas = next(a for a in (self.atlas(s, span_hold=True) for s in range(6))
                     if any(st["branch"].startswith("hold_")
                            for leaf in a["leaves"] for st in leaf["steps"]))
        leaf = next(leaf for leaf in atlas["leaves"]
                    if any(s["branch"].startswith("hold_") for s in leaf["steps"]))
        lam = (leaf["lo"] + leaf["hi"]) / 2
        contested = sum(s["branch"] in ("A", "B") for s in leaf["steps"])
        self.assertEqual(payments(leaf, lam, 1 - lam)["contested_steps"], contested)
        del atlas["manifest"]["span_hold"]
        with self.assertRaisesRegex(AssertionError, "Held steps"):
            check_partition(atlas)


if __name__ == "__main__":
    unittest.main()
