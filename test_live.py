"""Live API contracts, input isolation, stream fidelity, and concurrency guards."""

from copy import deepcopy
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.live import (
    BusyError, LiveRequest, LiveService, StaleRunError, default_inputs, live_messages,
)
from token_auction_lab.walkthrough import Handler


class HeldEngine:
    """Synthetic fixture: hold one worker until the test permits completion."""
    def __init__(self, fail=False, reference=False, fail_no_ad=False):
        self.release = threading.Event()
        self.started = threading.Event()
        self.fail = fail
        self.reference = reference
        self.fail_no_ad = fail_no_ad
        self.requests = []

    def load(self):
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("Fixture was not released")

    def decode(self, token):
        return ["shuttle", "weekend", "courtyard"][token]

    def run(self, request, output, emit, retain, *, should_stop=None):
        self.requests.append(request)
        if should_stop and should_stop():
            return {"text": "", "tokens": 0, "stop_reason": "user_stop", "mode": "synthetic_fixture"}
        p_a, p_b = np.array([.6, .3, .1]), np.array([.1, .3, .6])
        tape = RandomTape(.9, .2, .7, .4)
        allocation = TwoBidderAuction(p_a, p_b).resolve(
            request.advertisers[0].bid, request.advertisers[1].bid, tape
        )
        event = {**asdict(allocation), "type": "token", "step": 0, "tape": asdict(tape),
            "bid_a": request.advertisers[0].bid, "bid_b": request.advertisers[1].bid}
        emit({"type": "round_start", "manifest": {"mode": "synthetic_fixture"}})
        retain(0, [p_a, p_b] + ([np.array([.2, .7, .1])] if self.reference else []))
        emit(event)
        emit({"type": "validation_start"})
        if self.fail:
            raise AssertionError("deliberately failed cache check")
        if self.reference:
            emit({"type": "round_end", "mode": "synthetic_fixture", "text": "ad result"})
            emit({"type": "no_ad_event", "event": {"type": "round_start", "manifest": {
                "mode": "synthetic_fixture", "prompts": [{"text": "user prompt only"}]}}})
            emit({"type": "no_ad_event", "event": {"type": "token", "step": 0,
                "token_id": 1, "decoded_text_so_far": "weekend", "payment_a": 0, "payment_b": 0}})
            if self.fail_no_ad:
                raise AssertionError("deliberately failed no-ad cache check")
            emit({"type": "no_ad_event", "event": {"type": "round_end",
                "text": "weekend", "mode": "synthetic_fixture", "payment_a": 0, "payment_b": 0}})
        return {"text": self.decode(allocation.token_id), "mode": "synthetic_fixture"}


class LiveContractTests(unittest.TestCase):
    def test_every_editable_prompt_field_is_used_but_bids_are_not(self):
        payload = default_inputs()
        original = LiveRequest.parse(payload)
        payload["advertisers"][0]["bid"] = 17
        payload["advertisers"][1]["bid"] = .0123
        changed = LiveRequest.parse(payload)
        for side in (0, 1):
            self.assertEqual(live_messages(original, side), live_messages(changed, side))
        for side in (0, 1):
            system = live_messages(original, side)[0]["content"]
            self.assertIn(original.advertisers[side].brand, system)
            self.assertIn(original.advertisers[side].facts, system)
            self.assertIn(original.advertisers[side].objective, system)
            self.assertIn(original.instructions, system)
            self.assertEqual(live_messages(original, side)[1]["content"], original.prompt)
        edited = deepcopy(payload)
        edited["advertisers"][0]["objective"] = "Write about the bicycles first."
        request = LiveRequest.parse(edited)
        self.assertNotEqual(live_messages(request, 0), live_messages(changed, 0))
        self.assertEqual(live_messages(request, 1), live_messages(changed, 1))
        edited["advertisers"][0]["facts"] = "New explicit bicycle fact."
        request = LiveRequest.parse(edited)
        self.assertIn("New explicit bicycle fact.", live_messages(request, 0)[0]["content"])
        self.assertEqual(live_messages(request, 1), live_messages(changed, 1))

    def test_private_brand_facts_and_objective_edits_do_not_leak_to_rival(self):
        payload = default_inputs()
        original = LiveRequest.parse(payload)
        for side in (0, 1):
            for field in ("brand", "facts", "objective"):
                with self.subTest(side=side, field=field):
                    edited = deepcopy(payload)
                    sentinel = f"PRIVATE_{side}_{field}_SENTINEL"
                    edited["advertisers"][side][field] = sentinel
                    request = LiveRequest.parse(edited)
                    self.assertIn(sentinel, live_messages(request, side)[0]["content"])
                    self.assertEqual(live_messages(request, 1 - side),
                        live_messages(original, 1 - side))

    def test_shared_task_and_instructions_edits_reach_both_contexts(self):
        payload = default_inputs()
        original = LiveRequest.parse(payload)
        for field in ("prompt", "instructions"):
            edited = deepcopy(payload)
            edited[field] = "SHARED_EDIT_SENTINEL"
            request = LiveRequest.parse(edited)
            for side in (0, 1):
                rendered = live_messages(request, side)
                self.assertNotEqual(rendered, live_messages(original, side))
                self.assertTrue(any("SHARED_EDIT_SENTINEL" in m["content"] for m in rendered))

    def test_validation_and_separate_zero_bid_baseline(self):
        invalid = [
            {"advertisers": []}, {"seed": 1.2}, {"max_tokens": 4097},
            {"max_tokens": 0}, {"max_tokens": 1.5}, {"max_tokens": True},
            {"prompt": ""}, {"temperature": float("nan")}, {"seed": True},
        ]
        for change in invalid:
            with self.assertRaises(ValueError):
                LiveRequest.parse({**default_inputs(), **change})
        self.assertEqual(default_inputs()["max_tokens"], 4096)
        self.assertEqual(LiveRequest.parse({**default_inputs(), "max_tokens": 97}).max_tokens, 97)
        payload = default_inputs()
        payload["advertisers"][0]["bid"] = -1
        with self.assertRaises(ValueError):
            LiveRequest.parse(payload)
        for ad in payload["advertisers"]:
            ad["bid"] = 0
        request = LiveRequest.parse(payload)
        self.assertTrue(request.baseline)
        baseline = live_messages(request, None)
        self.assertEqual(baseline, [{"role": "user", "content": request.prompt}])
        self.assertNotIn(request.advertisers[0].facts, baseline[0]["content"])

    def test_continuation_inputs_validate_and_never_enter_prompts(self):
        defaults = default_inputs()
        self.assertEqual(defaults["continuation_tokens"], 3)
        self.assertEqual(defaults["continuation_threshold"], .98)
        enabled = LiveRequest.parse(defaults)
        disabled = LiveRequest.parse({**defaults, "continuation_tokens": 0, "continuation_threshold": .75})
        for side in (0, 1, None):
            self.assertEqual(live_messages(enabled, side), live_messages(disabled, side))
        for changes in [{"continuation_tokens": -1}, {"continuation_tokens": 33},
                {"continuation_tokens": True}, {"continuation_threshold": .5},
                {"continuation_threshold": float("nan")}]:
            with self.assertRaises(ValueError):
                LiveRequest.parse({**defaults, **changes})
        defaults.pop("continuation_tokens")
        defaults.pop("continuation_threshold")
        self.assertEqual(LiveRequest.parse(defaults).continuation_tokens, 0)

    def test_carried_inspector_distinguishes_actual_output_from_hypothetical_mixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            service = LiveService(Path(tmp), engine)
            a, b, no_ad = np.array([.99, .01, 0]), np.array([0, .01, .99]), np.array([.2, .7, .1])
            event = {"token_id": 0, "branch": "A", "bid_a": 1, "bid_b": 1,
                "selection_rule": "winner_continuation", "continuation": {"source_step": 0},
                "tape": None, "payment_a": 0, "payment_b": 0}
            service.current = {"run_id": "synthetic-carry", "events": [event], "vectors": {0: [a, b, no_ad]}}
            result = service.inspect("synthetic-carry", 0)
            rows = {r["token_id"]: r for r in result["rows"]}
            self.assertEqual(result["selection_rule"], "winner_continuation")
            self.assertEqual(rows[0]["p_output"], 1)
            self.assertEqual(rows[2]["p_output"], 0)
            self.assertAlmostEqual(rows[0]["q"], .495)
            self.assertAlmostEqual(rows[0]["delta_output_no_ad"], .8)
            self.assertAlmostEqual(result["no_ad_comparison"]["tv_output"], .8)
            self.assertFalse(result["counterfactual"])
            self.assertEqual(result["outcome"]["payment_a"], 0)
            with self.assertRaisesRegex(ValueError, "previous win"):
                service.inspect("synthetic-carry", 0, a=2)

    def test_no_ad_prompt_is_independent_of_every_ad_input(self):
        payload = default_inputs()
        original = live_messages(LiveRequest.parse(payload), None)
        payload["instructions"] = "AD INSTRUCTIONS SENTINEL"
        for i, ad in enumerate(payload["advertisers"]):
            ad.update(brand=f"BRAND {i}", facts="FACT SENTINEL", objective="OBJECTIVE SENTINEL", bid=99)
        self.assertEqual(live_messages(LiveRequest.parse(payload), None), original)
        self.assertEqual(default_inputs()["advertisers"][0]["brand"], "Bayview Hotel")
        self.assertIn("Fort Mason", default_inputs()["prompt"])

    def test_no_ad_probabilities_and_delta_stay_fixed_under_preview_bids(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(reference=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            run_id = service.start(default_inputs())
            service.worker.join(2)
            original = service.inspect(run_id, 0, token_id=0)
            preview = service.inspect(run_id, 0, token_id=0, a=3, b=1)
            self.assertAlmostEqual(original["rows"][0]["p_no_ad"], .2)
            self.assertAlmostEqual(original["rows"][0]["delta_q_no_ad"], .15)
            self.assertAlmostEqual(original["no_ad_comparison"]["tv_q"], .4)
            self.assertAlmostEqual(preview["rows"][0]["q"], .475)
            self.assertEqual(preview["rows"][0]["p_no_ad"], original["rows"][0]["p_no_ad"])
            self.assertEqual(preview["rows"][0]["delta_q_no_ad"], original["rows"][0]["delta_q_no_ad"])
            self.assertEqual(preview["no_ad_comparison"]["tv_q"], original["no_ad_comparison"]["tv_q"])
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "complete")
            self.assertEqual(snapshot["no_ad"]["result"]["text"], "weekend")
            self.assertEqual(snapshot["no_ad"]["events"][0]["payment_a"], 0)

    def test_no_ad_failure_preserves_both_outputs_and_original_ad_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(reference=True, fail_no_ad=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            service.start(default_inputs())
            service.worker.join(2)
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertEqual(snapshot["result"]["text"], "ad result")
            self.assertEqual(len(snapshot["events"]), 1)
            self.assertEqual(snapshot["no_ad"]["events"][0]["decoded_text_so_far"], "weekend")
            self.assertIsNone(snapshot["no_ad"]["result"])
            self.assertIn("no-ad cache", snapshot["error"])

    def test_one_worker_immutable_snapshot_and_replaced_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            service = LiveService(Path(tmp), engine)
            payload = default_inputs()
            first = service.start(payload)
            self.assertTrue(engine.started.wait(1))
            payload["advertisers"][0]["bid"] = 50
            payload["prompt"] = "Edited after the request started"
            try:
                with self.assertRaises(BusyError):
                    service.start(payload)
                self.assertEqual(service.snapshot()["inputs"]["advertisers"][0]["bid"], 1)
            finally:
                engine.release.set()
                service.worker.join(2)
            self.assertEqual(service.snapshot()["status"], "complete")
            self.assertEqual(engine.requests[0].advertisers[0].bid, 1)
            second = service.start(payload)
            service.worker.join(2)
            self.assertNotEqual(first, second)
            with self.assertRaises(StaleRunError):
                service.inspect(first, 0)
            with self.assertRaises(StaleRunError):
                service.snapshot(first)

    def test_inspection_uses_full_vectors_and_does_not_change_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            run_id = service.start(default_inputs())
            service.worker.join(2)
            before = service.snapshot()
            preview = service.inspect(run_id, 0, a=.6, b=1, limit=1)
            self.assertEqual(preview["outcome"]["branch"], "B")
            self.assertAlmostEqual(preview["outcome"]["payment_b"], .9)
            self.assertEqual(preview["total"], 3)
            exact = service.inspect(run_id, 0, token_id=1)
            self.assertAlmostEqual(exact["rows"][0]["p_a"], .3)
            self.assertAlmostEqual(exact["rows"][0]["p_b"], .3)
            self.assertEqual(before, service.snapshot())
            self.assertEqual(service.snapshot(after=1)["events"], [])

    def test_failed_validation_preserves_tokens_and_unlocks_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(fail=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            failed = service.start(default_inputs())
            service.worker.join(2)
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertEqual(len(snapshot["events"]), 1)
            self.assertIsNone(snapshot["result"])
            self.assertIn("deliberately failed", snapshot["error"])
            self.assertTrue(service.file(failed, "error.json").is_file())
            engine.fail = False
            service.start(default_inputs())
            service.worker.join(2)
            self.assertEqual(service.snapshot()["status"], "complete")

    def test_stop_keeps_worker_locked_and_resets_for_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            service = LiveService(Path(tmp), engine)
            first = service.start(default_inputs())
            self.assertTrue(engine.started.wait(1))
            try:
                with self.assertRaises(StaleRunError):
                    service.stop("another-run")
                self.assertFalse(service.snapshot()["stop_requested"])
                self.assertTrue(service.stop(first)["stop_requested"])
                self.assertTrue(service.stop(first)["stop_requested"])
                self.assertEqual(service.snapshot()["status"], "loading")
                with self.assertRaises(BusyError):
                    service.start(default_inputs())
            finally:
                engine.release.set()
                service.worker.join(2)
            stopped = service.snapshot()
            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["result"]["stop_reason"], "user_stop")
            second = service.start(default_inputs())
            service.worker.join(2)
            self.assertEqual(service.snapshot()["status"], "complete")
            self.assertFalse(service.snapshot()["stop_requested"])
            with self.assertRaises(StaleRunError):
                service.stop(first)
            self.assertFalse(service.stop(second)["stop_requested"])

    def test_http_rejects_cross_origin_and_busy_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.live = LiveService(Path(tmp), engine)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/api/live"

            def post(origin=None, route="/run", payload=None):
                headers = {"Content-Type": "application/json"}
                if origin:
                    headers["Origin"] = origin
                return urlopen(Request(url + route,
                    data=json.dumps(default_inputs() if payload is None else payload).encode(),
                    headers=headers), timeout=3)

            try:
                with self.assertRaises(HTTPError) as rejected:
                    post("https://example.com")
                self.assertEqual(rejected.exception.code, 403)
                with post() as response:
                    self.assertEqual(response.status, 202)
                    run_id = json.load(response)["run_id"]
                with self.assertRaises(HTTPError) as busy:
                    post()
                self.assertEqual(busy.exception.code, 409)
                with self.assertRaises(HTTPError) as rejected:
                    post("https://example.com", "/stop", {"run_id": run_id})
                self.assertEqual(rejected.exception.code, 403)
                with self.assertRaises(HTTPError) as stale:
                    post(route="/stop", payload={"run_id": "other-run"})
                self.assertEqual(stale.exception.code, 409)
                with self.assertRaises(HTTPError) as invalid:
                    post(route="/stop", payload=[])
                self.assertEqual(invalid.exception.code, 400)
                with post(route="/stop", payload={"run_id": run_id}) as response:
                    self.assertTrue(json.load(response)["stop_requested"])
            finally:
                engine.release.set()
                if server.live.worker:
                    server.live.worker.join(2)
                server.shutdown()
                server.server_close()
                thread.join(2)


if __name__ == "__main__":
    unittest.main()
