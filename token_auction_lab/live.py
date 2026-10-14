"""Single-active-round local inference service; no remote model or paid services."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
from importlib.resources import files
import json
import math
from pathlib import Path
import platform
import threading
import time
import uuid

from token_auction_lab.prompts import ADVERTISER_PROMPT_POLICY
from token_auction_lab.limits import MAX_OUTPUT_TOKENS
from token_auction_lab.continuation import validate_continuation

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ROOT = Path(__file__).resolve().parents[1]
BUSY = {"loading", "prefilling", "generating", "validating", "no-ad-generating", "no-ad-validating"}


class BusyError(Exception):
    pass


class StaleRunError(Exception):
    pass


def text_field(value, name, limit, *, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'0' if empty else '1'}–{limit} characters.")
    return value


def numeric(value, name, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class Bidder:
    brand: str
    facts: str
    objective: str
    bid: float


@dataclass(frozen=True)
class LiveRequest:
    prompt: str
    instructions: str
    advertisers: tuple[Bidder, Bidder]
    seed: int
    temperature: float
    max_tokens: int
    continuation_tokens: int = 0
    continuation_threshold: float = .98

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Expected an input object.")
        ads = value.get("advertisers")
        if not isinstance(ads, list) or len(ads) != 2 or any(not isinstance(ad, dict) for ad in ads):
            raise ValueError("Exactly two advertisers are required.")
        carry_tokens = value.get("continuation_tokens", 0)
        carry_threshold = value.get("continuation_threshold", .98)
        validate_continuation(carry_tokens, carry_threshold)
        return cls(
            text_field(value.get("prompt"), "Shared prompt", 6000),
            text_field(value.get("instructions"), "Shared instructions", 8000, empty=True),
            tuple(Bidder(
                text_field(ad.get("brand"), f"Advertiser {i + 1} name", 160),
                text_field(ad.get("facts"), f"Advertiser {i + 1} facts", 6000, empty=True),
                text_field(ad.get("objective"), f"Advertiser {i + 1} objective", 6000, empty=True),
                numeric(ad.get("bid"), f"Advertiser {i + 1} bid", 0, 1000),
            ) for i, ad in enumerate(ads)),
            numeric(value.get("seed"), "Seed", 0, 2**32 - 1, integer=True),
            numeric(value.get("temperature"), "Temperature", .05, 2),
            numeric(value.get("max_tokens"), "Safety token limit", 1, MAX_OUTPUT_TOKENS, integer=True),
            carry_tokens, carry_threshold,
        )

    @property
    def baseline(self):
        return all(ad.bid == 0 for ad in self.advertisers)


def default_inputs():
    return json.loads(files("token_auction_lab").joinpath("live_defaults.json").read_text())


def live_messages(request: LiveRequest, advertiser: int | None):
    """Only the selected advertiser's brief is read; bids never enter prompts."""
    if advertiser not in (None, 0, 1):
        raise ValueError("Exactly two advertisers are supported.")
    if advertiser is None:
        # The reference is the unmodified user prompt, with no ad-related system text.
        return [{"role": "user", "content": request.prompt}]
    else:
        ad = request.advertisers[advertiser]
        system = f"{request.instructions}\n\n" \
            f"Your advertiser: {ad.brand}\nYour advertiser objective:\n{ad.objective}\n\n" \
            f"Your advertiser's fact sheet:\n{ad.facts}"
    return [{"role": "system", "content": system}, {"role": "user", "content": request.prompt}]


class InferenceEngine:
    def __init__(self):
        self.model = self.tokenizer = None
        self.metadata = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        torch.set_num_threads(8)
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else (
            "float16" if device == "cuda" else "float32"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            dtype=getattr(torch, dtype), attn_implementation="sdpa",
        ).to(device).eval()
        self.metadata = {
            "model_id": MODEL, "model_revision": REVISION,
            "tokenizer": MODEL, "tokenizer_revision": REVISION,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_vocab_size": self.model.config.vocab_size, "tokenizer_size": len(self.tokenizer),
            "device": device, "dtype": dtype, "attention_implementation": "sdpa",
            "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
            "platform": platform.platform(), "python": platform.python_version(),
            "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
            "model_load_seconds": time.perf_counter() - started,
            "model_load_scope": "disk only; reused between live rounds",
            "local_files_only": True, "torch_num_threads": torch.get_num_threads(),
            "eos_token_ids": self.model.generation_config.eos_token_id,
            "sampling_filters": None, "scenario": "editable-live-inputs",
            "timing_note": "Ad generation includes prefill, both bidder forwards and the no-ad observer forward, "
                "sampling, full-vector retention for the active inspector, trace I/O, and "
                "synchronization. "
                "Cache validation and initial model load are separate.",
        }

    def decode(self, token_id):
        return self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

    def run(self, request, output, on_event, on_distributions, *, should_stop=None):
        import torch
        from token_auction_lab.runner import RoundConfig, run_round

        torch.manual_seed(request.seed)
        prompts = []
        for i in ([None] if request.baseline else [0, 1, None]):
            text = self.tokenizer.apply_chat_template(
                live_messages(request, i), tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > 2048:
                raise ValueError(f"Context {i if i is not None else 'baseline'} has {len(ids)} "
                    "prompt tokens; this local demo allows 2048 per context. Shorten the briefs.")
            prompts.append((text, ids))
        sources = [*sorted((ROOT / "token_auction_lab").glob("*.py")), ROOT / "auction.py"]
        metadata = {**self.metadata, "torch_seed": request.seed, "editable_inputs": asdict(request),
            "advertiser_prompt_policy": ADVERTISER_PROMPT_POLICY,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources},
            "reference_sha256": hashlib.sha256((ROOT / "auction.py").read_bytes()).hexdigest(),
        }
        config = RoundConfig(
            request.advertisers[0].bid, request.advertisers[1].bid,
            seed=request.seed, temperature=request.temperature, max_tokens=request.max_tokens,
            baseline=request.baseline,
            continuation_tokens=0 if request.baseline else request.continuation_tokens,
            continuation_threshold=request.continuation_threshold,
        )
        metadata["no_ad_prompt_policy"] = "Only the user message; excludes shared ad instructions, briefs, objectives, and bids."
        if request.baseline:
            return run_round(self.model, self.tokenizer, prompts, config, output, metadata,
                on_event=on_event, on_distributions=on_distributions, should_stop=should_stop)
        no_ad_prompt = prompts.pop()
        result = run_round(self.model, self.tokenizer, prompts, config, output, metadata,
            on_event=on_event, on_distributions=on_distributions, no_ad_prompt=no_ad_prompt,
            should_stop=should_stop)

        return result


class LiveService:
    """One active input snapshot and worker, with an accompanying no-ad reference."""

    def __init__(self, output_root=None, engine=None):
        self.output_root = output_root or ROOT / "artifacts/live-lab"
        self.engine = engine or InferenceEngine()
        self.lock = threading.Lock()
        self.current = None
        self.worker = None
        self.stop_event = None

    def start(self, payload):
        request = LiveRequest.parse(payload)
        with self.lock:
            if self.current and self.current["status"] in BUSY:
                raise BusyError("Inference is already running. Inputs are locked until it finishes.")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{stamp}-live-{uuid.uuid4().hex[:10]}"
            self.current = {
                "run_id": run_id, "status": "loading", "inputs": asdict(request),
                "events": [], "vectors": {}, "manifest": None, "result": None,
                "error": None, "created_at": datetime.now(timezone.utc).isoformat(),
                "stop_requested": False,
                "no_ad": {"events": [], "manifest": None, "result": None},
            }
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._run,
                args=(request, run_id, self.stop_event), daemon=True)
            self.worker.start()
            return run_id

    def stop(self, run_id):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Choose the active run to stop.")
        with self.lock:
            if not self.current or self.current["run_id"] != run_id:
                raise StaleRunError("That run is no longer active; no stop was requested.")
            if self.current["status"] in BUSY:
                self.stop_event.set()
                self.current["stop_requested"] = True
            return {"run_id": run_id, "stop_requested": self.current["stop_requested"]}

    def _run(self, request, run_id, stop_event):
        output = self.output_root / run_id
        try:
            self.engine.load()
            with self.lock:
                self.current["status"] = "prefilling"

            def retain(step, vectors):
                # Read-only arrays, retained only for the currently active run.
                copies = tuple(p.copy() for p in vectors)
                for p in copies:
                    p.flags.writeable = False
                with self.lock:
                    self.current["vectors"][step] = copies

            def emit(event):
                with self.lock:
                    if event["type"] == "round_start":
                        self.current["manifest"] = deepcopy(event["manifest"])
                    elif event["type"] == "token":
                        self.current["status"] = "generating"
                        self.current["events"].append(deepcopy(event))
                    elif event["type"] == "round_end":
                        self.current["result"] = deepcopy(event)
                    elif event["type"] == "validation_start":
                        self.current["status"] = "validating"
                    elif event["type"] == "no_ad_event":
                        e = event["event"]
                        if e["type"] == "round_start":
                            self.current["status"] = "no-ad-generating"
                            self.current["no_ad"]["manifest"] = deepcopy(e["manifest"])
                        elif e["type"] == "token":
                            self.current["no_ad"]["events"].append(deepcopy(e))
                        elif e["type"] == "validation_start":
                            self.current["status"] = "no-ad-validating"
                        elif e["type"] == "round_end":
                            self.current["no_ad"]["result"] = deepcopy(e)

            result = self.engine.run(request, output, emit, retain,
                should_stop=stop_event.is_set)
            with self.lock:
                self.current["result"] = result
                self.current["status"] = "stopped" if stop_event.is_set() else "complete"
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.json").write_text(json.dumps({
                "run_id": run_id, "status": "error", "error": message,
                "inputs": asdict(request), "model_id": MODEL, "model_revision": REVISION,
            }, indent=2) + "\n")
            with self.lock:
                self.current["error"] = message
                self.current["status"] = "error"

    def snapshot(self, run_id=None, after=0):
        with self.lock:
            if not self.current:
                return {"status": "idle", "defaults": default_inputs(), "model_id": MODEL,
                    "model_revision": REVISION}
            if run_id is not None and self.current["run_id"] != run_id:
                raise StaleRunError("A new input snapshot has replaced this run.")
            value = {key: value for key, value in self.current.items()
                if key not in {"vectors", "events"}}
            value["events"] = self.current["events"][after:]
            value["event_count"] = len(self.current["events"])
            return deepcopy(value)

    def inspect(self, run_id, step, a=None, b=None, token_id=None, offset=0, limit=20):
        import numpy as np
        from auction import RandomTape, TwoBidderAuction

        if step < 0 or offset < 0 or not 1 <= limit <= 50:
            raise ValueError("Invalid inspector page or step.")
        with self.lock:
            if not self.current or self.current["run_id"] != run_id:
                raise StaleRunError("This inspector belongs to a replaced run.")
            if step >= len(self.current["events"]):
                raise ValueError("That token has not been generated yet.")
            event = deepcopy(self.current["events"][step])
            vectors = self.current["vectors"][step]
        baseline = event["branch"] == "baseline"
        carried = event.get("selection_rule") == "winner_continuation"
        if baseline:
            q = vectors[0]
            outcome = {"token_id": event["token_id"], "branch": "baseline",
                "payment_a": 0, "payment_b": 0}
        else:
            a = event["bid_a"] if a is None else numeric(a, "Counterfactual A bid", 0, 1000)
            b = event["bid_b"] if b is None else numeric(b, "Counterfactual B bid", 0, 1000)
            auction = TwoBidderAuction(*vectors[:2])
            q = auction.mixture(a, b)
            if carried:
                if a != event["bid_a"] or b != event["bid_b"]:
                    raise ValueError("A carried token depends on a previous win. Inspect that win or refresh inference to change bids.")
                outcome = {"token_id": event["token_id"], "branch": event["branch"],
                    "payment_a": 0.0, "payment_b": 0.0}
            else:
                outcome = asdict(auction.resolve(a, b, RandomTape(**event["tape"])))
        p_no_ad = vectors[2] if not baseline and len(vectors) == 3 else None
        actual_q = q if baseline else auction.mixture(event["bid_a"], event["bid_b"])
        if token_id is not None:
            if not 0 <= token_id < len(q):
                raise ValueError(f"Token ID must be between 0 and {len(q)-1}.")
            ids = [token_id]
        else:
            ids = [int(i) for i in np.argsort(-q, kind="stable")[offset:offset + limit]]

        def row(i):
            value = {"token_id": int(i), "token": self.engine.decode(int(i)), "q": float(q[i])}
            value["p_output"] = float(i == event["token_id"]) if carried else float(actual_q[i])
            if baseline:
                value["p"] = float(vectors[0][i])
            else:
                value.update(p_a=float(vectors[0][i]), p_b=float(vectors[1][i]))
                if p_no_ad is not None:
                    value.update(p_no_ad=float(p_no_ad[i]),
                        delta_q_no_ad=float(actual_q[i] - p_no_ad[i]),
                        delta_output_no_ad=float(value["p_output"] - p_no_ad[i]))
            return value

        # Always show selected and residual candidates, including outside the top rows.
        special = {event["token_id"], outcome["token_id"]}
        special.update(event[k] for k in ("candidate_a", "candidate_b") if event.get(k) is not None)
        return {"run_id": run_id, "step": step, "baseline": baseline,
            "selection_rule": event.get("selection_rule", "baseline" if baseline else "auction"),
            "continuation": event.get("continuation"),
            "rows": [row(i) for i in ids], "pinned": [row(i) for i in sorted(special)],
            "total": len(q), "offset": offset, "limit": limit,
            "outcome": {**outcome, "token": self.engine.decode(outcome["token_id"])},
            "bid_a": a, "bid_b": b,
            "counterfactual": not baseline and (a != event["bid_a"] or b != event["bid_b"]),
            "no_ad_comparison": None if p_no_ad is None else {
                "conditioning": "Same auction-selected prefix; only the user prompt in the reference context.",
                "tv_a": float(np.abs(vectors[0] - p_no_ad).sum() / 2),
                "tv_b": float(np.abs(vectors[1] - p_no_ad).sum() / 2),
                "tv_q": float(np.abs(actual_q - p_no_ad).sum() / 2),
                "tv_output": float(1 - p_no_ad[event["token_id"]]) if carried else float(np.abs(actual_q - p_no_ad).sum() / 2),
                "tv_preview": float(np.abs(q - p_no_ad).sum() / 2),
            },
        }

    def file(self, run_id, name):
        with self.lock:
            if not self.current or self.current["run_id"] != run_id:
                raise StaleRunError("Only the active run is available here.")
        if name not in {"manifest.json", "events.jsonl", "result.json", "error.json",
                "no-ad/manifest.json", "no-ad/events.jsonl", "no-ad/result.json"}:
            raise ValueError("Unknown run file.")
        path = self.output_root / run_id / name
        if not path.is_file():
            raise ValueError("That file is not available yet.")
        return path
