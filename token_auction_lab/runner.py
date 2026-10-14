"""Milestone 1: real distributions -> supplied reference auction -> shared token IDs."""

import json
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.continuation import WinnerContinuation, validate_continuation
from token_auction_lab.decoder import SharedPrefixDecoder
from token_auction_lab.limits import MAX_OUTPUT_TOKENS
from token_auction_lab.prompts import prompt_hash


@dataclass(frozen=True)
class RoundConfig:
    bid_a: float = 1.0
    bid_b: float = 1.0
    seed: int = 7
    temperature: float = 0.8
    max_tokens: int = 48
    top_k: int = 8
    baseline: bool = False
    checkpoint_steps: tuple[int, ...] = (0, 8, 24)
    continuation_tokens: int = 0
    continuation_threshold: float = .98

    def __post_init__(self):
        validate_continuation(self.continuation_tokens, self.continuation_threshold)
        if self.baseline and self.continuation_tokens:
            raise ValueError("The no-ad baseline cannot grant advertiser continuation.")
        if not all(math.isfinite(b) and b >= 0 for b in (self.bid_a, self.bid_b)):
            raise ValueError("Bids must be finite and nonnegative.")
        total = self.bid_a + self.bid_b
        if not math.isfinite(total):
            raise ValueError("Bid sum must be finite.")
        if self.baseline and total != 0:
            raise ValueError("The no-ad baseline has no bids.")
        if not self.baseline and total == 0:
            raise ValueError("Use --baseline for all-zero bids.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        if (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int)
                or not 1 <= self.max_tokens <= MAX_OUTPUT_TOKENS or not 1 <= self.top_k <= 20):
            raise ValueError(f"max_tokens must be an integer in 1..{MAX_OUTPUT_TOKENS} "
                "and top_k must be 1..20.")
        if len(self.checkpoint_steps) > 4 or any(s < 0 for s in self.checkpoint_steps):
            raise ValueError("Retain at most four nonnegative inspection checkpoints per run.")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device, sampled_mps_peak: int) -> dict:
    """Separate OS process high-water marks from device allocator measurements."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory = {
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "process_peak_scope": "process lifetime through generation end; includes model load "
        "and any earlier rounds, not an isolated per-round device peak",
    }
    if device.type == "cuda":
        memory.update(kind="cuda_peak_allocated", bytes=torch.cuda.max_memory_allocated(device))
    elif device.type == "mps":
        memory.update(
            kind="mps_current_allocated_not_peak",
            bytes=torch.mps.current_allocated_memory(),
            sampled_peak_allocated_bytes=sampled_mps_peak,
            sampling_note="Sampled after prefill and each token; excludes transient allocations "
            "inside forwards. The MPS API does not expose an allocator peak.",
        )
    else:
        memory.update(kind="device_memory_unavailable", bytes=None)
    return memory


def top_tokens(tokenizer, p_a, p_b, q, k: int, p_no_ad=None) -> list[dict]:
    ids: set[int] = set()
    for p in ((p_a, p_b, q) if p_no_ad is None else (p_a, p_b, q, p_no_ad)):
        count = min(k, len(p))
        ids.update(int(i) for i in np.argpartition(p, -count)[-count:])
    return [
        {
            "token_id": i,
            "token": tokenizer.decode([i], clean_up_tokenization_spaces=False),
            "p_a": float(p_a[i]),
            "p_b": float(p_b[i]),
            "q": float(q[i]),
            **({"p_no_ad": float(p_no_ad[i]), "delta_q_no_ad": float(q[i] - p_no_ad[i])}
               if p_no_ad is not None else {}),
        }
        for i in sorted(ids, key=lambda i: (-q[i], i))
    ]


def eos_ids(model, tokenizer) -> set[int]:
    values = model.generation_config.eos_token_id
    result = set(values if isinstance(values, (tuple, list)) else [values])
    result.add(tokenizer.eos_token_id)
    return {int(i) for i in result if i is not None}


@torch.inference_mode()
def run_round(
    model,
    tokenizer,
    prompts: list[tuple[str, list[int]]],
    config: RoundConfig,
    output: Path,
    metadata: dict,
    *,
    mode: str = "live_model",
    on_event: Callable[[dict], None] | None = None,
    on_distributions: Callable[[int, list[np.ndarray]], None] | None = None,
    no_ad_prompt: tuple[str, list[int]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    expected = 1 if config.baseline else 2
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} prompt contexts.")
    if config.baseline and no_ad_prompt is not None:
        raise ValueError("A baseline round cannot also have a no-ad reference observer.")
    output.mkdir(parents=True, exist_ok=False)
    mode = "live_baseline" if config.baseline and mode == "live_model" else mode
    manifest = {
        **metadata,
        "schema_version": 1,
        "round_id": output.name,
        "mode": mode,
        "label": "SPONSORED SIMULATION" if not config.baseline else "NO-AD BASELINE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "prompts": [
            {"text": text, "sha256": prompt_hash(text), "token_ids": ids} for text, ids in prompts
        ],
        "sampler": "no_ad_categorical" if config.baseline else (
            "experimental_winner_continuation" if config.continuation_tokens else "auction.TwoBidderAuction"),
        "continuation_policy": {
            "max_extra_tokens": config.continuation_tokens,
            "confidence_threshold": config.continuation_threshold,
            "probabilities": "After the configured temperature, over the full vocabulary.",
            "payment": "No additional charge for carried tokens.",
            "rule": "A contested winner may emit the unique token meeting the confidence threshold. "
                "Recheck at each prefix; stop at the first miss or the extra-token limit. "
                "Carried tokens do not renew the allowance. Shared outcomes grant no continuation.",
            "distribution_note": "On carried steps, actual output probability is 1 for the chosen token "
                "and 0 otherwise; q remains the hypothetical reference auction mixture. "
                "Reference auction guarantees do not extend to this sequence mechanism.",
            "rng_note": "Carried steps consume no random tape; fresh auctions use the next tape.",
        },
        "credit_unit": "simulated",
        "rng": "numpy.PCG64",
        "stopping_policy": "Sample until a configured EOS token, max_tokens safety limit, "
            "or a user stop request. Finish and record any selected token before stopping.",
        "replay_note": "Recorded tokens replay without inference. A seed alone does not "
        "guarantee identical regeneration across software or hardware.",
    }
    if no_ad_prompt is not None:
        reference_text, reference_ids = no_ad_prompt
        manifest["no_ad_reference"] = {
            "text": reference_text, "sha256": prompt_hash(reference_text),
            "token_ids": reference_ids,
            "conditioning": "User prompt only, followed by the exact auction-selected token IDs.",
            "role": "Read-only probability observer; no bid, sampling, or payment.",
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if on_event:
        on_event({"type": "round_start", "manifest": manifest})
    rng = np.random.default_rng(config.seed)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    decoder = SharedPrefixDecoder(model, [ids for _, ids in prompts])
    reference = SharedPrefixDecoder(model, [no_ad_prompt[1]]) if no_ad_prompt else None
    synchronize(device)
    prefilled = time.perf_counter()
    sampled_mps_peak = torch.mps.current_allocated_memory() if device.type == "mps" else 0
    first_token_time = None
    cumulative_a = cumulative_b = 0.0
    disagreements = []
    stop_reason = "length"
    text = ""
    continuation = WinnerContinuation(config.continuation_tokens, config.continuation_threshold)
    carried_tokens = 0
    with (output / "events.jsonl").open("w") as trace:
        for step in range(config.max_tokens):
            if should_stop and should_stop():
                stop_reason = "user_stop"
                break
            distributions = decoder.distributions(config.temperature)
            p_no_ad = reference.distributions(config.temperature)[0] if reference else None
            if on_distributions:
                on_distributions(step, distributions + ([p_no_ad] if reference else []))
            if config.baseline:
                p = distributions[0]
                u = float(rng.random())
                cdf = np.cumsum(p)
                cdf[-1] = 1.0
                selected = int(np.searchsorted(cdf, u, side="right"))
                event = {"branch": "baseline", "uniform": u, "payment_a": 0.0, "payment_b": 0.0}
                event["selected_probabilities"] = {"p": float(p[selected])}
            else:
                p_a, p_b = distributions
                auction = TwoBidderAuction(p_a, p_b)
                carried = continuation.take(p_a, p_b)
                if carried:
                    allocation, carry_info = carried
                    tape = None
                    carried_tokens += 1
                else:
                    tape = RandomTape.draw(rng)
                    allocation = auction.resolve(config.bid_a, config.bid_b, tape)
                    continuation.observe_auction(allocation, step)
                selected = allocation.token_id
                q = auction.mixture(config.bid_a, config.bid_b)
                if not (
                    0 <= allocation.payment_a <= config.bid_a + 1e-10
                    and 0 <= allocation.payment_b <= config.bid_b + 1e-10
                ):
                    raise RuntimeError("Reference payment exceeded a fixed bid.")
                cumulative_a += allocation.payment_a
                cumulative_b += allocation.payment_b
                disagreements.append(auction.disagreement)
                event = {
                    **asdict(allocation),
                    "bid_a": config.bid_a,
                    "bid_b": config.bid_b,
                    "shared_mass": auction.shared_mass,
                    "disagreement_mass": auction.disagreement,
                    "selection_rule": "winner_continuation" if carried else "auction",
                    "continuation": carry_info if carried else None,
                    "tape": asdict(tape) if tape else None,
                    "threshold_a": config.bid_b * tape.contest / (1 - tape.contest) if tape else None,
                    "threshold_b": config.bid_a * (1 - tape.contest) / tape.contest if tape else None,
                    "top_tokens": top_tokens(tokenizer, p_a, p_b, q, config.top_k, p_no_ad),
                    "selected_probabilities": {
                        "p_a": float(p_a[selected]),
                        "p_b": float(p_b[selected]),
                        "q": float(q[selected]),
                        "p_output": 1.0 if carried else float(q[selected]),
                    },
                    "candidate_a_text": None if allocation.candidate_a is None else tokenizer.decode(
                        [allocation.candidate_a], clean_up_tokenization_spaces=False
                    ),
                    "candidate_b_text": None if allocation.candidate_b is None else tokenizer.decode(
                        [allocation.candidate_b], clean_up_tokenization_spaces=False
                    ),
                }
                for row in event["top_tokens"]:
                    row["p_output"] = float(row["token_id"] == selected) if carried else row["q"]
                if reference:
                    event["selected_probabilities"].update(
                        p_no_ad=float(p_no_ad[selected]),
                        delta_q_no_ad=float(q[selected] - p_no_ad[selected]),
                        delta_output_no_ad=float((1.0 if carried else q[selected]) - p_no_ad[selected]),
                    )
                    event["no_ad_comparison"] = {
                        "tv_a": float(np.abs(p_a - p_no_ad).sum() / 2),
                        "tv_b": float(np.abs(p_b - p_no_ad).sum() / 2),
                        "tv_q": float(np.abs(q - p_no_ad).sum() / 2),
                        "tv_output": float(1 - p_no_ad[selected]) if carried else float(np.abs(q - p_no_ad).sum() / 2),
                    }
                if step in config.checkpoint_steps:
                    filename = f"checkpoint-{step:03}.npz"
                    np.savez_compressed(
                        output / filename,
                        p_a=p_a,
                        p_b=p_b,
                        prefix=np.array(decoder.generated, dtype=np.int64),
                        **({"p_no_ad": p_no_ad} if reference else {}),
                    )
                    event["checkpoint"] = filename
            decoder.advance(selected)
            if reference:
                reference.advance(selected)
                if reference.generated != decoder.generated:
                    raise RuntimeError("The no-ad observer must follow the exact auction prefix.")
            # Decode the entire shared sequence, not individual token strings:
            # byte-fallback/UTF-8 pieces can combine across token boundaries.
            text = tokenizer.decode(
                decoder.generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            synchronize(device)
            elapsed = time.perf_counter() - started
            if device.type == "mps":
                sampled_mps_peak = max(sampled_mps_peak, torch.mps.current_allocated_memory())
            if first_token_time is None:
                first_token_time = elapsed
            event.update(
                {
                    "type": "token",
                    "round_id": output.name,
                    "mode": mode,
                    "step": step,
                    "token_id": selected,
                    "token_text": tokenizer.decode([selected], clean_up_tokenization_spaces=False),
                    "decoded_text_so_far": text,
                    "cumulative_a": cumulative_a,
                    "cumulative_b": cumulative_b,
                    "elapsed_seconds": elapsed,
                }
            )
            trace.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
            if on_event:
                on_event(event)
            if selected in eos_ids(model, tokenizer):
                stop_reason = "eos"
                break
        synchronize(device)
        ended = time.perf_counter()
        memory = memory_snapshot(device, sampled_mps_peak)
        # The numerical check is outside generation latency measurements.
        dtype = next(model.parameters()).dtype
        atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.08, 0.02)
        if dtype == torch.float32 and device.type == "mps":
            # Full-prefix GEMM and single-token GEMV accumulate differently on MPS.
            # The 4B-model diagnostic observed up to 2.29e-4 logit drift; retain
            # an independent probability-space bound as well (see SMOKE_RESULTS.md).
            atol = 3e-4
        max_probability_tv = 1e-4 if dtype == torch.float32 else None
        validation_started = time.perf_counter()
        if on_event:
            on_event({"type": "validation_start", "round_id": output.name})
        errors = decoder.check_cache(
            atol=atol,
            rtol=rtol,
            temperature=config.temperature,
            max_probability_tv=max_probability_tv,
        )
        reference_validation = None
        if reference:
            reference_errors = reference.check_cache(
                atol=atol, rtol=rtol, temperature=config.temperature,
                max_probability_tv=max_probability_tv,
            )
            reference_validation = {
                "shared_suffix_verified": reference.generated == decoder.generated,
                "cache_max_absolute_errors": reference_errors,
                "cache_probability_tvs": reference.cache_probability_tvs,
            }
        synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        count = len(decoder.generated)
        result = {
            "type": "round_end",
            "round_id": output.name,
            "mode": mode,
            "bid_a": config.bid_a,
            "bid_b": config.bid_b,
            "seed": config.seed,
            "token_ids": decoder.generated,
            "text": text,
            "tokens": count,
            "carried_tokens": carried_tokens,
            "stop_reason": stop_reason,
            "payment_a": cumulative_a,
            "payment_b": cumulative_b,
            "prefill_seconds": prefilled - started,
            "seconds": ended - started,
            "time_to_first_token_seconds": first_token_time,
            "seconds_per_token": (ended - started) / count if count else None,
            "decode_seconds_per_token": (ended - prefilled) / count if count else None,
            "mean_disagreement_mass": float(np.mean(disagreements)) if disagreements else None,
            "shared_suffix_verified": True,
            "cache_max_absolute_errors": errors,
            "cache_probability_tvs": decoder.cache_probability_tvs,
            "cache_tolerance": {
                "atol": atol,
                "rtol": rtol,
                "max_probability_tv": max_probability_tv,
            },
            "cache_validation_seconds": validation_seconds,
            "memory": memory,
        }
        if reference_validation is not None:
            result["no_ad_reference_validation"] = reference_validation
        trace.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if on_event:
        on_event(result)
    return result
