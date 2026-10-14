"""Live inference CLI. Model weights and credentials never enter trace files."""

import argparse
import hashlib
import json
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import auction
from token_auction_lab.prompts import ADVERTISER_PROMPT_POLICY, render, scenarios
from token_auction_lab.runner import RoundConfig, run_round


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--revision", help="A commit SHA; resolved to a SHA before loading if omitted."
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use cached files; requires a full revision SHA.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )


def pinned_revision(parser: argparse.ArgumentParser, args) -> bool:
    pinned = bool(args.revision and re.fullmatch(r"[0-9a-f]{40}", args.revision))
    if args.local_files_only and not pinned:
        parser.error("--local-files-only requires --revision with a full 40-character commit SHA.")
    return pinned


def load_model(args, pinned: bool, seed: int):
    """Resolve device/dtype/revision, then load ONE tokenizer and model at the same SHA.

    Returns (tokenizer, model, revision, device, dtype_name, load_seconds).
    """
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = (
            "bfloat16"
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else ("float16" if device == "cuda" else "float32")
        )
    # Resolve mutable refs once, then give BOTH tokenizer and model the same SHA.
    revision = (
        args.revision
        if pinned
        else HfApi().model_info(args.model, revision=args.revision).sha
    )
    print(
        json.dumps(
            {
                "status": "loading",
                "model": args.model,
                "revision": revision,
                "device": device,
                "dtype": dtype_name,
            }
        ),
        flush=True,
    )
    start = time.perf_counter()
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, local_files_only=args.local_files_only
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=revision,
            dtype=getattr(torch, dtype_name),
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        )
        .to(device)
        .eval()
    )
    return tokenizer, model, revision, device, dtype_name, time.perf_counter() - start


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--scenario", choices=tuple(scenarios()), default="transit-hotel")
    parser.add_argument(
        "--bids", action="append", help="A,B pair; repeatable. Default: 1,0; 1,1; 0,1."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument(
        "--baseline", action="store_true", help="Also run the separate no-ad baseline."
    )
    args = parser.parse_args()
    pinned = pinned_revision(parser, args)
    pairs = [
        tuple(float(x) for x in pair.split(",")) for pair in (args.bids or ["1,0", "1,1", "0,1"])
    ]
    if any(len(pair) != 2 for pair in pairs):
        parser.error("--bids must be A,B.")
    configs = [
        RoundConfig(a, b, seed=args.seed, temperature=args.temperature, max_tokens=args.max_tokens)
        for a, b in pairs
    ]
    if args.baseline:
        configs.append(
            RoundConfig(
                0,
                0,
                seed=args.seed,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                baseline=True,
            )
        )
    tokenizer, model, revision, device, dtype_name, load_seconds = load_model(
        args, pinned, args.seed
    )
    metadata = {
        "model_id": args.model,
        "model_revision": revision,
        "tokenizer": args.model,
        "tokenizer_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "model_vocab_size": model.config.vocab_size,
        "tokenizer_size": len(tokenizer),
        "device": device,
        "dtype": dtype_name,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "platform": platform.platform(),
        "attention_implementation": "sdpa",
        "scenario": args.scenario,
        "advertiser_prompt_policy": ADVERTISER_PROMPT_POLICY,
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "python": platform.python_version(),
        "model_load_seconds": load_seconds,
        "model_load_scope": "disk load only"
        if args.local_files_only
        else "download/cache and load",
        "local_files_only": args.local_files_only,
        "torch_seed": args.seed,
        "torch_num_threads": torch.get_num_threads(),
        "eos_token_ids": model.generation_config.eos_token_id,
        "sampling_filters": None,
        "timing_note": "Generation includes both sequential context forwards, full-vocabulary "
        "sampling, trace/checkpoint writes, and synchronization. Each selected token is fed "
        "to both contexts before its event is emitted. Cache validation is timed separately.",
        "reference_sha256": hashlib.sha256(Path(auction.__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                *sorted(Path(__file__).resolve().parent.glob("*.py")),
                Path(__file__).resolve().parent / "fact_sheets.json",
                Path(auction.__file__).resolve(),
            ]
        },
    }
    scenario = scenarios()[args.scenario]
    paired_prompts = [render(tokenizer, scenario, i) for i in (0, 1)]
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        kind = "baseline" if config.baseline else f"a{config.bid_a:g}-b{config.bid_b:g}"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-{args.scenario}-{kind}-seed{args.seed}-{uuid.uuid4().hex[:6]}"
        prompts = [render(tokenizer, scenario, None)] if config.baseline else paired_prompts
        result = run_round(model, tokenizer, prompts, config, args.output / name, metadata)
        summaries.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
