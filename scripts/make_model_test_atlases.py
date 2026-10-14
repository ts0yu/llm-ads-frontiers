"""Run the REAL atlas generators on tiny RANDOM models, then publish for the explorer.

    python scripts/make_model_test_atlases.py   # -> walkthrough/atlas_data/model-test-*

Everything this writes is labeled ``synthetic_model_test`` (MOONSHOTS.md section 5):
random weights, so the text is word salad and the images are noise. It checks the
pipeline end to end (generator CLI -> atlas JSON -> publish -> explorer), not model
quality. The generators' own CLI ``main`` functions run with ``mode=
"synthetic_model_test"``; only model loading is replaced (text and simplex), because
Hugging Face downloads are not available and the weights are random anyway.

* text: ``token_auction_lab.atlas`` on a tiny random Qwen3 and a word-level test
  tokenizer built from the scenario prompts (brand names are single tokens).
* simplex: ``token_auction_lab.simplex_atlas`` (three bidders), same kind of model.
* image: ``token_auction_lab.diffusion_atlas`` on a tiny random SDXL pipeline saved to
  a temporary directory (character-level CLIP tokenizer, 1-block VAE).

Model weights stay in a temporary directory and are never committed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import sys
import tempfile
from importlib.metadata import distributions
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEXT_MODEL_ID = "tiny-random-qwen3-wordlevel (synthetic_model_test)"
IMAGE_MODEL_DIR = "tiny-random-sdxl"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
WORD_RE = re.compile(r"\w+|[^\w\s]")


# ------------------------------------------------------------------------- tiny text
def scenario_texts(text_scenarios=("transit-hotel",), simplex_scenarios=("three-hotels",)):
    """All message text the CLIs will render, and every brand name (one token each)."""
    from token_auction_lab.prompts import messages, scenarios
    from token_auction_lab.prompts_n import brands, messages_n, scenarios_n

    texts, names = [], set()
    for name in text_scenarios:
        scenario = scenarios()[name]
        names |= {ad["brand"] for ad in scenario["advertisers"]}
        texts += [m["content"] for i in (0, 1) for m in messages(scenario, i)]
    for name in simplex_scenarios:
        scenario = scenarios_n()[name]
        names |= set(brands(scenario))
        texts += [m["content"] for i in range(3) for m in messages_n(scenario, i)]
    return texts, sorted(names)


def word_tokenizer(texts: list[str], brands: list[str]):
    """A real ``PreTrainedTokenizerFast`` with a word-level vocabulary: test-only, not Qwen's."""
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0}
    for text in texts:
        for word in WORD_RE.findall(text):
            vocab.setdefault(word, len(vocab))
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tokenizer.add_tokens([AddedToken(b, normalized=False) for b in brands])
    tokenizer.add_special_tokens({"eos_token": "<|im_end|>",
                                  "additional_special_tokens": ["<|im_start|>"]})
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def tiny_qwen(vocab_size: int, eos_token_id: int, seed: int = 17):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=2048, eos_token_id=eos_token_id,
        # Large random weights give peaked, disagreeing distributions (more contests).
        initializer_range=0.5,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


def text_metadata(model, tokenizer, seed: int) -> dict:
    import torch

    return {
        "model_id": TEXT_MODEL_ID,
        "model_revision": None,
        "model_note": f"Qwen3ForCausalLM with random weights (torch.manual_seed({seed}), "
                      "initializer_range 0.5); built in memory, never saved.",
        "tokenizer": "word-level test tokenizer (scripts/make_model_test_atlases.py)",
        "tokenizer_revision": None,
        "tokenizer_class": type(tokenizer).__name__,
        "model_vocab_size": model.config.vocab_size,
        "tokenizer_size": len(tokenizer),
        "device": "cpu",
        "dtype": "float32",
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "attention_implementation": "sdpa",
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "python": platform.python_version(),
        "torch_num_threads": torch.get_num_threads(),
        "generator": "scripts/make_model_test_atlases.py",
    }


def run_text(out: Path, *, scenario: str, seeds: list[int], max_tokens: int,
             temperature: float, verify: int, model_seed: int = 17) -> Path:
    from token_auction_lab import atlas

    texts, names = scenario_texts((scenario,), ())
    tokenizer = word_tokenizer(texts, names)
    model = tiny_qwen(len(tokenizer), tokenizer.eos_token_id, model_seed)
    loaded = (model, tokenizer, text_metadata(model, tokenizer, model_seed))
    argv = ["--scenario", scenario, "--seeds", *map(str, seeds), "--max-tokens",
            str(max_tokens), "--temperature", str(temperature), "--verify", str(verify),
            "--device", "cpu", "--dtype", "float32", "--output", str(out)]
    with patch("token_auction_lab.atlas._load", return_value=loaded):
        atlas.main(argv, mode="synthetic_model_test")
    return out


def run_simplex(out: Path, *, scenario: str, seeds: list[int], max_tokens: int,
                temperature: float, verify: int, max_cells: int, model_seed: int = 23) -> Path:
    from token_auction_lab import simplex_atlas

    texts, names = scenario_texts((), (scenario,))
    tokenizer = word_tokenizer(texts, names)
    model = tiny_qwen(len(tokenizer), tokenizer.eos_token_id, model_seed)
    loaded = (tokenizer, model, None, "cpu", "float32", 0.0)
    argv = ["--model", TEXT_MODEL_ID, "--scenario", scenario, "--seeds", *map(str, seeds),
            "--max-tokens", str(max_tokens), "--temperature", str(temperature),
            "--verify", str(verify), "--max-cells", str(max_cells), "--device", "cpu",
            "--dtype", "float32", "--output", str(out)]
    with patch("token_auction_lab.cli.load_model", return_value=loaded):
        simplex_atlas.main(argv, mode="synthetic_model_test")
    return out


# ------------------------------------------------------------------------ tiny image
def tiny_sdxl_dir(root: Path, max_length: int = 256) -> Path:
    """Save a tiny random SDXL pipeline (character-level CLIP tokenizer, 1-block VAE)."""
    import torch
    from diffusers import (
        AutoencoderKL,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
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
    # Character-level, so the window is widened to hold the real scene prompts untruncated.
    tokenizer = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"),
                              model_max_length=max_length)
    torch.manual_seed(0)
    text_cfg = dict(vocab_size=len(vocab), intermediate_size=16, num_attention_heads=2,
                    num_hidden_layers=2, max_position_embeddings=max_length,
                    bos_token_id=vocab["<|startoftext|>"], eos_token_id=vocab["<|endoftext|>"])
    vae = AutoencoderKL(block_out_channels=(8,), in_channels=3, out_channels=3,
                        latent_channels=4, norm_num_groups=4,
                        down_block_types=("DownEncoderBlock2D",),
                        up_block_types=("UpDecoderBlock2D",))
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=12,
        attention_head_dim=2, norm_num_groups=4, addition_embed_type="text_time",
        addition_time_embed_dim=4, projection_class_embeddings_input_dim=4 * 6 + 10)
    scheduler = EulerAncestralDiscreteScheduler(
        beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
        timestep_spacing="leading", steps_offset=1)
    enc1 = CLIPTextModel(CLIPTextConfig(hidden_size=6, **text_cfg))
    enc2 = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=6, projection_dim=10,
                                                      **text_cfg))
    pipe = StableDiffusionXLPipeline(vae, enc1, enc2, tokenizer, tokenizer, unet, scheduler)
    out = root / IMAGE_MODEL_DIR
    pipe.save_pretrained(out)
    return out


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_image(out: Path, *, scene: str, seeds: list[int], steps: int, size: int,
              verify: int, work: Path) -> Path:
    import torch

    from token_auction_lab import diffusion_atlas

    tiny_sdxl_dir(work)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny convolutions are much slower with thread contention
    try:
        # A relative --model keeps temporary paths out of the committed manifest.
        with working_directory(work):
            diffusion_atlas.main(
                ["--model", IMAGE_MODEL_DIR, "--local-files-only", "--device", "cpu",
                 "--dtype", "float32", "--scene", scene, "--seeds", *map(str, seeds),
                 "--steps", str(steps), "--guidance", "6", "--size", str(size),
                 "--verify", str(verify), "--jpeg-size", "0", "--jpeg-quality", "80",
                 "--output", str(out.resolve())],
                mode="synthetic_model_test")
    finally:
        torch.set_num_threads(threads)
    return out


# ------------------------------------------------------------------------------ main
def generate(work: Path, args) -> list[Path]:
    import torch

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    runs = []
    runs.append(run_text(work / "model-test-text", scenario=args.text_scenario,
                         seeds=args.text_seeds, max_tokens=args.text_tokens,
                         temperature=args.temperature, verify=2))
    runs.append(run_simplex(work / "model-test-simplex", scenario=args.simplex_scenario,
                            seeds=args.simplex_seeds, max_tokens=args.simplex_tokens,
                            temperature=args.temperature, verify=3,
                            max_cells=args.max_cells))
    model_dir = work / "models"
    model_dir.mkdir(exist_ok=True)
    runs.append(run_image(work / "model-test-image", scene=args.scene, seeds=args.image_seeds,
                          steps=args.image_steps, size=args.image_size, verify=2,
                          work=model_dir))
    return runs


def main(argv: list[str] | None = None) -> int:
    from scripts import publish_atlases

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=None,
                        help="keep generator outputs here (default: a temporary directory)")
    parser.add_argument("--dest", type=Path, default=publish_atlases.DEST)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--text-scenario", default="transit-hotel")
    parser.add_argument("--text-seeds", type=int, nargs="+", default=[3])
    parser.add_argument("--text-tokens", type=int, default=24)
    parser.add_argument("--simplex-scenario", default="three-hotels")
    parser.add_argument("--simplex-seeds", type=int, nargs="+", default=[5])
    parser.add_argument("--simplex-tokens", type=int, default=12)
    parser.add_argument("--max-cells", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scene", default="cafe")
    parser.add_argument("--image-seeds", type=int, nargs="+", default=[2])
    parser.add_argument("--image-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args(argv)
    with contextlib.ExitStack() as stack:
        work = args.work or Path(stack.enter_context(tempfile.TemporaryDirectory()))
        work.mkdir(parents=True, exist_ok=True)
        runs = generate(work, args)
        if not args.no_publish:
            publish_atlases.main([*map(str, runs), "--dest", str(args.dest),
                                  "--no-auto-tokenizer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
