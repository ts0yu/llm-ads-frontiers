"""n advertiser contexts, one model, independent KV caches, one shared token suffix.

Research extension of `decoder.SharedPrefixDecoder` (whose two-context demo
behaviour is unchanged). The code below is copied from it and generalized to
n >= 1 prefixes, plus `fork()` for exact atlas search: a fork deep-copies every
cache, so the fork and the original continue with bitwise the same numerics as
an uninterrupted run that took the same tokens.
"""

import copy

import numpy as np
import torch

from token_auction_lab.decoder import Context, probabilities


class MultiPrefixDecoder:
    """Sequential forwards are deliberate; correctness precedes batching."""

    @torch.inference_mode()
    def __init__(self, model, prefixes: list[list[int]]):
        if not prefixes or any(not p for p in prefixes):
            raise ValueError("Provide at least one nonempty prefix.")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.generated: list[int] = []
        self.contexts: list[Context] = []
        self.forward_passes = 0
        for prefix in prefixes:
            ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=self.device)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=True,
                return_dict=True,
            )
            self.forward_passes += 1
            self.contexts.append(
                Context(
                    tuple(prefix),
                    list(prefix),
                    mask,
                    output.past_key_values,
                    output.logits[0, -1].detach(),
                )
            )
        self.assert_separate_caches()

    def assert_separate_caches(self) -> None:
        caches = [id(c.cache) for c in self.contexts]
        if len(set(caches)) != len(caches):
            raise RuntimeError("Advertiser contexts unexpectedly share a cache.")

    @property
    def vocab_size(self) -> int:
        return self.contexts[0].logits.numel()

    def distributions(self, temperature: float) -> list[np.ndarray]:
        return [probabilities(c.logits, temperature) for c in self.contexts]

    @torch.inference_mode()
    def fork(self) -> "MultiPrefixDecoder":
        """Independent copy: separate caches, masks, logits, and token lists."""
        clone = object.__new__(MultiPrefixDecoder)
        clone.model, clone.device = self.model, self.device
        clone.generated = list(self.generated)
        clone.forward_passes = 0
        clone.contexts = [
            Context(
                c.prefix,
                list(c.token_ids),
                c.attention_mask.clone(),
                copy.deepcopy(c.cache),
                c.logits.clone(),
            )
            for c in self.contexts
        ]
        clone.assert_separate_caches()
        for mine, theirs in zip(self.contexts, clone.contexts):
            if mine.cache is theirs.cache:
                raise RuntimeError("A fork shares a cache with its parent.")
        return clone

    @torch.inference_mode()
    def advance(self, token_id: int) -> None:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError("Token ID is outside the common vocabulary.")
        # The selected integer is fed directly to every context. There is no
        # tokenization, chat-template application, or per-token chat turn here.
        for context in self.contexts:
            length = len(context.token_ids)
            ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            mask = torch.ones((1, length + 1), dtype=torch.long, device=self.device)
            position = torch.tensor([length], dtype=torch.long, device=self.device)
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position.unsqueeze(0),
                cache_position=position,
                past_key_values=context.cache,
                use_cache=True,
                return_dict=True,
            )
            self.forward_passes += 1
            context.token_ids.append(token_id)
            context.attention_mask = mask
            context.cache = output.past_key_values
            context.logits = output.logits[0, -1].detach()
        self.generated.append(token_id)
        self.assert_shared_suffix()

    def assert_shared_suffix(self) -> None:
        for context in self.contexts:
            if context.token_ids[len(context.prefix) :] != self.generated:
                raise RuntimeError("Advertisers received different generated token IDs.")

    @torch.inference_mode()
    def check_cache(
        self,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        temperature: float = 1.0,
        max_probability_tv: float | None = None,
    ) -> list[float]:
        """Compare cached logits against full-prefix recomputation, without a cache."""
        errors = []
        self.cache_probability_tvs = []
        for context in self.contexts:
            ids = torch.tensor([context.token_ids], dtype=torch.long, device=self.device)
            positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
            actual = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            ).logits[0, -1]
            error = float((actual.float() - context.logits.float()).abs().max().item())
            torch.testing.assert_close(actual, context.logits, atol=atol, rtol=rtol)
            tv = float(
                np.abs(
                    probabilities(actual, temperature) - probabilities(context.logits, temperature)
                ).sum()
                / 2
            )
            if max_probability_tv is not None and tv > max_probability_tv:
                raise AssertionError(f"Cached probability TV {tv} exceeds {max_probability_tv}.")
            self.cache_probability_tvs.append(tv)
            errors.append(error)
        return errors
