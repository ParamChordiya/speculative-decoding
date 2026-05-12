"""Unified model interface with explicit KV cache management.

Every component of the speculative decoding pipeline — the draft model, the
target model, and the rejection sampler — talks to models through this wrapper
rather than directly through HuggingFace model objects.  This keeps KV cache
bookkeeping in one place and makes the decoding loops easy to read.

Background: what is the KV cache?
──────────────────────────────────
A transformer's self-attention layer computes three projections of every input
token: Query (Q), Key (K), and Value (V).  The attention output for the current
token is a weighted sum of all Value vectors, where the weights come from the
dot-products of the current Query with every past Key.

During autoregressive generation we process one new token at a time.  Without
caching, we would recompute K and V for every previous token at every step —
redundant work that scales quadratically with sequence length.  The KV cache
stores the Key and Value tensors from all previous steps so only the new
token's K and V need to be computed each time.

Shape convention used throughout this file:
    past_key_values : tuple[tuple[Tensor, Tensor], ...]
        Outer tuple  — one element per transformer layer
        Inner tuple  — (key_tensor, value_tensor) for that layer
        Each tensor  — shape [batch_size, num_heads, seq_len, head_dim]

    seq_len grows by 1 each decode step as new K/V pairs are appended.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


class ModelWrapper:
    """Wraps a HuggingFace causal LM with explicit, manipulable KV cache.

    The primary reason this class exists — beyond tidying up the interface —
    is :meth:`truncate_cache`.  Speculative decoding rejects draft tokens at
    arbitrary positions and must roll the KV cache back to exactly that
    position before continuing.  HuggingFace models have no built-in method
    for this; we do it by slicing the stored tensors directly.

    Args:
        model:     A loaded ``AutoModelForCausalLM`` instance (already on the
                   correct device and in eval mode).
        tokenizer: The matching ``AutoTokenizer``.
        device:    The primary device the model lives on.  For models spread
                   across multiple GPUs via ``device_map="auto"``, pass the
                   device of the first layer (``next(model.parameters()).device``).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Run one forward pass and return logits + updated KV cache.

        This is the only method that touches the underlying model.  All
        decoding loops call this rather than the model directly so that
        cache type handling stays in one place.

        When ``past_key_values`` is provided, ``input_ids`` should contain
        only the *new* tokens (those not already represented in the cache).
        The model automatically attends over the cached positions without
        them being in ``input_ids``.

        Args:
            input_ids:        Token ID tensor of shape ``[batch, new_seq_len]``.
                              During prefill this is the full prompt; during
                              decode it is typically a single token
                              ``[batch, 1]``.
            past_key_values:  Cached K/V tensors from a previous forward call,
                              or ``None`` for the first pass.  Shape per layer:
                              ``[batch, num_heads, cached_len, head_dim]``.
            use_cache:        Whether to return updated KV cache.  Set to
                              ``False`` only when the cache is not needed (e.g.
                              prefill of a prompt you will discard).

        Returns:
            ``(logits, new_past_key_values)`` where logits has shape
            ``[batch, new_seq_len, vocab_size]`` and ``new_past_key_values``
            is ``None`` when ``use_cache=False``.
        """
        outputs = self._model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs.logits, outputs.past_key_values

    # ------------------------------------------------------------------
    # KV cache utilities
    # ------------------------------------------------------------------

    def truncate_cache(
        self,
        past_key_values: tuple,
        keep_length: int,
    ) -> tuple:
        """Slice the KV cache to retain only the first *keep_length* positions.

        WHY THIS METHOD EXISTS
        ─────────────────────
        Speculative decoding works in rounds.  Each round the draft model
        proposes K tokens; the target model then scores all K positions in a
        single forward pass.  Rejection sampling checks each position left-to-
        right and stops at the first rejected token — say position j.

        At that point the KV cache contains entries for positions 0 … j+K-1
        (the original context plus all K draft tokens the target just scored).
        But position j was wrong, so we must act as if positions j … j+K-1
        were never generated.  The next speculation round must start from
        position j with a clean cache.

        We do this by slicing dim 2 (the sequence-length axis) of every key
        and value tensor down to ``keep_length = j``.  The sliced tensors are
        contiguous views, not copies, so this is essentially free.

        Tensor layout reminder:
            dim 0  — batch
            dim 1  — num_heads
            dim 2  — seq_len  ← this is what we slice
            dim 3  — head_dim

        Args:
            past_key_values: Full KV cache from a recent forward call.
                             Shape per layer: ``[batch, heads, seq_len, head_dim]``.
            keep_length:     Number of token positions to retain.  Must be
                             ≤ the current cache length.

        Returns:
            Truncated KV cache with the same structure but ``seq_len`` reduced
            to ``keep_length`` in every layer's key and value tensor.
        """
        return tuple(
            (k[:, :, :keep_length, :], v[:, :, :keep_length, :])
            for k, v in past_key_values
        )

    def get_cache_length(self, past_key_values: tuple | None) -> int:
        """Return the number of token positions currently stored in the cache.

        The sequence length is read from dim 2 of the first layer's key tensor.
        All layers always have the same cached length, so checking only the
        first layer is sufficient.

        Args:
            past_key_values: A KV cache tuple, or ``None`` (e.g. at the start
                             of generation before any forward pass has run).

        Returns:
            Number of cached positions, or 0 if ``past_key_values`` is ``None``.
        """
        if past_key_values is None:
            return 0
        # past_key_values[0] is the first layer's (key, value) pair.
        # key shape: [batch, num_heads, seq_len, head_dim] — seq_len is dim 2.
        return past_key_values[0][0].shape[2]

    def reset_cache(self) -> None:
        """Signal that the cache should be discarded.

        Returns ``None``, which is the sentinel value meaning "no cache" in
        every forward call.  Callers can use this to make intent explicit::

            cache = wrapper.reset_cache()   # clearer than cache = None
            logits, cache = wrapper.forward(input_ids, cache)

        The KV cache is passed explicitly through every call rather than stored
        inside the wrapper.  This is a deliberate design choice: speculative
        decoding needs two independent caches (draft and target) that are
        managed separately, truncated at different points, and sometimes
        discarded mid-round.  Storing cache state inside the wrapper would make
        that coordination much harder to reason about.
        """
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """The underlying HuggingFace model."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The model's tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Primary device the model's first parameter lives on."""
        return self._device

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the model's vocabulary."""
        return self._model.config.vocab_size

    @property
    def param_count(self) -> int:
        """Total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self._model.parameters())

    @property
    def memory_footprint_mb(self) -> float:
        """Approximate on-device memory used by model weights, in megabytes."""
        return self._model.get_memory_footprint() / 1024 ** 2

    @property
    def eos_token_id(self) -> int | None:
        """EOS token ID from the tokenizer, or ``None`` if not defined."""
        return self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModelWrapper("
            f"params={self.param_count / 1e6:.1f}M, "
            f"vocab={self.vocab_size}, "
            f"device={self.device})"
        )
