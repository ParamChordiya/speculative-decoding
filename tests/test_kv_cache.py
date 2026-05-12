"""Integration tests for ModelWrapper KV cache management.

These tests load GPT-2 (small, 124 M parameters) and verify:
  1. The cache has the expected shape after a forward pass.
  2. truncate_cache() correctly slices the sequence-length dimension.
  3. A forward pass with a truncated cache completes without error and
     produces logits / a new cache of the expected shape.

GPT-2 small architecture constants used for shape assertions:
  n_layer = 12   (transformer layers → outer cache tuple length)
  n_head  = 12   (attention heads   → dim 1 of each K/V tensor)
  n_embd  = 768  (hidden dimension)
  head_dim = 64  (n_embd / n_head   → dim 3 of each K/V tensor)

KV cache shape recap:
  past_key_values : tuple of n_layer tuples
  each inner tuple: (key, value)
  each tensor shape: [batch_size, n_head, seq_len, head_dim]

NOTE: These are integration tests — they download GPT-2 weights on first run
(~500 MB) and require torch + transformers to be installed.  Run them with:
  pytest tests/test_kv_cache.py -v
"""

import torch
import pytest

from src.models.loader import load_model
from src.models.wrapper import ModelWrapper

# ---------------------------------------------------------------------------
# GPT-2 small architecture constants
# ---------------------------------------------------------------------------
_N_LAYERS = 12
_N_HEADS = 12
_HEAD_DIM = 64   # 768 / 12
_VOCAB = 50257
_BATCH = 1

# A prompt long enough to always tokenise to ≥ 20 tokens with the GPT-2 BPE.
_LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly."
)


# ---------------------------------------------------------------------------
# Shared fixture — load GPT-2 once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wrapper() -> ModelWrapper:
    """Load GPT-2 small in float32 (works on CPU without CUDA)."""
    model, tokenizer = load_model("gpt2", "float32")
    device = next(model.parameters()).device
    return ModelWrapper(model, tokenizer, device)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _encode(wrapper: ModelWrapper, n_tokens: int) -> torch.Tensor:
    """Return a [1, n_tokens] input_ids tensor on the wrapper's device."""
    ids = wrapper.tokenizer.encode(_LONG_PROMPT, return_tensors="pt")
    assert ids.shape[1] >= n_tokens, (
        f"Prompt tokenises to only {ids.shape[1]} tokens; need >= {n_tokens}. "
        "Extend _LONG_PROMPT."
    )
    return ids[:, :n_tokens].to(wrapper.device)


# ---------------------------------------------------------------------------
# 1. Cache shape after a forward pass
# ---------------------------------------------------------------------------

class TestCacheShape:
    """Verify the cache structure returned by a fresh forward pass."""

    def test_outer_tuple_length_equals_num_layers(self, wrapper):
        """One (key, value) pair per transformer layer."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        assert len(past_kv) == _N_LAYERS

    def test_each_layer_is_key_value_pair(self, wrapper):
        """Each layer entry is a 2-tuple (key, value)."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        for layer_idx, entry in enumerate(past_kv):
            assert len(entry) == 2, (
                f"Layer {layer_idx}: expected (key, value) pair, got {len(entry)} elements"
            )

    def test_key_shape(self, wrapper):
        """Key tensors: [batch, n_heads, seq_len, head_dim]."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        expected = (_BATCH, _N_HEADS, 20, _HEAD_DIM)
        for layer_idx, (k, _) in enumerate(past_kv):
            assert k.shape == expected, (
                f"Layer {layer_idx} key: got {tuple(k.shape)}, expected {expected}"
            )

    def test_value_shape(self, wrapper):
        """Value tensors: [batch, n_heads, seq_len, head_dim]."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        expected = (_BATCH, _N_HEADS, 20, _HEAD_DIM)
        for layer_idx, (_, v) in enumerate(past_kv):
            assert v.shape == expected, (
                f"Layer {layer_idx} value: got {tuple(v.shape)}, expected {expected}"
            )

    def test_get_cache_length_matches_input(self, wrapper):
        """get_cache_length() must return the number of tokens processed."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        assert wrapper.get_cache_length(past_kv) == 20

    def test_get_cache_length_none_returns_zero(self, wrapper):
        """get_cache_length(None) must return 0 (no cache yet)."""
        assert wrapper.get_cache_length(None) == 0


# ---------------------------------------------------------------------------
# 2. Cache truncation
# ---------------------------------------------------------------------------

class TestCacheTruncation:
    """Verify truncate_cache() slices the seq_len dimension correctly."""

    def test_truncated_outer_length_unchanged(self, wrapper):
        """Truncation must not change the number of layers."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        truncated = wrapper.truncate_cache(past_kv, keep_length=10)
        assert len(truncated) == _N_LAYERS

    def test_truncated_key_shape(self, wrapper):
        """After truncation to 10, key tensors must have seq_len=10."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        truncated = wrapper.truncate_cache(past_kv, keep_length=10)
        expected = (_BATCH, _N_HEADS, 10, _HEAD_DIM)
        for layer_idx, (k, _) in enumerate(truncated):
            assert k.shape == expected, (
                f"Layer {layer_idx} truncated key: got {tuple(k.shape)}, expected {expected}"
            )

    def test_truncated_value_shape(self, wrapper):
        """After truncation to 10, value tensors must have seq_len=10."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        truncated = wrapper.truncate_cache(past_kv, keep_length=10)
        expected = (_BATCH, _N_HEADS, 10, _HEAD_DIM)
        for layer_idx, (_, v) in enumerate(truncated):
            assert v.shape == expected, (
                f"Layer {layer_idx} truncated value: got {tuple(v.shape)}, expected {expected}"
            )

    def test_get_cache_length_after_truncation(self, wrapper):
        """get_cache_length() must reflect the truncated length."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        truncated = wrapper.truncate_cache(past_kv, keep_length=10)
        assert wrapper.get_cache_length(truncated) == 10

    def test_truncate_to_one_position(self, wrapper):
        """Edge case: truncating to a single cached position must work."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        truncated = wrapper.truncate_cache(past_kv, keep_length=1)
        assert wrapper.get_cache_length(truncated) == 1

    def test_original_cache_is_unmodified(self, wrapper):
        """truncate_cache() must not mutate the original cache tensors."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)
        _ = wrapper.truncate_cache(past_kv, keep_length=10)
        # Original must still have seq_len=20
        assert wrapper.get_cache_length(past_kv) == 20


# ---------------------------------------------------------------------------
# 3. Forward pass with a truncated cache
# ---------------------------------------------------------------------------

class TestForwardWithCache:
    """Verify that a truncated cache can be fed back into a forward pass."""

    def test_next_token_logits_shape(self, wrapper):
        """
        Simulate rejection at position 10:
          - Run 20 tokens through the model to build a cache.
          - Truncate the cache to 10 positions (reject tokens 10-19).
          - Feed one new token conditioned on the truncated cache.
          - Expect logits of shape [batch, 1, vocab_size].
        """
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)

        truncated = wrapper.truncate_cache(past_kv, keep_length=10)

        # A single new token — in real speculative decoding this would be the
        # replacement token sampled from the adjusted distribution p'(x).
        next_token = input_ids[:, 10:11]
        logits, new_kv = wrapper.forward(next_token, past_key_values=truncated)

        assert logits.shape == (_BATCH, 1, _VOCAB)

    def test_new_cache_length_after_one_step(self, wrapper):
        """Cache length must be keep_length + 1 after one more token."""
        input_ids = _encode(wrapper, 20)
        _, past_kv = wrapper.forward(input_ids)

        truncated = wrapper.truncate_cache(past_kv, keep_length=10)
        next_token = input_ids[:, 10:11]
        _, new_kv = wrapper.forward(next_token, past_key_values=truncated)

        assert wrapper.get_cache_length(new_kv) == 11  # 10 cached + 1 new

    def test_forward_no_cache_logits_shape(self, wrapper):
        """Baseline: forward pass with no cache must return correct logit shape."""
        input_ids = _encode(wrapper, 5)
        logits, _ = wrapper.forward(input_ids)
        assert logits.shape == (_BATCH, 5, _VOCAB)

    def test_forward_use_cache_false_returns_none_cache(self, wrapper):
        """When use_cache=False the returned past_key_values must be None."""
        input_ids = _encode(wrapper, 5)
        _, past_kv = wrapper.forward(input_ids, use_cache=False)
        assert past_kv is None


# ---------------------------------------------------------------------------
# 4. Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_vocab_size(self, wrapper):
        assert wrapper.vocab_size == _VOCAB

    def test_param_count_approx(self, wrapper):
        # GPT-2 small has 124 M parameters
        assert 120_000_000 < wrapper.param_count < 130_000_000

    def test_memory_footprint_positive(self, wrapper):
        assert wrapper.memory_footprint_mb > 0

    def test_eos_token_id_not_none(self, wrapper):
        assert wrapper.eos_token_id is not None

    def test_repr(self, wrapper):
        r = repr(wrapper)
        assert "ModelWrapper" in r
        assert "params=" in r
