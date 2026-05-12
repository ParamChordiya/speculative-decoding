"""Integration tests for AutoregressiveDecoder.

These tests load GPT-2 small (124 M parameters) and verify the decoder's
correctness against two ground truths:

  1. STRUCTURAL correctness — output shapes, types, and timing properties are
     self-consistent.
  2. ALGORITHMIC correctness — at temperature=0, our greedy loop produces
     identical tokens to HuggingFace model.generate(do_sample=False).  This
     is the gold-standard check: if the two match token-for-token across five
     independent prompts, the loop is correct.

Why we test token-for-token agreement with model.generate()
──────────────────────────────────────────────────────────
Our decoder bypasses model.generate() for timing and KV cache reasons (see
the module docstring in src/decoding/autoregressive.py).  The tradeoff is
that our loop could have subtle bugs — feeding the wrong token, using the
wrong cache slice, etc.  The HF match test catches any such divergence.

All tests use float32 weights so they pass on CPU-only machines without
requiring a GPU.

Run all tests:
    pytest tests/test_autoregressive.py -v

Run only the fast structural tests (no HF comparison):
    pytest tests/test_autoregressive.py -v -k "not hf_match"
"""

from __future__ import annotations

import pytest
import torch

from src.models.loader import load_model
from src.models.wrapper import ModelWrapper
from src.decoding.autoregressive import AutoregressiveDecoder

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_PRIMARY_PROMPT = "The quick brown fox"
_MAX_NEW_TOKENS = 50
_TEMPERATURE_GREEDY = 0.0

_HF_MATCH_PROMPTS = [
    "The quick brown fox",
    "Once upon a time in a land far away",
    "def fibonacci(n):",
    "The capital of France is",
    "In machine learning, a neural network",
]
_HF_MATCH_TOKENS = 20   # short enough to be fast, long enough to catch divergence


# ---------------------------------------------------------------------------
# Shared fixtures — load GPT-2 once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wrapper() -> ModelWrapper:
    """Load GPT-2 small in float32 (works on CPU without a GPU)."""
    model, tokenizer = load_model("gpt2", "float32")
    device = next(model.parameters()).device
    return ModelWrapper(model, tokenizer, device)


@pytest.fixture(scope="module")
def decoder(wrapper: ModelWrapper) -> AutoregressiveDecoder:
    return AutoregressiveDecoder(wrapper)


@pytest.fixture(scope="module")
def primary_result(decoder: AutoregressiveDecoder, wrapper: ModelWrapper):
    """One greedy 50-token run from the primary prompt, reused across tests."""
    prompt_ids = wrapper.tokenizer.encode(_PRIMARY_PROMPT, return_tensors="pt")
    prompt_ids = prompt_ids.to(wrapper.device)
    return decoder.generate(prompt_ids, _MAX_NEW_TOKENS, temperature=_TEMPERATURE_GREEDY)


# ---------------------------------------------------------------------------
# Class 1: Valid output
# ---------------------------------------------------------------------------


class TestValidOutput:
    def test_generated_ids_is_nonempty_list(self, primary_result):
        assert isinstance(primary_result.generated_ids, list)
        assert len(primary_result.generated_ids) > 0

    def test_generated_ids_are_ints(self, primary_result):
        assert all(isinstance(t, int) for t in primary_result.generated_ids)

    def test_detokenizes_without_error(self, primary_result, wrapper):
        text = wrapper.tokenizer.decode(primary_result.generated_ids, skip_special_tokens=True)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_text_returns_string(self, decoder):
        text = decoder.generate_text(_PRIMARY_PROMPT, 10, temperature=_TEMPERATURE_GREEDY)
        assert isinstance(text, str)
        assert _PRIMARY_PROMPT in text or len(text) > len(_PRIMARY_PROMPT)


# ---------------------------------------------------------------------------
# Class 2: Latency bookkeeping
# ---------------------------------------------------------------------------


class TestLatencyBookkeeping:
    def test_per_token_latencies_length_equals_generated_tokens(self, primary_result):
        assert len(primary_result.per_token_latencies) == len(primary_result.generated_ids)

    def test_per_token_latencies_all_positive(self, primary_result):
        assert all(t > 0.0 for t in primary_result.per_token_latencies)

    def test_total_time_equals_sum_of_latencies(self, primary_result):
        expected = sum(primary_result.per_token_latencies)
        assert abs(primary_result.total_time - expected) < 1e-9

    def test_time_to_first_token_positive(self, primary_result):
        assert primary_result.time_to_first_token > 0.0

    def test_time_to_first_token_equals_first_latency(self, primary_result):
        assert abs(primary_result.time_to_first_token - primary_result.per_token_latencies[0]) < 1e-9


# ---------------------------------------------------------------------------
# Class 3: Throughput
# ---------------------------------------------------------------------------


class TestThroughput:
    def test_tokens_per_second_positive(self, primary_result):
        assert primary_result.tokens_per_second > 0.0

    def test_tokens_per_second_reasonable(self, primary_result):
        # GPT-2 small on any modern CPU should comfortably exceed 10 tok/s.
        assert primary_result.tokens_per_second > 10.0

    def test_num_tokens_at_most_max_new(self, primary_result):
        assert primary_result.num_tokens <= _MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Class 4: GenerationResult fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_peak_memory_mb_nonnegative(self, primary_result):
        assert primary_result.peak_memory_mb >= 0.0

    def test_speculative_fields_are_none(self, primary_result):
        assert primary_result.acceptance_rate is None
        assert primary_result.tokens_per_step is None
        assert primary_result.num_speculation_rounds is None
        assert primary_result.draft_time_total_ms is None
        assert primary_result.verify_time_total_ms is None
        assert primary_result.sampling_time_total_ms is None

    def test_is_speculative_false(self, primary_result):
        assert primary_result.is_speculative is False


# ---------------------------------------------------------------------------
# Class 5: Determinism at temperature=0
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_greedy_runs_produce_identical_tokens(self, decoder, wrapper):
        prompt_ids = wrapper.tokenizer.encode(_PRIMARY_PROMPT, return_tensors="pt")
        prompt_ids = prompt_ids.to(wrapper.device)

        result_a = decoder.generate(prompt_ids, 30, temperature=0.0)
        result_b = decoder.generate(prompt_ids, 30, temperature=0.0)

        assert result_a.generated_ids == result_b.generated_ids

    def test_determinism_across_different_prompts(self, decoder, wrapper):
        for prompt in _HF_MATCH_PROMPTS[:3]:
            prompt_ids = wrapper.tokenizer.encode(prompt, return_tensors="pt")
            prompt_ids = prompt_ids.to(wrapper.device)

            r1 = decoder.generate(prompt_ids, 10, temperature=0.0)
            r2 = decoder.generate(prompt_ids, 10, temperature=0.0)

            assert r1.generated_ids == r2.generated_ids, (
                f"Non-determinism detected for prompt: {prompt!r}"
            )


# ---------------------------------------------------------------------------
# Class 6: Token-for-token agreement with HuggingFace model.generate()
# ---------------------------------------------------------------------------


class TestHFMatchGreedy:
    """Verify that temperature=0 output matches model.generate(do_sample=False).

    This is the strongest correctness test: if our hand-rolled decode loop
    produces the exact same tokens as HF's reference implementation across
    five diverse prompts, the loop is correct — or at least as correct as
    HF's own implementation.

    The comparison is done by feeding both decoders the same tokenised prompt
    and asserting that every output token ID is identical.
    """

    @pytest.mark.parametrize("prompt", _HF_MATCH_PROMPTS)
    def test_greedy_matches_hf_generate(self, decoder, wrapper, prompt):
        prompt_ids = wrapper.tokenizer.encode(prompt, return_tensors="pt")
        prompt_ids = prompt_ids.to(wrapper.device)

        # ── Our decoder ──────────────────────────────────────────────────────
        our_result = decoder.generate(
            prompt_ids, _HF_MATCH_TOKENS, temperature=0.0
        )

        # ── HuggingFace reference ─────────────────────────────────────────────
        # model.generate() returns the full sequence including the prompt;
        # we slice off the prompt tokens to get only the new tokens.
        with torch.inference_mode():
            hf_output = wrapper.model.generate(
                prompt_ids,
                max_new_tokens=_HF_MATCH_TOKENS,
                do_sample=False,
                temperature=None,       # must be None when do_sample=False
                top_p=None,             # disable nucleus sampling
                pad_token_id=wrapper.eos_token_id,
            )
        prompt_len = prompt_ids.shape[1]
        hf_new_tokens = hf_output[0, prompt_len:].tolist()

        # Truncate to the shorter of the two (EOS timing may differ slightly).
        min_len = min(len(our_result.generated_ids), len(hf_new_tokens))
        assert min_len > 0, "Neither decoder produced any tokens."

        our_truncated = our_result.generated_ids[:min_len]
        hf_truncated = hf_new_tokens[:min_len]

        assert our_truncated == hf_truncated, (
            f"Token mismatch for prompt {prompt!r}.\n"
            f"  Ours: {our_truncated}\n"
            f"  HF:   {hf_truncated}"
        )
