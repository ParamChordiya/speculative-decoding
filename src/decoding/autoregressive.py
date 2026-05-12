"""Autoregressive baseline decoder.

WHY WE BUILD OUR OWN LOOP INSTEAD OF USING model.generate()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model.generate() is a high-level convenience function that hides exactly what
we need to measure.  We build our own loop for three reasons:

1. PER-TOKEN GPU-SYNCHRONISED TIMING
   We need the latency of every individual decode step so we can compute
   per-token latency distributions (p50, p95, p99) and compare them against
   the per-round latencies of the speculative decoder.  model.generate() does
   not expose this — it returns the full output sequence after all tokens are
   done.  We would have to time the whole call, losing the per-token signal.

2. KV CACHE CONTROL SHARED WITH THE SPECULATIVE DECODER
   The speculative decoder (src/decoding/speculative.py) needs to truncate
   the KV cache mid-generation when a draft token is rejected.  Both decoders
   must go through ModelWrapper.forward() / ModelWrapper.truncate_cache() so
   that the KV cache bookkeeping is identical and the benchmarks are comparing
   apples to apples.  model.generate() manages its own internal cache object
   that we cannot manipulate.

3. model.generate() IS A BLACK BOX
   It bundles beam search, repetition penalties, logits processors, and a
   dozen other features we do not want.  Any of those could affect latency
   measurements in ways that are hard to account for.  An explicit loop is
   100 lines of code we understand completely, with no hidden work.

The correctness check in tests/test_autoregressive.py verifies that our
greedy (temperature=0) output matches model.generate(do_sample=False)
token-for-token, so we know the loop is correct.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.wrapper import ModelWrapper
from src.profiling.metrics import GenerationResult
from src.profiling.timer import cuda_sync_time


class AutoregressiveDecoder:
    """Token-by-token autoregressive decoder with explicit KV cache management.

    Each call to :meth:`generate` runs a two-phase loop:

    * **Prefill** — process the full prompt in one forward pass to populate
      the KV cache and get the distribution over the first new token.
    * **Decode** — generate one token at a time using the cached keys/values.

    All timing uses :func:`~src.profiling.timer.cuda_sync_time` so latencies
    reflect actual GPU completion time, not CPU queue time.

    Args:
        model: A :class:`~src.models.wrapper.ModelWrapper` wrapping a loaded
               causal LM.  The wrapper must already be in eval mode on its
               target device.
    """

    def __init__(self, model: ModelWrapper) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Main generation entry-point
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> GenerationResult:
        """Generate up to *max_new_tokens* tokens autoregressively.

        Args:
            prompt_ids:     Token IDs for the input prompt, shape ``[1, seq_len]``.
                            Must already be on the model's device.
            max_new_tokens: Maximum number of new tokens to generate.
                            Generation stops earlier if EOS is produced.
            temperature:    Sampling temperature.  ``0.0`` → greedy argmax.
                            ``> 0`` → sample from ``softmax(logits / temperature)``.

        Returns:
            A fully populated :class:`~src.profiling.metrics.GenerationResult`.
            All speculative-only fields are ``None``.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ── PREFILL ──────────────────────────────────────────────────────────
        #
        # WHAT IS PREFILL?
        # The prompt tokens are all known upfront, so we can process the entire
        # sequence in one parallel forward pass — every token attends to all
        # preceding tokens via causal masking, but all positions are computed
        # simultaneously on the GPU.
        #
        # WHY IS DECODE DIFFERENT?
        # During generation each new token depends on the one we just sampled,
        # which we did not know at the start.  That data dependency forces
        # sequential one-token-at-a-time processing.  With the KV cache we
        # only need to run one row through the weight matrices (a GEMV instead
        # of a GEMM), but we still cannot parallelise across future tokens.
        #
        # TIME-TO-FIRST-TOKEN measures this prefill + first sample step, which
        # is the latency the user experiences before seeing any output.

        t_prefill_start = cuda_sync_time()
        logits, cache = self._model.forward(prompt_ids, use_cache=True)
        # logits shape: [1, prompt_len, vocab_size] — we only need the last position.
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]
        first_token_id = self._sample(next_token_logits, temperature)
        t_prefill_end = cuda_sync_time()

        time_to_first_token = t_prefill_end - t_prefill_start
        generated_ids: list[int] = [first_token_id]
        per_token_latencies: list[float] = [time_to_first_token]

        if first_token_id == self._model.eos_token_id:
            return self._build_result(
                generated_ids, per_token_latencies, time_to_first_token
            )

        # ── DECODE LOOP ──────────────────────────────────────────────────────
        for _ in range(max_new_tokens - 1):
            t_step_start = cuda_sync_time()

            # Feed only the single most-recently generated token.  The KV cache
            # already holds keys/values for every prior position, so the model
            # only needs to compute attention for this one new token.
            token_tensor = torch.tensor(
                [[generated_ids[-1]]], dtype=torch.long, device=self._model.device
            )
            logits, cache = self._model.forward(
                token_tensor, past_key_values=cache, use_cache=True
            )
            # logits shape after a single-token forward: [1, 1, vocab_size]
            next_token_logits = logits[:, 0, :]  # [1, vocab_size]

            # ── TEMPERATURE SAMPLING ──────────────────────────────────────────
            # WHY DOES TEMPERATURE WORK?
            #
            # Before softmax the model produces raw logit scores. The softmax
            # converts them to a probability distribution:
            #
            #   p(x) = exp(z_x / T) / Σ exp(z_j / T)
            #
            # T = 1.0  (neutral)  — standard softmax, model's trained distribution.
            # T → 0    (cold)     — dividing large logits by a tiny number amplifies
            #                       the gap between them.  The highest-scoring token
            #                       approaches probability 1 → greedy / deterministic.
            # T → ∞    (hot)      — all logits / T → 0, so exp(z/T) → 1 for all j,
            #                       softmax → uniform distribution → maximum entropy /
            #                       maximum randomness.
            #
            # Practical effect: T < 1 makes output more focused and repetitive;
            # T > 1 makes it more creative and surprising (and more error-prone).
            next_id = self._sample(next_token_logits, temperature)

            t_step_end = cuda_sync_time()
            per_token_latencies.append(t_step_end - t_step_start)
            generated_ids.append(next_id)

            if next_id == self._model.eos_token_id:
                break

        return self._build_result(generated_ids, per_token_latencies, time_to_first_token)

    # ------------------------------------------------------------------
    # Convenience: text in → text out
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> str:
        """Tokenize *prompt*, generate, and decode the output to a string.

        Returns the full text (prompt + generated continuation).

        Args:
            prompt:         Plain-text prompt string.
            max_new_tokens: Maximum new tokens to generate.
            temperature:    Sampling temperature (0.0 = greedy).

        Returns:
            Decoded string containing the prompt followed by the continuation.
        """
        prompt_ids = self._model.tokenizer.encode(prompt, return_tensors="pt")
        prompt_ids = prompt_ids.to(self._model.device)
        result = self.generate(prompt_ids, max_new_tokens, temperature)
        all_ids = prompt_ids[0].tolist() + result.generated_ids
        return self._model.tokenizer.decode(all_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        """Sample (or argmax) a token ID from a ``[1, vocab_size]`` logit tensor.

        Args:
            logits:      Raw (pre-softmax) logit tensor of shape ``[1, vocab_size]``.
            temperature: Sampling temperature.  ``0.0`` triggers greedy argmax.

        Returns:
            Scalar token ID as a Python ``int``.
        """
        if temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _build_result(
        self,
        generated_ids: list[int],
        per_token_latencies: list[float],
        time_to_first_token: float,
    ) -> GenerationResult:
        """Package raw measurements into a :class:`GenerationResult`."""
        total_time = sum(per_token_latencies)
        peak_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        return GenerationResult(
            generated_ids=generated_ids,
            per_token_latencies=per_token_latencies,
            total_time=total_time,
            peak_memory_mb=peak_bytes / 1024 ** 2,
            time_to_first_token=time_to_first_token,
            # Speculative-only fields — not applicable for the baseline.
            acceptance_rate=None,
            tokens_per_step=None,
            num_speculation_rounds=None,
            draft_time_total_ms=None,
            verify_time_total_ms=None,
            sampling_time_total_ms=None,
        )
