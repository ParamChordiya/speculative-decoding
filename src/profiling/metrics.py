"""Result data structures for autoregressive and speculative decoding benchmarks.

Both decoders return a :class:`GenerationResult`.  Autoregressive runs leave the
speculative-only fields as ``None``; speculative runs populate them.  This single
structure is intentional — it lets every analysis and plotting function work
uniformly over both decoder types without special-casing.

Benchmark runs are identified by a :class:`BenchmarkConfig`, whose
:meth:`~BenchmarkConfig.config_hash` can be used to deduplicate results files
and to group runs by experimental condition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """All measurements from a single generation run.

    Populated by both ``AutoregressiveDecoder`` and ``SpeculativeDecoder``.
    Speculative-only fields (acceptance_rate, etc.) are ``None`` for baseline
    autoregressive runs, so every analysis function can handle both with a
    single code path.

    Attributes:
        generated_ids:           Token IDs produced by the decoder, in order.
        per_token_latencies:     For autoregressive: seconds per generated token.
                                 For speculative:    seconds per speculation round.
                                 Used to compute tail-latency percentiles.
        total_time:              Wall-clock seconds from first forward pass to
                                 last token.  Excludes prompt encoding.
        peak_memory_mb:          Peak GPU memory allocated during generation,
                                 in megabytes (torch.cuda.max_memory_allocated).
                                 0.0 on CPU-only machines.
        time_to_first_token:     Seconds from prompt submission to the first
                                 output token.  Measures prefill + one decode step.

        acceptance_rate:         (Speculative only) Fraction of draft tokens
                                 accepted across all speculation rounds.
                                 In [0, 1]; higher is better.
        tokens_per_step:         (Speculative only) Average number of tokens
                                 produced per speculation round (accepted + 1
                                 bonus/replacement).  Theoretical max = K + 1.
        num_speculation_rounds:  (Speculative only) Total speculation rounds run.
        draft_time_total_ms:     (Speculative only) Cumulative time spent in
                                 all draft model forward passes, in milliseconds.
        verify_time_total_ms:    (Speculative only) Cumulative time spent in
                                 all target model verification passes.
        sampling_time_total_ms:  (Speculative only) Cumulative time spent in
                                 rejection sampling + adjusted distribution math.
    """

    # --- required fields (no defaults) -------------------------------------
    generated_ids: list[int]
    per_token_latencies: list[float]
    total_time: float
    peak_memory_mb: float
    time_to_first_token: float

    # --- speculative-only fields (default None for autoregressive) ---------
    acceptance_rate: Optional[float] = field(default=None)
    tokens_per_step: Optional[float] = field(default=None)
    num_speculation_rounds: Optional[int] = field(default=None)
    draft_time_total_ms: Optional[float] = field(default=None)
    verify_time_total_ms: Optional[float] = field(default=None)
    sampling_time_total_ms: Optional[float] = field(default=None)

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        """Number of tokens generated (excluding the prompt)."""
        return len(self.generated_ids)

    @property
    def tokens_per_second(self) -> float:
        """Average throughput: generated tokens ÷ total wall-clock seconds.

        Returns 0.0 if ``total_time`` is zero (avoids ZeroDivisionError).
        """
        if self.total_time == 0.0:
            return 0.0
        return self.num_tokens / self.total_time

    @property
    def latency_p50(self) -> float:
        """Median per-token (or per-round) latency in seconds."""
        return float(np.percentile(self.per_token_latencies, 50))

    @property
    def latency_p95(self) -> float:
        """95th-percentile per-token (or per-round) latency in seconds.

        Tail latency matters for interactive applications: p95 is the latency
        experienced by 1 in 20 tokens in the worst case.
        """
        return float(np.percentile(self.per_token_latencies, 95))

    @property
    def latency_p99(self) -> float:
        """99th-percentile per-token (or per-round) latency in seconds."""
        return float(np.percentile(self.per_token_latencies, 99))

    @property
    def is_speculative(self) -> bool:
        """True if this result came from a speculative decoding run."""
        return self.acceptance_rate is not None

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all fields and computed properties.

        Computed properties are included under their exact attribute names so
        that a consumer of the JSON does not need to recompute them.  The dict
        is suitable for :func:`json.dumps` without further transformation.
        """
        base = asdict(self)
        base.update(
            {
                "num_tokens": self.num_tokens,
                "tokens_per_second": self.tokens_per_second,
                "latency_p50": self.latency_p50,
                "latency_p95": self.latency_p95,
                "latency_p99": self.latency_p99,
                "is_speculative": self.is_speculative,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationResult":
        """Reconstruct a GenerationResult from a dict produced by :meth:`to_dict`.

        Computed property keys present in the dict are silently ignored —
        they will be recalculated on demand from the stored fields.
        """
        return cls(
            generated_ids=d["generated_ids"],
            per_token_latencies=d["per_token_latencies"],
            total_time=d["total_time"],
            peak_memory_mb=d["peak_memory_mb"],
            time_to_first_token=d["time_to_first_token"],
            acceptance_rate=d.get("acceptance_rate"),
            tokens_per_step=d.get("tokens_per_step"),
            num_speculation_rounds=d.get("num_speculation_rounds"),
            draft_time_total_ms=d.get("draft_time_total_ms"),
            verify_time_total_ms=d.get("verify_time_total_ms"),
            sampling_time_total_ms=d.get("sampling_time_total_ms"),
        )

    def to_json_line(self) -> str:
        """Serialise to a single JSON line suitable for JSONL output files.

        Returns a string ending with ``\\n`` so lines can be appended directly
        to an open file without extra formatting::

            with open("results/run.jsonl", "a") as f:
                f.write(result.to_json_line())
        """
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    def summary(self) -> str:
        """Return a compact one-line human-readable summary.

        Examples::

            # Autoregressive
            AR  | 50 tok | 12.5 tok/s | p50=78ms p95=102ms | mem=1823MB

            # Speculative
            SD  | 50 tok | 28.3 tok/s | p50=31ms p95=44ms  | accept=0.78 rounds=18 | mem=4201MB
        """
        mode = "SD " if self.is_speculative else "AR "
        base = (
            f"{mode} | {self.num_tokens} tok"
            f" | {self.tokens_per_second:.1f} tok/s"
            f" | p50={self.latency_p50 * 1000:.0f}ms"
            f" p95={self.latency_p95 * 1000:.0f}ms"
            f" | mem={self.peak_memory_mb:.0f}MB"
        )
        if self.is_speculative:
            base += (
                f" | accept={self.acceptance_rate:.2f}"
                f" rounds={self.num_speculation_rounds}"
            )
        return base


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Fully describes the conditions of one benchmark run.

    Used alongside :class:`GenerationResult` to record *what* was measured,
    not just *what the outcome was*.  Storing the config with every result
    makes the results files self-describing and allows grouping/filtering
    without a separate experiment log.

    Attributes:
        model_pair_name:  Key into the model registry, e.g. ``"gpt2_dev"`` or
                          ``"tinyllama_llama3"``.
        K:                Number of draft tokens per speculation round.
                          ``None`` for autoregressive baseline runs.
        temperature:      Sampling temperature used during generation.
                          ``0.0`` = greedy.
        max_new_tokens:   Maximum number of new tokens the decoder was allowed
                          to produce.
        prompt_domain:    Domain of the prompt used: ``"code"``,
                          ``"conversation"``, or ``"summarization"``.
        seed:             Random seed used for reproducibility.
    """

    model_pair_name: str
    K: Optional[int]
    temperature: float
    max_new_tokens: int
    prompt_domain: str
    seed: int

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def is_speculative(self) -> bool:
        """True if this config describes a speculative decoding run (K is set)."""
        return self.K is not None

    @property
    def decoder_label(self) -> str:
        """Short label suitable for plot legends: ``"AR"`` or ``"SD-K4"``."""
        return "AR" if self.K is None else f"SD-K{self.K}"

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all config fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkConfig":
        """Reconstruct a BenchmarkConfig from a dict produced by :meth:`to_dict`."""
        return cls(
            model_pair_name=d["model_pair_name"],
            K=d["K"],
            temperature=d["temperature"],
            max_new_tokens=d["max_new_tokens"],
            prompt_domain=d["prompt_domain"],
            seed=d["seed"],
        )

    def config_hash(self) -> str:
        """Return a short deterministic hex hash that uniquely identifies this config.

        Computed as the first 12 hex characters of the SHA-256 hash of the
        sorted JSON representation of :meth:`to_dict`.  Deterministic across
        runs and Python versions.

        Use cases:
          - Deduplicating results: skip a run if a result file with this hash
            already exists.
          - Grouping: collect all ``GenerationResult`` objects with the same
            config hash.

        Returns:
            12-character lowercase hex string, e.g. ``"a3f9c1b20d44"``.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return (
            f"BenchmarkConfig("
            f"pair={self.model_pair_name!r}, "
            f"decoder={self.decoder_label}, "
            f"domain={self.prompt_domain!r}, "
            f"temp={self.temperature}, "
            f"max_new={self.max_new_tokens}, "
            f"seed={self.seed})"
        )
