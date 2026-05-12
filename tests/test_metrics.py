"""Tests for GenerationResult and BenchmarkConfig data structures.

Covers:
  - Computed properties (tokens_per_second, latency percentiles)
  - Speculative vs autoregressive field handling
  - JSON serialisation and round-trip fidelity
  - JSONL output format
  - BenchmarkConfig.config_hash determinism and collision resistance
  - Edge cases (single token, zero time, all-None speculative fields)
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.profiling.metrics import BenchmarkConfig, GenerationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Latencies chosen so np.percentile values are exact and easy to verify:
#   [0.1, 0.2, 0.3, 0.4, 0.5]
#   p50 = 0.3
#   p95 = 0.48   (linear interp: idx=3.8 → 0.4 + 0.8*(0.5-0.4))
#   p99 = 0.496  (linear interp: idx=3.96 → 0.4 + 0.96*(0.5-0.4))
_LATENCIES = [0.1, 0.2, 0.3, 0.4, 0.5]
_P50 = 0.3
_P95 = 0.48
_P99 = 0.496


@pytest.fixture
def ar_result() -> GenerationResult:
    """A typical autoregressive GenerationResult (5 tokens, 1.0 s total)."""
    return GenerationResult(
        generated_ids=[10, 20, 30, 40, 50],
        per_token_latencies=_LATENCIES,
        total_time=1.0,
        peak_memory_mb=1024.0,
        time_to_first_token=0.1,
    )


@pytest.fixture
def sd_result() -> GenerationResult:
    """A typical speculative GenerationResult with all optional fields set."""
    return GenerationResult(
        generated_ids=[10, 20, 30, 40, 50],
        per_token_latencies=_LATENCIES,
        total_time=0.5,
        peak_memory_mb=4096.0,
        time_to_first_token=0.08,
        acceptance_rate=0.78,
        tokens_per_step=4.12,
        num_speculation_rounds=18,
        draft_time_total_ms=120.5,
        verify_time_total_ms=310.2,
        sampling_time_total_ms=9.3,
    )


@pytest.fixture
def config_ar() -> BenchmarkConfig:
    return BenchmarkConfig(
        model_pair_name="gpt2_dev",
        K=None,
        temperature=0.0,
        max_new_tokens=128,
        prompt_domain="code",
        seed=42,
    )


@pytest.fixture
def config_sd() -> BenchmarkConfig:
    return BenchmarkConfig(
        model_pair_name="gpt2_dev",
        K=4,
        temperature=0.0,
        max_new_tokens=128,
        prompt_domain="code",
        seed=42,
    )


# ---------------------------------------------------------------------------
# GenerationResult — basic fields
# ---------------------------------------------------------------------------


class TestGenerationResultFields:
    def test_num_tokens(self, ar_result):
        assert ar_result.num_tokens == 5

    def test_generated_ids_stored(self, ar_result):
        assert ar_result.generated_ids == [10, 20, 30, 40, 50]

    def test_total_time_stored(self, ar_result):
        assert ar_result.total_time == 1.0

    def test_peak_memory_stored(self, ar_result):
        assert ar_result.peak_memory_mb == 1024.0

    def test_time_to_first_token_stored(self, ar_result):
        assert ar_result.time_to_first_token == 0.1

    def test_speculative_fields_none_for_ar(self, ar_result):
        assert ar_result.acceptance_rate is None
        assert ar_result.tokens_per_step is None
        assert ar_result.num_speculation_rounds is None
        assert ar_result.draft_time_total_ms is None
        assert ar_result.verify_time_total_ms is None
        assert ar_result.sampling_time_total_ms is None

    def test_is_speculative_false_for_ar(self, ar_result):
        assert ar_result.is_speculative is False

    def test_is_speculative_true_for_sd(self, sd_result):
        assert sd_result.is_speculative is True

    def test_speculative_fields_populated(self, sd_result):
        assert sd_result.acceptance_rate == pytest.approx(0.78)
        assert sd_result.tokens_per_step == pytest.approx(4.12)
        assert sd_result.num_speculation_rounds == 18
        assert sd_result.draft_time_total_ms == pytest.approx(120.5)
        assert sd_result.verify_time_total_ms == pytest.approx(310.2)
        assert sd_result.sampling_time_total_ms == pytest.approx(9.3)


# ---------------------------------------------------------------------------
# GenerationResult — computed properties
# ---------------------------------------------------------------------------


class TestComputedProperties:
    def test_tokens_per_second(self, ar_result):
        # 5 tokens / 1.0 s = 5.0
        assert ar_result.tokens_per_second == pytest.approx(5.0)

    def test_tokens_per_second_faster_run(self, sd_result):
        # 5 tokens / 0.5 s = 10.0
        assert sd_result.tokens_per_second == pytest.approx(10.0)

    def test_tokens_per_second_zero_time(self):
        r = GenerationResult(
            generated_ids=[1, 2, 3],
            per_token_latencies=[0.0, 0.0, 0.0],
            total_time=0.0,
            peak_memory_mb=0.0,
            time_to_first_token=0.0,
        )
        assert r.tokens_per_second == 0.0

    def test_latency_p50(self, ar_result):
        assert ar_result.latency_p50 == pytest.approx(_P50)

    def test_latency_p95(self, ar_result):
        assert ar_result.latency_p95 == pytest.approx(_P95)

    def test_latency_p99(self, ar_result):
        assert ar_result.latency_p99 == pytest.approx(_P99)

    def test_latency_p50_matches_numpy(self, ar_result):
        expected = float(np.percentile(_LATENCIES, 50))
        assert ar_result.latency_p50 == pytest.approx(expected)

    def test_latency_p95_matches_numpy(self, ar_result):
        expected = float(np.percentile(_LATENCIES, 95))
        assert ar_result.latency_p95 == pytest.approx(expected)

    def test_latency_p99_matches_numpy(self, ar_result):
        expected = float(np.percentile(_LATENCIES, 99))
        assert ar_result.latency_p99 == pytest.approx(expected)

    def test_percentiles_ordered(self, ar_result):
        assert ar_result.latency_p50 <= ar_result.latency_p95 <= ar_result.latency_p99

    def test_latency_p50_single_token(self):
        r = GenerationResult(
            generated_ids=[7],
            per_token_latencies=[0.25],
            total_time=0.25,
            peak_memory_mb=0.0,
            time_to_first_token=0.25,
        )
        assert r.latency_p50 == pytest.approx(0.25)
        assert r.latency_p95 == pytest.approx(0.25)
        assert r.latency_p99 == pytest.approx(0.25)

    def test_computed_properties_are_floats(self, ar_result):
        assert isinstance(ar_result.tokens_per_second, float)
        assert isinstance(ar_result.latency_p50, float)
        assert isinstance(ar_result.latency_p95, float)
        assert isinstance(ar_result.latency_p99, float)

    def test_computed_properties_are_finite(self, ar_result):
        assert math.isfinite(ar_result.tokens_per_second)
        assert math.isfinite(ar_result.latency_p50)
        assert math.isfinite(ar_result.latency_p95)
        assert math.isfinite(ar_result.latency_p99)


# ---------------------------------------------------------------------------
# GenerationResult — to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_returns_dict(self, ar_result):
        assert isinstance(ar_result.to_dict(), dict)

    def test_stored_fields_present(self, ar_result):
        d = ar_result.to_dict()
        for key in (
            "generated_ids", "per_token_latencies", "total_time",
            "peak_memory_mb", "time_to_first_token",
        ):
            assert key in d, f"missing key: {key}"

    def test_computed_properties_included(self, ar_result):
        d = ar_result.to_dict()
        for key in ("num_tokens", "tokens_per_second", "latency_p50", "latency_p95", "latency_p99", "is_speculative"):
            assert key in d, f"missing computed key: {key}"

    def test_computed_values_correct(self, ar_result):
        d = ar_result.to_dict()
        assert d["num_tokens"] == 5
        assert d["tokens_per_second"] == pytest.approx(5.0)
        assert d["latency_p50"] == pytest.approx(_P50)
        assert d["latency_p95"] == pytest.approx(_P95)
        assert d["latency_p99"] == pytest.approx(_P99)
        assert d["is_speculative"] is False

    def test_optional_fields_none_for_ar(self, ar_result):
        d = ar_result.to_dict()
        assert d["acceptance_rate"] is None
        assert d["tokens_per_step"] is None
        assert d["num_speculation_rounds"] is None

    def test_optional_fields_populated_for_sd(self, sd_result):
        d = sd_result.to_dict()
        assert d["acceptance_rate"] == pytest.approx(0.78)
        assert d["num_speculation_rounds"] == 18

    def test_dict_is_json_serialisable(self, ar_result):
        d = ar_result.to_dict()
        serialised = json.dumps(d)  # must not raise
        assert isinstance(serialised, str)

    def test_dict_is_json_serialisable_sd(self, sd_result):
        json.dumps(sd_result.to_dict())


# ---------------------------------------------------------------------------
# GenerationResult — JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def _roundtrip(self, result: GenerationResult) -> GenerationResult:
        return GenerationResult.from_dict(json.loads(json.dumps(result.to_dict())))

    def test_ar_roundtrip_generated_ids(self, ar_result):
        rt = self._roundtrip(ar_result)
        assert rt.generated_ids == ar_result.generated_ids

    def test_ar_roundtrip_latencies(self, ar_result):
        rt = self._roundtrip(ar_result)
        assert rt.per_token_latencies == pytest.approx(ar_result.per_token_latencies)

    def test_ar_roundtrip_total_time(self, ar_result):
        rt = self._roundtrip(ar_result)
        assert rt.total_time == pytest.approx(ar_result.total_time)

    def test_ar_roundtrip_optional_fields_none(self, ar_result):
        rt = self._roundtrip(ar_result)
        assert rt.acceptance_rate is None
        assert rt.num_speculation_rounds is None

    def test_ar_roundtrip_computed_properties_unchanged(self, ar_result):
        rt = self._roundtrip(ar_result)
        assert rt.tokens_per_second == pytest.approx(ar_result.tokens_per_second)
        assert rt.latency_p50 == pytest.approx(ar_result.latency_p50)
        assert rt.latency_p95 == pytest.approx(ar_result.latency_p95)

    def test_sd_roundtrip_speculative_fields(self, sd_result):
        rt = self._roundtrip(sd_result)
        assert rt.acceptance_rate == pytest.approx(sd_result.acceptance_rate)
        assert rt.tokens_per_step == pytest.approx(sd_result.tokens_per_step)
        assert rt.num_speculation_rounds == sd_result.num_speculation_rounds
        assert rt.draft_time_total_ms == pytest.approx(sd_result.draft_time_total_ms)
        assert rt.verify_time_total_ms == pytest.approx(sd_result.verify_time_total_ms)
        assert rt.sampling_time_total_ms == pytest.approx(sd_result.sampling_time_total_ms)

    def test_sd_roundtrip_is_speculative_preserved(self, sd_result):
        rt = self._roundtrip(sd_result)
        assert rt.is_speculative is True


# ---------------------------------------------------------------------------
# GenerationResult — JSONL output
# ---------------------------------------------------------------------------


class TestJsonLine:
    def test_ends_with_newline(self, ar_result):
        assert ar_result.to_json_line().endswith("\n")

    def test_is_valid_json(self, ar_result):
        line = ar_result.to_json_line()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_no_embedded_newlines(self, ar_result):
        # A JSONL line must be a single line — no internal newlines.
        content = ar_result.to_json_line().rstrip("\n")
        assert "\n" not in content

    def test_multiple_lines_parseable(self, ar_result, sd_result):
        blob = ar_result.to_json_line() + sd_result.to_json_line()
        lines = [l for l in blob.splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            json.loads(line)


# ---------------------------------------------------------------------------
# GenerationResult — summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_is_string(self, ar_result):
        assert isinstance(ar_result.summary(), str)

    def test_ar_summary_contains_mode(self, ar_result):
        assert "AR" in ar_result.summary()

    def test_sd_summary_contains_mode(self, sd_result):
        assert "SD" in sd_result.summary()

    def test_summary_contains_token_count(self, ar_result):
        assert "5" in ar_result.summary()

    def test_sd_summary_contains_acceptance(self, sd_result):
        assert "accept" in sd_result.summary()

    def test_sd_summary_contains_rounds(self, sd_result):
        assert "18" in sd_result.summary()


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_is_speculative_false_when_k_none(self, config_ar):
        assert config_ar.is_speculative is False

    def test_is_speculative_true_when_k_set(self, config_sd):
        assert config_sd.is_speculative is True

    def test_decoder_label_ar(self, config_ar):
        assert config_ar.decoder_label == "AR"

    def test_decoder_label_sd(self, config_sd):
        assert config_sd.decoder_label == "SD-K4"

    def test_to_dict_contains_all_fields(self, config_ar):
        d = config_ar.to_dict()
        for key in ("model_pair_name", "K", "temperature", "max_new_tokens", "prompt_domain", "seed"):
            assert key in d

    def test_to_dict_values(self, config_sd):
        d = config_sd.to_dict()
        assert d["model_pair_name"] == "gpt2_dev"
        assert d["K"] == 4
        assert d["temperature"] == 0.0
        assert d["max_new_tokens"] == 128
        assert d["prompt_domain"] == "code"
        assert d["seed"] == 42

    def test_to_dict_k_none_for_ar(self, config_ar):
        assert config_ar.to_dict()["K"] is None

    def test_repr_contains_pair(self, config_ar):
        assert "gpt2_dev" in repr(config_ar)

    def test_repr_contains_decoder_label(self, config_sd):
        assert "SD-K4" in repr(config_sd)


class TestBenchmarkConfigRoundTrip:
    def _roundtrip(self, cfg: BenchmarkConfig) -> BenchmarkConfig:
        return BenchmarkConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))

    def test_ar_roundtrip(self, config_ar):
        rt = self._roundtrip(config_ar)
        assert rt.model_pair_name == config_ar.model_pair_name
        assert rt.K is None
        assert rt.temperature == config_ar.temperature
        assert rt.seed == config_ar.seed

    def test_sd_roundtrip(self, config_sd):
        rt = self._roundtrip(config_sd)
        assert rt.K == 4
        assert rt.prompt_domain == "code"


class TestConfigHash:
    def test_returns_string(self, config_ar):
        assert isinstance(config_ar.config_hash(), str)

    def test_returns_12_hex_chars(self, config_ar):
        h = config_ar.config_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, config_ar):
        assert config_ar.config_hash() == config_ar.config_hash()

    def test_same_config_same_hash(self):
        c1 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 42)
        c2 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 42)
        assert c1.config_hash() == c2.config_hash()

    def test_different_k_different_hash(self):
        c1 = BenchmarkConfig("gpt2_dev", 2, 0.0, 128, "code", 42)
        c2 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 42)
        assert c1.config_hash() != c2.config_hash()

    def test_different_pair_different_hash(self):
        c1 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 42)
        c2 = BenchmarkConfig("tinyllama_llama3", 4, 0.0, 128, "code", 42)
        assert c1.config_hash() != c2.config_hash()

    def test_different_seed_different_hash(self):
        c1 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 42)
        c2 = BenchmarkConfig("gpt2_dev", 4, 0.0, 128, "code", 99)
        assert c1.config_hash() != c2.config_hash()

    def test_ar_and_sd_different_hash(self, config_ar, config_sd):
        # K=None vs K=4 must produce different hashes.
        assert config_ar.config_hash() != config_sd.config_hash()
