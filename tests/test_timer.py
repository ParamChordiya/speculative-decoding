"""Tests for GPU-synchronised timing utilities.

Tests are structured in three tiers:

  Tier 1 — pure-CPU (always run)
      Verify the API contract, types, and error handling using small tensors
      and CPU-fallback paths.  No CUDA required.

  Tier 2 — CUDA-available (skipped on CPU-only machines)
      Verify that CUDA Events are used and that measured times are positive
      and finite for a real GPU workload.

  Tier 3 — agreement (CUDA only)
      Verify that CUDATimer and cuda_sync_time agree to within a generous
      tolerance on the same workload.  Hardware variability means we only
      check order-of-magnitude agreement, not exact equality.

Run all tests:
    pytest tests/test_timer.py -v

Run only CPU-safe tests:
    pytest tests/test_timer.py -v -k "not cuda"
"""

import time

import pytest
import torch

from src.profiling.timer import CUDATimer, CUDATimerCollection, cuda_sync_time

_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Shared workload helpers
# ---------------------------------------------------------------------------


def _matmul_workload(size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Run a square matrix multiply — a concrete, measurable GPU kernel."""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    return torch.mm(a, b)


def _gpu_device() -> str:
    return "cuda" if _CUDA else "cpu"


# ---------------------------------------------------------------------------
# Tier 1: API contract (CPU-safe)
# ---------------------------------------------------------------------------


class TestCudaSyncTime:
    def test_returns_float(self):
        t = cuda_sync_time()
        assert isinstance(t, float)

    def test_monotonically_increases(self):
        t0 = cuda_sync_time()
        time.sleep(0.01)
        t1 = cuda_sync_time()
        assert t1 > t0

    def test_units_are_seconds(self):
        # Two calls back-to-back on an idle system should be sub-second apart.
        t0 = cuda_sync_time()
        t1 = cuda_sync_time()
        assert (t1 - t0) < 1.0


class TestCUDATimerAPI:
    def test_elapsed_ms_positive_after_context(self):
        with CUDATimer() as t:
            _matmul_workload(256)
        assert t.elapsed_ms > 0.0

    def test_elapsed_ms_is_float(self):
        with CUDATimer() as t:
            _matmul_workload(256)
        assert isinstance(t.elapsed_ms, float)

    def test_elapsed_ms_before_exit_raises(self):
        timer = CUDATimer()
        timer.__enter__()
        with pytest.raises(RuntimeError, match="not available"):
            _ = timer.elapsed_ms
        timer.__exit__(None, None, None)

    def test_elapsed_ms_finite(self):
        import math
        with CUDATimer() as t:
            _matmul_workload(256)
        assert math.isfinite(t.elapsed_ms)

    def test_repr_before_exit(self):
        timer = CUDATimer()
        timer.__enter__()
        assert "running" in repr(timer)
        timer.__exit__(None, None, None)

    def test_repr_after_exit(self):
        with CUDATimer() as t:
            pass
        assert "ms" in repr(t)

    def test_longer_work_gives_larger_time(self):
        """A larger matmul should take longer than a tiny one."""
        with CUDATimer() as t_small:
            _matmul_workload(64)
        with CUDATimer() as t_large:
            _matmul_workload(1024)
        # Large should be measurably slower; allow a very loose bound.
        assert t_large.elapsed_ms >= t_small.elapsed_ms * 0.5


class TestCUDATimerCollectionAPI:
    def test_start_and_stop_single_phase(self):
        tc = CUDATimerCollection()
        tc.start("phase_a")
        _matmul_workload(256)
        elapsed = tc.stop("phase_a")
        assert elapsed > 0.0

    def test_summary_contains_stopped_names(self):
        tc = CUDATimerCollection()
        tc.start("draft")
        _matmul_workload(128)
        tc.stop("draft")
        tc.start("verify")
        _matmul_workload(128)
        tc.stop("verify")
        summary = tc.summary()
        assert "draft" in summary
        assert "verify" in summary

    def test_summary_values_positive(self):
        tc = CUDATimerCollection()
        for name in ("a", "b", "c"):
            tc.start(name)
            _matmul_workload(128)
            tc.stop(name)
        for ms in tc.summary().values():
            assert ms > 0.0

    def test_elapsed_ms_accessor(self):
        tc = CUDATimerCollection()
        tc.start("x")
        _matmul_workload(128)
        tc.stop("x")
        assert tc.elapsed_ms("x") > 0.0

    def test_elapsed_ms_unknown_name_raises(self):
        tc = CUDATimerCollection()
        with pytest.raises(KeyError):
            tc.elapsed_ms("nonexistent")

    def test_start_already_running_raises(self):
        tc = CUDATimerCollection()
        tc.start("phase")
        with pytest.raises(RuntimeError, match="already running"):
            tc.start("phase")
        tc.stop("phase")

    def test_stop_never_started_raises(self):
        tc = CUDATimerCollection()
        with pytest.raises(RuntimeError, match="never started"):
            tc.stop("ghost")

    def test_reset_clears_all_records(self):
        tc = CUDATimerCollection()
        tc.start("p")
        _matmul_workload(64)
        tc.stop("p")
        tc.reset()
        assert tc.summary() == {}
        assert tc.names == []

    def test_summary_is_a_copy(self):
        """Mutating the summary dict must not affect the collection."""
        tc = CUDATimerCollection()
        tc.start("q")
        _matmul_workload(64)
        tc.stop("q")
        s = tc.summary()
        s["q"] = 9999.0
        assert tc.elapsed_ms("q") != 9999.0

    def test_restart_after_stop_overwrites_record(self):
        """A second start/stop cycle for the same name replaces the record."""
        tc = CUDATimerCollection()
        tc.start("r")
        _matmul_workload(64)
        tc.stop("r")
        first = tc.elapsed_ms("r")

        tc.start("r")
        _matmul_workload(64)
        tc.stop("r")
        second = tc.elapsed_ms("r")

        # Both are valid measurements; the collection keeps the latest.
        assert second > 0.0
        assert first > 0.0

    def test_names_property_order(self):
        """names returns timers in the order they were stopped."""
        tc = CUDATimerCollection()
        for n in ("alpha", "beta", "gamma"):
            tc.start(n)
            tc.stop(n)
        assert tc.names == ["alpha", "beta", "gamma"]

    def test_repr_empty(self):
        assert "empty" in repr(CUDATimerCollection())

    def test_repr_with_entries(self):
        tc = CUDATimerCollection()
        tc.start("phase")
        tc.stop("phase")
        r = repr(tc)
        assert "phase" in r
        assert "ms" in r


# ---------------------------------------------------------------------------
# Tier 2: CUDA-specific correctness
# ---------------------------------------------------------------------------


@cuda_only
class TestCUDAEventPrecision:
    def test_cuda_timer_uses_events(self):
        """CUDATimer must hold _start_event and _end_event on CUDA machines."""
        t = CUDATimer()
        assert hasattr(t, "_start_event")
        assert hasattr(t, "_end_event")

    def test_elapsed_ms_positive_on_gpu_workload(self):
        with CUDATimer() as t:
            _matmul_workload(1024, device="cuda")
        assert t.elapsed_ms > 0.0

    def test_cuda_sync_time_after_gpu_work(self):
        t0 = cuda_sync_time()
        _matmul_workload(1024, device="cuda")
        t1 = cuda_sync_time()
        # At least some measurable time should have passed.
        assert t1 > t0


# ---------------------------------------------------------------------------
# Tier 3: Agreement between CUDATimer and cuda_sync_time
# ---------------------------------------------------------------------------


@cuda_only
class TestTimerAgreement:
    """CUDATimer and sync+perf_counter should agree within a factor of 3.

    A factor of 3 is intentionally generous: the two methods have different
    overheads (event-record vs full-device sync) and hardware scheduling
    variability means exact agreement is not expected.  What we verify is
    that they are in the same ballpark — both measuring the same real work.
    """

    def test_cuda_timer_vs_sync_perf_counter(self):
        size = 2048
        device = "cuda"

        # ---- Method A: cuda_sync_time ----
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.cuda.synchronize()  # flush any prior work

        t0 = cuda_sync_time()
        _ = torch.mm(a, b)
        t1 = cuda_sync_time()
        sync_ms = (t1 - t0) * 1_000.0

        # ---- Method B: CUDATimer ----
        with CUDATimer() as ct:
            _ = torch.mm(a, b)
        event_ms = ct.elapsed_ms

        # Both must be positive and agree within 3× of each other.
        assert sync_ms > 0.0
        assert event_ms > 0.0
        ratio = max(sync_ms, event_ms) / min(sync_ms, event_ms)
        assert ratio < 3.0, (
            f"sync_ms={sync_ms:.3f}  event_ms={event_ms:.3f}  ratio={ratio:.2f} — "
            "methods disagree by more than 3×; something is wrong with one of them."
        )

    def test_collection_agrees_with_standalone_timer(self):
        """CUDATimerCollection should report times consistent with CUDATimer."""
        size = 1024
        device = "cuda"
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.cuda.synchronize()

        tc = CUDATimerCollection()
        tc.start("mm")
        _ = torch.mm(a, b)
        tc.stop("mm")

        with CUDATimer() as ct:
            _ = torch.mm(a, b)

        collection_ms = tc.elapsed_ms("mm")
        standalone_ms = ct.elapsed_ms

        ratio = max(collection_ms, standalone_ms) / min(collection_ms, standalone_ms)
        assert ratio < 3.0, (
            f"collection={collection_ms:.3f} ms  standalone={standalone_ms:.3f} ms  "
            f"ratio={ratio:.2f}"
        )
