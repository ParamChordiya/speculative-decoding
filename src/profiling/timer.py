"""GPU-synchronised timing utilities.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY time.time() AND time.perf_counter() ARE WRONG FOR GPU TIMING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUDA operations are ASYNCHRONOUS.  When you call a PyTorch operation like
torch.mm() or model.forward(), the CPU does not wait for the GPU to finish.
It enqueues the operation in a CUDA stream and returns immediately.

This means:

    t0 = time.perf_counter()
    result = torch.mm(a, b)       # CPU returns ~instantly; GPU is still busy
    t1 = time.perf_counter()
    elapsed = t1 - t0             # measures ~microseconds of CPU queue time,
                                  # NOT the actual GPU compute time

The GPU might still be multiplying matrices microseconds, milliseconds, or
even seconds after t1 is recorded.  The CPU-side timestamp is essentially
meaningless for measuring GPU work.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT torch.cuda.synchronize() DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

torch.cuda.synchronize() blocks the CPU until ALL previously queued GPU
operations on ALL CUDA streams have completed.  With it, you can get a
correct (if slightly conservative) measurement:

    t0 = time.perf_counter()
    result = torch.mm(a, b)
    torch.cuda.synchronize()      # CPU waits here until GPU finishes
    t1 = time.perf_counter()
    elapsed = t1 - t0             # now includes actual GPU compute time

The downside: the synchronize() call itself has overhead — it flushes the
CPU-GPU pipeline and forces a context switch.  For benchmarking a sequence
of kernels this overhead accumulates and distorts the measurement, especially
for short operations.  It also serialises the GPU: any other streams that
could have run concurrently are stalled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUDA EVENTS: WHY THEY ARE MORE PRECISE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A CUDA Event is a timestamp recorded INSIDE the GPU stream, by the GPU
itself, at the exact moment the event is reached in the execution queue.
The timeline looks like this:

    CPU:  [record start event] --> [queue kernels] --> [record end event]
                                                                  |
    GPU:  ... [start event fires] -- [kernels execute] -- [end event fires]

event.record() is non-blocking: the CPU queues a "drop a timestamp here"
instruction and moves on.  The GPU executes it in order with the surrounding
kernels.  start_event.elapsed_time(end_event) measures the GPU-side wall
time between those two timestamps with sub-microsecond precision.

Advantages over synchronize + perf_counter:
  1. No pipeline stall between measurements — other GPU work can continue.
  2. Measures GPU time, not CPU time, so CPU scheduling jitter is excluded.
  3. Reads of elapsed_time() only require a synchronize on the END event,
     not the entire device.

One synchronize is still needed before reading elapsed_time() — to ensure
the end event has actually been recorded — but it is scoped to just that
event rather than the whole device.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Type

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA = torch.cuda.is_available()


def _cpu_ms() -> float:
    """Return current wall time in milliseconds (CPU clock)."""
    return time.perf_counter() * 1_000.0


# ---------------------------------------------------------------------------
# 1. Simple synchronise-then-stamp
# ---------------------------------------------------------------------------


def cuda_sync_time() -> float:
    """Return a wall-clock timestamp (seconds) after draining the GPU queue.

    Calls torch.cuda.synchronize() to force the CPU to wait until all queued
    GPU operations complete, then records the CPU clock.  Simple and correct,
    but the synchronize call adds measurable overhead and serialises the device
    — prefer :class:`CUDATimer` for tight benchmarking loops.

    On CPU-only machines the synchronize is skipped and time.perf_counter()
    is returned directly.

    Returns:
        Current time in seconds (same epoch as time.perf_counter()).
    """
    if _CUDA:
        torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# 2. CUDATimer — context manager using CUDA Events
# ---------------------------------------------------------------------------


class CUDATimer:
    """Measure GPU execution time between two points using CUDA Events.

    CUDA Events are timestamps recorded inside the GPU stream by the GPU
    itself, avoiding the CPU-GPU synchronisation overhead of the
    synchronize+perf_counter approach.  The CPU only synchronises once, when
    you read :attr:`elapsed_ms`, to confirm the end event has fired.

    Falls back to CPU timing (time.perf_counter) on machines without CUDA so
    that code using this class works unchanged in CPU-only environments.

    Usage::

        with CUDATimer() as t:
            output = model(input_ids)

        print(f"forward pass: {t.elapsed_ms:.2f} ms")

    Attributes:
        elapsed_ms: GPU-measured time between enter and exit, in milliseconds.
                    Raises RuntimeError if read before the context exits.
    """

    def __init__(self) -> None:
        self._elapsed_ms: float | None = None
        # Allocated here so they can be inspected after the context exits.
        if _CUDA:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._cpu_start: float = 0.0

    def __enter__(self) -> "CUDATimer":
        if _CUDA:
            # record() inserts a timestamp instruction into the current CUDA
            # stream; the GPU will stamp it when it reaches this point in the
            # queue.  Non-blocking on the CPU side.
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if _CUDA:
            self._end_event.record()
            # Synchronise only on this event — waits until the end timestamp
            # has been written by the GPU, without stalling other streams.
            torch.cuda.synchronize()
            # elapsed_time() returns the GPU-measured delta in milliseconds.
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self._elapsed_ms = (time.perf_counter() - self._cpu_start) * 1_000.0

    @property
    def elapsed_ms(self) -> float:
        """GPU-measured elapsed time in milliseconds.

        Raises:
            RuntimeError: if read before the ``with`` block has exited.
        """
        if self._elapsed_ms is None:
            raise RuntimeError(
                "elapsed_ms is not available until the CUDATimer context exits."
            )
        return self._elapsed_ms

    def __repr__(self) -> str:
        if self._elapsed_ms is None:
            return "CUDATimer(running)"
        return f"CUDATimer({self._elapsed_ms:.3f} ms)"


# ---------------------------------------------------------------------------
# 3. CUDATimerCollection — named multi-phase profiler
# ---------------------------------------------------------------------------


class CUDATimerCollection:
    """Manage named timers for profiling multiple phases in a pipeline.

    Typical use in a speculative decoding round::

        timers = CUDATimerCollection()

        timers.start("draft")
        draft_tokens = draft_model.generate(...)
        timers.stop("draft")

        timers.start("verify")
        logits, cache = target_model.forward(draft_tokens)
        timers.stop("verify")

        timers.start("sample")
        accepted = rejection_sample(logits, draft_probs)
        timers.stop("sample")

        print(timers)
        # CUDATimerCollection:
        #   draft  :  1.243 ms
        #   verify :  4.817 ms
        #   sample :  0.091 ms

    Starting a name that is already running raises :class:`RuntimeError`.
    Stopping a name that was never started raises :class:`RuntimeError`.
    Calling :meth:`start` for a name a second time (after it was stopped)
    overwrites the previous record for that name.
    """

    def __init__(self) -> None:
        # Completed timers: name → elapsed_ms
        self._records: dict[str, float] = {}
        # In-flight timers: name → CUDATimer that has been __enter__'d
        self._active: dict[str, CUDATimer] = {}

    # --- control ------------------------------------------------------------

    def start(self, name: str) -> None:
        """Begin timing the named phase.

        Args:
            name: Arbitrary label for this phase (e.g. ``"draft"``,
                  ``"verify"``, ``"sample"``).

        Raises:
            RuntimeError: if *name* is already running.
        """
        if name in self._active:
            raise RuntimeError(
                f"Timer {name!r} is already running. Call stop({name!r}) first."
            )
        timer = CUDATimer()
        timer.__enter__()
        self._active[name] = timer

    def stop(self, name: str) -> float:
        """Stop the named phase and record its elapsed time.

        Args:
            name: Label passed to the corresponding :meth:`start` call.

        Returns:
            Elapsed time for this phase in milliseconds.

        Raises:
            RuntimeError: if *name* was never started.
        """
        if name not in self._active:
            raise RuntimeError(
                f"Timer {name!r} was never started. Call start({name!r}) first."
            )
        timer = self._active.pop(name)
        timer.__exit__(None, None, None)
        self._records[name] = timer.elapsed_ms
        return timer.elapsed_ms

    def reset(self) -> None:
        """Discard all completed and in-flight timer records."""
        self._records.clear()
        self._active.clear()

    # --- querying -----------------------------------------------------------

    def elapsed_ms(self, name: str) -> float:
        """Return the recorded elapsed time for a completed timer.

        Args:
            name: Label of a previously stopped timer.

        Raises:
            KeyError: if *name* has no completed record.
        """
        if name not in self._records:
            raise KeyError(
                f"No completed timer named {name!r}. "
                f"Available: {list(self._records)}"
            )
        return self._records[name]

    def summary(self) -> dict[str, float]:
        """Return a snapshot of all completed timers as ``{name: elapsed_ms}``.

        Returns:
            Shallow copy — safe to modify without affecting the collection.
        """
        return dict(self._records)

    @property
    def names(self) -> list[str]:
        """Names of all completed timers in insertion order."""
        return list(self._records)

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._records:
            return "CUDATimerCollection(empty)"
        width = max(len(n) for n in self._records)
        lines = ["CUDATimerCollection:"]
        for name, ms in self._records.items():
            lines.append(f"  {name:<{width}} : {ms:>8.3f} ms")
        return "\n".join(lines)
