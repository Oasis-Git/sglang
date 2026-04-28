"""Backend interface for CUDA graph capture/replay.

A backend encapsulates *how* the model forward at one shape is captured
into a replayable artifact and how that artifact is invoked. The runner
above this interface is phase-aware (prefill vs decode) but
backend-agnostic — it never branches on backend type.

Today's three implementations:
- ``FullCudaGraphBackend``     — one ``torch.cuda.CUDAGraph`` per shape.
- ``BreakableCudaGraphBackend`` — segmented ``BreakableCUDAGraph`` per shape.
- ``TCPiecewiseCudaGraphBackend`` — torch.compile wraps the model;
  per-shape graphs live inside torch.compile's internal cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseCudaGraphBackend(ABC):
    """Capture/replay protocol for one cuda-graph backend.

    Lifecycle:
        1. ``prepare(runner)`` — one-time setup (e.g. wrap the model with
           torch.compile, install compilation hooks, allocate the pool
           handle).
        2. ``capture_one(shape_key, forward_fn, dummies)`` — record the
           replayable artifact for ``shape_key``. Called once per shape
           during the runner's outer capture loop.
        3. ``replay(shape_key, static_forward_batch, **kwargs)`` — invoke
           the captured artifact for ``shape_key`` with already-populated
           static buffers. May or may not consume ``static_forward_batch``
           depending on backend (Full/Breakable replay against static
           buffers and ignore it; TCPCG dispatches by shape via
           torch.compile and uses it).
        4. ``can_run(forward_batch)`` — backend-level "is this batch
           supported" check. Runner ANDs with phase-level checks.
        5. ``cleanup()`` — release pool, drop captured artifacts.
    """

    @abstractmethod
    def prepare(self, runner) -> None: ...

    @abstractmethod
    def can_run(self, forward_batch: "ForwardBatch") -> bool: ...

    @abstractmethod
    def capture_one(
        self,
        shape_key: Any,
        forward_fn,
        dummies: Optional[Any] = None,
    ) -> None: ...

    @abstractmethod
    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...
