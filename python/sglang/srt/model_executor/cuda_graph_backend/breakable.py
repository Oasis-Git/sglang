"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (``eager_on_graph`` decorators on attention/mamba layers).

Uses ``BreakableCUDAGraph`` / ``BreakableCUDAGraphCapture`` from
``cuda_graph_backend_utils.breakable_cuda_graph``. No torch.compile.

Phase 2c — minimal extraction:
  - ``make_graph()`` and ``capture_into(...)`` provide the primitive
    capture machinery as a backend-owned, runner-coupling-free pair.
  - ``legacy.CudaGraphRunner`` delegates to them on the breakable
    env-var path.
  - The ABC methods (``prepare`` / ``capture_one`` / ``replay``) stay
    NotImplementedError until Phase 3 (runner unification) wires the
    backend through the abstract interface.

Lifted from the breakable branch of
``cuda_graph_runner/legacy.py`` ``_create_device_graph`` and
``_capture_graph``. The prefill BCG runner
(``BreakableCudaGraphRunner``) is a separate class and migrates to
this backend in Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )


class BreakableCudaGraphBackend(BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention/mamba boundaries;
    attention metadata is recomputed at replay (outside the captured
    segments).
    """

    captures_attn_metadata = False

    # ------------------------------------------------------------------
    # Phase 2c primitives — direct, no runner-coupling.
    # ------------------------------------------------------------------

    @staticmethod
    def make_graph() -> BreakableCUDAGraph:
        """Allocate an empty breakable CUDA graph object. Mirrors the
        breakable branch of ``CudaGraphRunner._create_device_graph``.

        Raises ``RuntimeError`` on ROCm/HIP, where the underlying
        primitive is not supported (matches legacy behavior).
        """
        if is_hip():
            raise RuntimeError("Breakable CUDA graph is not supported on ROCm/HIP")
        return BreakableCUDAGraph()

    @staticmethod
    def capture_into(
        graph: BreakableCUDAGraph,
        pool: Any,
        stream: Any,
        run_once_fn: Callable[[], Any],
        *,
        debug_eager: bool = False,
        memory_saver_adapter: Optional[Any] = None,
    ) -> Any:
        """Run ``run_once_fn`` under a ``BreakableCUDAGraphCapture``
        context bound to ``graph``. Returns whatever ``run_once_fn``
        returned.

        ``debug_eager`` corresponds to ``--debug-cuda-graph``: wraps the
        closure with ``eager_on_graph(True)`` so every op runs eagerly
        while still going through the capture/replay path.

        Raises ``NotImplementedError`` if a memory-saver adapter is
        enabled — breakable capture is not compatible with it (legacy
        behavior).
        """
        if memory_saver_adapter is not None and memory_saver_adapter.enabled:
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )
        captured_fn = eager_on_graph(True)(run_once_fn) if debug_eager else run_once_fn
        with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool, stream=stream):
            out = captured_fn()
        return out

    # ------------------------------------------------------------------
    # Abstract interface — wired up in Phase 3.
    # ------------------------------------------------------------------

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError(
            "BreakableCudaGraphBackend.prepare lands in Phase 3 (runner unification)"
        )

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError(
            "BreakableCudaGraphBackend.capture_one lands in Phase 3 (runner unification)"
        )

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError(
            "BreakableCudaGraphBackend.replay lands in Phase 3 (runner unification)"
        )
