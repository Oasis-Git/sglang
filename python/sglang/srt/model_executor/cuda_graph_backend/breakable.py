"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (``eager_on_graph`` decorators on attention / mamba layers).

Uses ``BreakableCUDAGraph`` / ``BreakableCUDAGraphCapture`` from
``cuda_graph_backend_utils.breakable_cuda_graph``. No torch.compile.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.utils import is_hip


class BreakableCudaGraphBackend:
    """Segmented capture: graphs break at attention/mamba boundaries;
    attention metadata is recomputed at replay (outside the captured
    segments).
    """

    @staticmethod
    def make_graph() -> BreakableCUDAGraph:
        """Allocate an empty breakable CUDA graph object.

        Raises ``RuntimeError`` on ROCm/HIP, where the underlying
        primitive is not supported.
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
        enabled — breakable capture is not compatible with it.
        """
        if memory_saver_adapter is not None and memory_saver_adapter.enabled:
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )
        captured_fn = eager_on_graph(True)(run_once_fn) if debug_eager else run_once_fn
        with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool, stream=stream):
            out = captured_fn()
        return out
