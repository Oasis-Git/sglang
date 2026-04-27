"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (``eager_on_graph`` decorators on attention/mamba layers).

Uses ``BreakableCUDAGraph`` / ``BreakableCUDAGraphCapture`` from
``cuda_graph_backend_utils.breakable_cuda_graph``. No torch.compile.

Phase 2a placeholder declaring the ABC contract. Real extraction lands
in Phase 2c — lifted from today's
``BreakableCudaGraphRunner._capture_one`` and ``_capture_all``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )


class BreakableCudaGraphBackend(BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention/mamba boundaries;
    attention metadata recomputed at replay (outside the captured
    segments).
    """

    captures_attn_metadata = False

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError("BreakableCudaGraphBackend lands in Phase 2c")

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError("BreakableCudaGraphBackend lands in Phase 2c")

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError("BreakableCudaGraphBackend lands in Phase 2c")
