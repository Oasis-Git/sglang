"""FullCudaGraphBackend — captures the entire model forward as one
``torch.cuda.CUDAGraph`` per shape.

Phase 2a placeholder declaring the ABC contract. Real extraction lands
in Phase 2b — lifted from today's ``CudaGraphRunner.capture_one_batch_size``
in ``cuda_graph_runner.legacy``.
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


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One ``torch.cuda.CUDAGraph`` per shape; attention metadata captured
    inside the graph.
    """

    captures_attn_metadata = True

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError("FullCudaGraphBackend lands in Phase 2b")

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError("FullCudaGraphBackend lands in Phase 2b")

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError("FullCudaGraphBackend lands in Phase 2b")
