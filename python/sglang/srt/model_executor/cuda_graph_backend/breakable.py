"""BreakableCudaGraphBackend — segment-captured graphs with eager breaks.

Uses ``BreakableCUDAGraph`` / ``BreakableCUDAGraphCapture`` from
``cuda_graph_backend_utils.breakable_cuda_graph``. No torch.compile.

Phase 0 placeholder. Implementation lands in Phase 2 (lifted from
today's ``BreakableCudaGraphRunner._capture_one`` and friends).
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)


class BreakableCudaGraphBackend(BaseCudaGraphBackend):
    pass
