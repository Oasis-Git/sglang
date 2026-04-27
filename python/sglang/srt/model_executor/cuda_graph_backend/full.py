"""FullCudaGraphBackend — captures the entire model forward as one
``torch.cuda.CUDAGraph`` per shape.

Phase 0 placeholder. Implementation lands in Phase 2 (lifted from
today's ``CudaGraphRunner.capture_one_batch_size``).
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)


class FullCudaGraphBackend(BaseCudaGraphBackend):
    pass
