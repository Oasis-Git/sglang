"""TCPiecewiseCudaGraphBackend — torch.compile-based piecewise CUDA graph.

Uses ``CompilationConfig`` and the FX/inductor pipeline from
``sglang.srt.compilation``. Produces piecewise graphs by FX-splitting
the model forward at attention layers.

Phase 0 placeholder. Implementation lands in Phase 2 (lifted from
today's ``PiecewiseCudaGraphRunner.capture_one_batch_size``).
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)


class TCPiecewiseCudaGraphBackend(BaseCudaGraphBackend):
    pass
