"""Capture-mechanism backends for CUDA graphs.

A backend owns *how* a captured artifact is produced and replayed for
one shape; it is phase-agnostic. Runners (``cuda_graph_runner/``) own
*what* data flows in and out.

Public API:
  - ``BaseCudaGraphBackend`` — abstract interface (Phase 2a)
  - ``FullCudaGraphBackend`` — Phase 2b
  - ``BreakableCudaGraphBackend`` — Phase 2c
  - ``TCPiecewiseCudaGraphBackend`` — Phase 2d
"""

from sglang.srt.model_executor.cuda_graph_backend.base import (  # noqa: F401
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.breakable import (  # noqa: F401
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.full import (  # noqa: F401
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.tcpcg import (  # noqa: F401
    TCPiecewiseCudaGraphBackend,
)
