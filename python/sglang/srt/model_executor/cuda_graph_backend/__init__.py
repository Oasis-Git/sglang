"""Capture-mechanism backends for CUDA graphs.

A backend owns *how* a captured artifact is produced and replayed for
one shape; it is phase-agnostic. Runners (``cuda_graph_runner/``) own
*what* data flows in and out.

Public API:
  - ``FullCudaGraphBackend`` — single ``torch.cuda.CUDAGraph`` per shape.
  - ``BreakableCudaGraphBackend`` — segmented capture with eager break
    markers; no torch.compile.
  - ``TCPiecewiseCudaGraphBackend`` — torch.compile-driven piecewise
    capture; FX-splits the model at attention layers.
"""

from sglang.srt.model_executor.cuda_graph_backend.breakable import (  # noqa: F401
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.full import (  # noqa: F401
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.tcpcg import (  # noqa: F401
    TCPiecewiseCudaGraphBackend,
)
