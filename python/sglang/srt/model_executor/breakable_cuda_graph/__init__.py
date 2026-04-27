"""Transition shim — moved during the cg-refactor.

BCG primitives now live in
``sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph``.
This package re-exports them under the old import paths so unaudited
callers keep working. Removed in Phase 6.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
