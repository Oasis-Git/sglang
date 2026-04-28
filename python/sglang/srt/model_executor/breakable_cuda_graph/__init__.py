"""Backwards-compat shim — BCG primitives moved to
``sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph``.
This package re-exports them under the old import paths.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
