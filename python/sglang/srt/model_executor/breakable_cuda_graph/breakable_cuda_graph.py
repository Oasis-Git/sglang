"""Backwards-compat shim — see
``model_executor/breakable_cuda_graph/__init__.py``.

The real implementations live in
``model_executor/cuda_graph_backend_utils/breakable_cuda_graph/breakable_cuda_graph``.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.breakable_cuda_graph import *  # noqa: F401,F403
from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    _copy_output,
    break_graph,
    eager_on_graph,
    get_current_stream,
)
