"""Transition shim — see ``model_executor/breakable_cuda_graph/__init__.py``.

The real implementations live in
``model_executor/cuda_graph_backend_utils/breakable_cuda_graph/context``
after the cg-refactor relocation.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.context import (  # noqa: F401
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
