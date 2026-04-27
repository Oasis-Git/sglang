"""Transition shim — see ``model_executor/breakable_cuda_graph/__init__.py``.

The real implementations live in
``model_executor/cuda_graph_backend_utils/breakable_cuda_graph/cuda_utils``
after the cg-refactor relocation.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.cuda_utils import *  # noqa: F401,F403
from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.cuda_utils import (  # noqa: F401
    checkCudaErrors,
)
