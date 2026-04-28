"""Piecewise CUDA graph utilities — shared between BCG and tcpcg backends.

Public API:
  - ``is_in_piecewise_cuda_graph()`` — true while inside any piecewise capture.
  - ``enable_piecewise_cuda_graph()`` — context manager that toggles the flag.
  - ``ForwardContext`` + ``set_forward_context`` + ``get_forward_context``.
  - ``PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG``.

The torch.compile-warmup flag (``is_in_torch_compile_warmup``) lives in
``sglang.srt.compilation.compile_phase`` — it is torch.compile-internal,
not piecewise-shared.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.piecewise_cuda_graph.context_manager import (  # noqa: F401
    PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
    ForwardContext,
    enable_piecewise_cuda_graph,
    get_forward_context,
    is_in_piecewise_cuda_graph,
    set_forward_context,
)
