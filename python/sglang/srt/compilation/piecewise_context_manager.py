"""Transition shim — moved during the cg-refactor.

The CUDA-graph-capture flag and ``ForwardContext`` propagation now live
in ``sglang.srt.model_executor.cuda_graph_backend_utils.piecewise_cuda_graph.context_manager``.
The torch.compile-warmup flag and capture-stream context now live in
``sglang.srt.compilation.compile_phase``.

This module re-exports everything under the **old** names so unaudited
import sites keep working. Audited sites import from the new locations
directly. Removed in Phase 6 of the refactor.

Symbol rename map:

    is_in_piecewise_cuda_graph        -> is_in_cuda_graph_capture
    enable_piecewise_cuda_graph       -> enable_cuda_graph_capture
    is_in_pcg_torch_compile           -> is_in_torch_compile_warmup
    enable_piecewise_cuda_graph_compile -> enable_torch_compile_warmup
    PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG -> CUDA_GRAPH_CAPTURE_FAILED_MSG

Stream + ForwardContext helpers keep their names; only the home moved.
"""

from __future__ import annotations

# Capture flag + ForwardContext (new home).
from sglang.srt.model_executor.cuda_graph_backend_utils.piecewise_cuda_graph.context_manager import (  # noqa: F401
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    ForwardContext,
    enable_cuda_graph_capture,
    get_forward_context,
    is_in_cuda_graph_capture,
    set_forward_context,
)

# Torch-compile warmup flag + capture stream (new home).
from sglang.srt.compilation.compile_phase import (  # noqa: F401
    enable_torch_compile_warmup,
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
    set_pcg_capture_stream,
)

# Legacy-name aliases. Audited callers should import the renamed symbols
# directly from their new homes.
is_in_piecewise_cuda_graph = is_in_cuda_graph_capture
enable_piecewise_cuda_graph = enable_cuda_graph_capture
is_in_pcg_torch_compile = is_in_torch_compile_warmup
enable_piecewise_cuda_graph_compile = enable_torch_compile_warmup
PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG = CUDA_GRAPH_CAPTURE_FAILED_MSG
