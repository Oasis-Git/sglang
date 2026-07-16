"""Piecewise CUDA graph utilities — shared between Breakable and tc_piecewise backends.

Public API:
  - is_in_tc_piecewise_cuda_graph() — true while inside any piecewise capture.
  - enable_tc_piecewise_cuda_graph() — context manager that toggles the flag.
  - TCPCG_FAILURE_HINT — backend-switch suggestion plugged into
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG by the prefill runner.

The runner forward context (RunnerForwardContext + set_forward_context +
get_forward_context) lives in
sglang.srt.model_executor.runner_utils.forward_context.

The torch.compile-warmup flag (is_in_torch_compile_warmup) lives in
sglang.srt.compilation.compile_phase — it is torch.compile-internal,
not piecewise-shared.
"""

from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (  # noqa: F401
    TCPCG_FAILURE_HINT,
    enable_tc_piecewise_cuda_graph,
    is_in_tc_piecewise_cuda_graph,
)
