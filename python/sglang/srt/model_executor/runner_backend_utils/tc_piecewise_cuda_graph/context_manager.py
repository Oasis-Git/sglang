"""CUDA graph capture context manager.

Owns the process-global flag used by *every* piecewise-style backend
(currently breakable + tc_piecewise):

* _in_tc_piecewise_cuda_graph — a process-global flag set true while we
  are inside the capture or replay window of a piecewise CUDA graph.
  Read by model code that needs to take the static-buffer / fixed-shape
  branch. See refactor/plan.md §6.5 for the full semantics.

The runner forward context (RunnerForwardContext + set_forward_context +
get_forward_context) lives in
sglang.srt.model_executor.runner_utils.forward_context — it is set by
every prefill runner, not only the piecewise backends.

This module deliberately does **not** own torch.compile-specific state
(warmup flag, capture stream); those live in compilation/compile_phase.py.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

logger = logging.getLogger(__name__)

_in_tc_piecewise_cuda_graph = False


def is_in_tc_piecewise_cuda_graph() -> bool:
    """True while inside tc_piecewise CUDA graph capture/replay."""
    return _in_tc_piecewise_cuda_graph


@contextmanager
def enable_tc_piecewise_cuda_graph():
    """Mark the enclosed scope as inside a tc_piecewise CUDA graph
    capture/replay. Any exception raised inside is logged with the
    PCG-specific failure hint, then re-raised for the caller to handle.
    """
    global _in_tc_piecewise_cuda_graph
    _in_tc_piecewise_cuda_graph = True
    try:
        yield
    except Exception as exc:
        msg = PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG.format(
            backend=Backend.TC_PIECEWISE, suggestions=TCPCG_FAILURE_HINT
        )
        logger.error(f"{type(exc).__name__}: {exc}\n{msg}")
        raise
    finally:
        _in_tc_piecewise_cuda_graph = False


TCPCG_FAILURE_HINT = (
    "1. change to breakable by --cuda-graph-backend-prefill=breakable\n"
    "2. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled\n"
    "3. if it is an OOM problem, set --mem-fraction-static to a smaller value "
    "(e.g., 0.8 or 0.7) or set --cuda-graph-max-bs-prefill to a smaller value "
    "(e.g., 2048)\n"
)
