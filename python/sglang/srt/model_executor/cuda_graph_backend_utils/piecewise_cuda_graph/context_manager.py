"""CUDA graph capture context manager + forward-context propagation.

Owns two pieces of cross-cutting state used by *every* piecewise-style
backend (currently breakable + tcpcg):

* ``_in_cuda_graph_capture`` — a process-global flag set true while we
  are inside the capture or replay window of a piecewise CUDA graph.
  Read by model code that needs to take the static-buffer / fixed-shape
  branch. See ``refactor/plan.md`` §6.5 for the full semantics.
* ``ForwardContext`` — a dataclass propagated across attention/MoE
  layers during capture and replay so that submodules can reach the
  current ``ForwardBatch`` and per-layer metadata without threading
  arguments through every call site.

This module deliberately does **not** own torch.compile-specific state
(warmup flag, capture stream); those live in ``compilation/compile_phase.py``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_in_cuda_graph_capture = False


def is_in_cuda_graph_capture() -> bool:
    """True while inside the capture or replay window of any piecewise
    CUDA graph backend (BCG or tcpcg today; full backend does not toggle
    this flag and does not need it). See plan §6.5.
    """
    return _in_cuda_graph_capture


@contextmanager
def enable_cuda_graph_capture():
    """Mark the enclosed scope as "we are inside a piecewise CUDA graph
    capture/replay". Sets ``_in_cuda_graph_capture`` true for the duration.

    Errors during capture surface a hint that lets users disable the
    feature while filing a bug.
    """
    global _in_cuda_graph_capture
    _in_cuda_graph_capture = True
    try:
        yield
    except Exception as e:
        logger.error(
            "Piecewise CUDA Graph failed with error: %s\n%s",
            e,
            CUDA_GRAPH_CAPTURE_FAILED_MSG,
        )
        raise
    finally:
        _in_cuda_graph_capture = False


@dataclass
class ForwardContext:
    def __init__(self):
        self.forward_batch = None
        self.attention_layers = None
        self.quant_config = None
        self.moe_layers = None
        self.moe_fusions = None

    def set_forward_batch(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def set_attention_layers(self, layers: List[Any]):
        self.attention_layers = layers

    def set_quant_config(self, quant_config: Any):
        self.quant_config = quant_config

    def set_moe_layers(self, layers: List[Any]):
        self.moe_layers = layers

    def set_moe_fusions(self, fusions: List[Any]):
        self.moe_fusions = fusions


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        return None
    return _forward_context


@contextmanager
def set_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
):
    global _forward_context
    _forward_context = ForwardContext()
    _forward_context.set_forward_batch(forward_batch)
    _forward_context.set_attention_layers(attention_layers)
    _forward_context.set_quant_config(quant_config)
    _forward_context.set_moe_layers(moe_layers)
    _forward_context.set_moe_fusions(moe_fusions)
    try:
        yield
    finally:
        _forward_context = None


CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Piecewise CUDA Graph is enabled by default as an experimental feature.\n"
    "To work around this error, add --disable-piecewise-cuda-graph to your launch command.\n"
    "Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose"
)
