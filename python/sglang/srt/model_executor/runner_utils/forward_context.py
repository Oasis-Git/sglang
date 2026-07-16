"""Runner forward-context propagation.

RunnerForwardContext is the cross-cutting state the prefill runners set
around every model forward they drive — capture and replay for all
prefill CUDA-graph backends (tc_piecewise, breakable, full) as well as
HIP's eager PCG fallback. Attention/MoE/quantization/model code reads it
via get_forward_context() to reach the current ForwardBatch and
per-layer metadata without threading arguments through every call site.

Distinct from the per-forward-call
sglang.srt.model_executor.forward_context.AttnForwardContext, which
propagates the attention backend; this context carries runner-owned
capture/replay state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class RunnerForwardContext:
    forward_batch: Optional[ForwardBatch] = None
    attention_layers: Optional[List[Any]] = field(default=None)
    quant_config: Any = None
    moe_layers: Optional[List[Any]] = field(default=None)
    moe_fusions: Optional[List[Any]] = field(default=None)
    dsa_indexers: Optional[List[Any]] = field(default=None)
    num_tokens: Optional[int] = None
    raw_num_tokens: Optional[int] = None


_runner_forward_context: Optional[RunnerForwardContext] = None


def get_forward_context() -> Optional[RunnerForwardContext]:
    return _runner_forward_context


@contextmanager
def set_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
    dsa_indexers: Optional[List[Any]] = None,
    num_tokens: Optional[int] = None,
    raw_num_tokens: Optional[int] = None,
):
    global _runner_forward_context
    _runner_forward_context = RunnerForwardContext(
        forward_batch=forward_batch,
        attention_layers=attention_layers,
        quant_config=quant_config,
        moe_layers=moe_layers,
        moe_fusions=moe_fusions,
        dsa_indexers=dsa_indexers,
        num_tokens=num_tokens,
        raw_num_tokens=raw_num_tokens,
    )
    try:
        yield
    finally:
        _runner_forward_context = None
