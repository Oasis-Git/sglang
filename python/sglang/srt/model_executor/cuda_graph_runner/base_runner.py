"""Shared scaffolding for the prefill and decode CUDA graph runners.

The phase-specific subclasses (``DecodeCudaGraphRunner``,
``PrefillCudaGraphRunner``) own their own buffer dataclasses, capture
forward-mode, and ``can_run`` logic. This base contributes:

- ``freeze_gc`` — gc-freeze context used during capture
- ``get_batch_sizes_to_capture`` — bucket-sizing helper for decode
- abstract methods describing the contract a phase runner must fulfil
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """Optimize garbage collection during CUDA graph capture.

    Clean up first, then freeze remaining objects from being included in
    future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def get_batch_sizes_to_capture(
    model_runner: "ModelRunner", num_tokens_per_bs: int = 1
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters server_args.cuda_graph_bs by attention-tp/cp alignment
    constraints and clamps to req_to_token_pool.size.
    """
    from sglang.srt.layers.dp_attention import (
        get_attention_cp_size,
        get_attention_tp_size,
    )
    from sglang.srt.utils import require_gathered_buffer

    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    if mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()

    num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
    if max(capture_bs) > num_max_requests:
        capture_bs += [num_max_requests]

    capture_bs = [bs for bs in capture_bs if bs * num_tokens_per_bs % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


class BaseCudaGraphRunner(ABC):
    """Abstract base for phase-specific cuda-graph runners.

    A subclass implements one of the two phases (``DecodeCudaGraphRunner``
    or ``PrefillCudaGraphRunner``) and plugs in a backend that handles
    capture/replay mechanics. The runner orchestrates: bucket selection,
    static buffer population, attention metadata init, replay dispatch,
    and output slicing. The backend handles only "given a populated
    forward_batch, run the captured artifact for this shape".
    """

    @abstractmethod
    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        """Decide whether ``forward_batch`` should go through cuda graph
        replay (vs falling back to eager forward). Subclasses should AND
        their phase-level checks with ``self.backend.can_run(fb)``.
        """

    @abstractmethod
    def capture(self) -> None:
        """Outer capture loop. Iterates over shapes, calls
        ``self._capture_one`` for each.
        """

    @abstractmethod
    def replay(
        self,
        forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        """Dispatch one batch through cuda graph replay."""
